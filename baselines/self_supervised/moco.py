"""
Baseline: MoCo - Momentum Contrast for Unsupervised Visual Representation Learning (He et al., CVPR 2020)
自监督对比学习方法，使用动量编码器和队列
- 在训练集上进行监督学习
- 在验证集上进行自监督对比学习(无标签)
- 使用动量编码器维护大规模负样本队列

训练逻辑：每个epoch = 训练集监督训练 + 验证集MoCo对比学习

Paper: https://arxiv.org/abs/1911.05722
"""
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
from torchvision import transforms, datasets
from models.efficientnet_pytorch import EfficientNet
import time
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn import functional as F
from PIL import ImageFilter
import random

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--nw", type=int, default=20)
parser.add_argument("--lr", type=float, default=5e-3)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--dataset", type=str, default='CUB200')
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--temperature", type=float, default=0.07)  # MoCo的温度参数
parser.add_argument("--feature_dim", type=int, default=128)  # 投影头输出维度
parser.add_argument("--momentum", type=float, default=0.999)  # 动量编码器的更新系数
parser.add_argument("--queue_size", type=int, default=4096)  # 负样本队列大小
args = parser.parse_args()


class GaussianBlur:
    """高斯模糊增强"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class MoCoTransform:
    """MoCo的数据增强策略"""
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.49139968, 0.48215841, 0.44653091],
                               [0.24703223, 0.24348513, 0.26158784])
        ])

    def __call__(self, x):
        # query和key使用不同的增强
        return self.transform(x), self.transform(x)


class MoCoDataset(torch.utils.data.Dataset):
    """包装数据集以返回两个增强视图"""
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.dataset[index]
        img_q, img_k = self.transform(img)
        return img_q, img_k, label

    def __len__(self):
        return len(self.dataset)


class ProjectionHead(nn.Module):
    """MoCo的投影头(MLP)"""
    def __init__(self, in_dim, hidden_dim=2048, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class MoCo(nn.Module):
    """
    MoCo模型
    包含query编码器、key编码器(动量)和队列
    """
    def __init__(self, encoder_q, encoder_k, proj_q, proj_k,
                 dim=128, K=4096, m=0.999, T=0.07):
        super().__init__()

        self.K = K  # 队列大小
        self.m = m  # 动量系数
        self.T = T  # 温度

        # query编码器
        self.encoder_q = encoder_q
        self.proj_q = proj_q

        # key编码器(动量)
        self.encoder_k = encoder_k
        self.proj_k = proj_k

        # 初始化key编码器参数为query编码器参数
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # 创建队列
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """动量更新key编码器"""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """更新队列"""
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        # 替换队列中的元素
        if ptr + batch_size <= self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            # 循环队列
            remaining = self.K - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T

        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """
        im_q: query图像
        im_k: key图像
        """
        # query特征
        q, _ = self.encoder_q(im_q)
        q = self.proj_q(q)
        q = F.normalize(q, dim=1)

        # key特征(no gradient)
        with torch.no_grad():
            # 动量更新
            self._momentum_update_key_encoder()

            k, _ = self.encoder_k(im_k)
            k = self.proj_k(k)
            k = F.normalize(k, dim=1)

        # 计算logits
        # positive logits: [N, 1]
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: [N, K]
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: [N, 1+K]
        logits = torch.cat([l_pos, l_neg], dim=1)

        # 应用温度
        logits /= self.T

        # labels: 正样本在索引0
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # 更新队列
        self._dequeue_and_enqueue(k)

        return logits, labels


def _accuracy_(net, data_loader):
    net.eval()
    num_total = 0
    num_acc = 0
    num_acc_top5 = 0
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to('cuda')
            labels = labels.to('cuda')

            output, _ = net(imgs)
            _, pred = torch.max(output, 1)

            _, top5_pred = torch.topk(output, 5, dim=1)
            top5_correct = top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred))

            num_acc += torch.sum(pred == labels.detach_())
            num_acc_top5 += torch.sum(top5_correct)
            num_total += labels.size(0)

        LV = num_acc.detach().cpu().numpy() * 100 / num_total
        LV_top5 = num_acc_top5.detach().cpu().numpy() * 100 / num_total

    return LV, LV_top5


def main():
    # 标准数据转换
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.49139968, 0.48215841, 0.44653091],
                               [0.24703223, 0.24348513, 0.26158784])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.49139968, 0.48215841, 0.44653091],
                               [0.24703223, 0.24348513, 0.26158784])
        ]),
    }

    epochs = args.epochs
    base_lr = args.lr
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    nw = args.nw
    model_name = 'efficientnet-b3'
    model_weight_path = './efficientnet-b3.pth'
    channel = 1536
    num_classes = 200
    dataset = args.dataset

    print('*' * 50)
    print(f"Baseline: MoCo (Momentum Contrast)")
    print(f"batch_size: {batch_size}")
    print(f"Dataset: {dataset}, Model: {model_name}")
    print(f"Temperature: {args.temperature}")
    print(f"Feature dim: {args.feature_dim}")
    print(f"Momentum: {args.momentum}")
    print(f"Queue size: {args.queue_size}")
    print('*' * 50)

    # 加载数据
    if dataset == 'tiny_ImageNet':
        data_root = os.path.abspath(os.path.join(os.getcwd(), '../datasets/tiny-imagenet-200'))
    elif dataset == 'CUB200':
        data_root = os.path.abspath(os.path.join(os.getcwd(), '../datasets/CUB-200(+test)'))

    image_path = os.path.join(data_root)

    # 训练集（标准增强）
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'train'),
                                       transform=data_transform['train'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=nw)

    # 验证集（用于MoCo对比学习，需要PIL格式）
    val_dataset_base = datasets.ImageFolder(root=os.path.join(image_path, 'val'))
    val_dataset_moco = MoCoDataset(val_dataset_base, MoCoTransform())
    val_loader_moco = torch.utils.data.DataLoader(
        val_dataset_moco, batch_size=batch_size,
        shuffle=True, num_workers=nw, drop_last=True
    )

    # 验证集和测试集（用于评估）
    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                      transform=data_transform['val'])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=nw)

    test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                       transform=data_transform['val'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=nw)

    print(f"Training: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # 初始化主编码器（用于监督学习）
    encoder = EfficientNet.from_name('efficientnet-b3', num_classes=1000).to('cuda')
    encoder.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    encoder._fc = nn.Linear(channel, num_classes).to('cuda')
    encoder.aug = False

    # 初始化query编码器（用于MoCo，去掉分类头）
    encoder_q = EfficientNet.from_name('efficientnet-b3', num_classes=1000).to('cuda')
    encoder_q.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    encoder_q._fc = nn.Identity()
    encoder_q.aug = False

    # 初始化key编码器(动量)
    encoder_k = EfficientNet.from_name('efficientnet-b3', num_classes=1000).to('cuda')
    encoder_k.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    encoder_k._fc = nn.Identity()
    encoder_k.aug = False

    # 投影头
    proj_q = ProjectionHead(channel, hidden_dim=2048, out_dim=args.feature_dim).to('cuda')
    proj_k = ProjectionHead(channel, hidden_dim=2048, out_dim=args.feature_dim).to('cuda')

    # 创建MoCo模型
    moco = MoCo(
        encoder_q, encoder_k, proj_q, proj_k,
        dim=args.feature_dim,
        K=args.queue_size,
        m=args.momentum,
        T=args.temperature
    ).to('cuda')

    criterion = nn.CrossEntropyLoss()

    # 监督学习优化器
    optimizer = torch.optim.SGD(encoder.parameters(), lr=base_lr,
                               weight_decay=weight_decay, momentum=0)

    schedule = OneCycleLR(
        optimizer,
        max_lr=base_lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.3,
        anneal_strategy='cos',
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25,
        final_div_factor=1000
    )

    # MoCo优化器（只优化query编码器和投影头）
    moco_optimizer = torch.optim.Adam(
        list(moco.encoder_q.parameters()) + list(moco.proj_q.parameters()),
        lr=args.lr * 0.1
    )

    best_acc_val = 0.0
    best_acc_test = 0.0

    for epoch in range(epochs):
        # ===== 阶段1: 训练集监督训练 =====
        encoder.train()

        supervised_loss_epoch = []
        num_correct = 0
        num_total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Supervised")

        for imgs, labels in train_bar:
            imgs = imgs.to('cuda')
            labels = labels.to('cuda')

            optimizer.zero_grad()
            outputs, _ = encoder(imgs)
            loss = criterion(outputs, labels)

            _, pred = torch.max(outputs, dim=1)
            num_correct += torch.sum(pred == labels.detach_())
            num_total += labels.size(0)

            supervised_loss_epoch.append(loss.item())
            loss.backward()
            optimizer.step()
            schedule.step()

            train_bar.set_description(
                f"Epoch {epoch+1}/{epochs} - Supervised Loss: {loss.item():.3f}"
            )

        train_acc = num_correct.detach().cpu().numpy() * 100 / num_total
        avg_supervised_loss = sum(supervised_loss_epoch) / len(supervised_loss_epoch)

        # 同步encoder_q的权重（除了分类头）
        with torch.no_grad():
            for (name_enc, param_enc), (name_q, param_q) in zip(
                encoder.named_parameters(), moco.encoder_q.named_parameters()
            ):
                if '_fc' not in name_enc:
                    param_q.data.copy_(param_enc.data)

        # ===== 阶段2: 验证集MoCo对比学习 =====
        moco.train()

        moco_loss_epoch = []
        val_bar = tqdm(val_loader_moco, desc=f"Epoch {epoch+1}/{epochs} - MoCo")

        for img_q, img_k, _ in val_bar:
            img_q = img_q.to('cuda')
            img_k = img_k.to('cuda')

            moco_optimizer.zero_grad()

            # MoCo forward
            logits, target_labels = moco(img_q, img_k)

            # 对比损失
            loss = criterion(logits, target_labels)

            moco_loss_epoch.append(loss.item())
            loss.backward()
            moco_optimizer.step()

            val_bar.set_description(
                f"Epoch {epoch+1}/{epochs} - MoCo Loss: {loss.item():.3f}"
            )

        avg_moco_loss = sum(moco_loss_epoch) / len(moco_loss_epoch)

        # 同步回encoder的权重（除了分类头）
        with torch.no_grad():
            for (name_enc, param_enc), (name_q, param_q) in zip(
                encoder.named_parameters(), moco.encoder_q.named_parameters()
            ):
                if '_fc' not in name_enc:
                    param_enc.data.copy_(param_q.data)

        # 评估
        val_acc, val_acc5 = _accuracy_(encoder, val_loader)
        test_acc, test_acc5 = _accuracy_(encoder, test_loader)

        if val_acc > best_acc_val:
            print(f"\n{'='*50}")
            print("Better model found!")
            print(f"{'='*50}")
            best_acc_val = val_acc
            best_acc_test = test_acc

        print(f"\nEpoch {epoch+1}:")
        print(f"  Supervised Loss={avg_supervised_loss:.4f}, MoCo Loss={avg_moco_loss:.4f}")
        print(f"  Train Acc={train_acc:.2f}%")
        print(f"  Val Acc={val_acc:.2f}%, Test Acc={test_acc:.2f}%, "
              f"Val Top5={val_acc5:.2f}%, Test Top5={test_acc5:.2f}%")

    print(f"\n{'='*50}")
    print(f"Training Finished!")
    print(f"Best Val Acc: {best_acc_val:.2f}%")
    print(f"Best Test Acc: {best_acc_test:.2f}%")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
