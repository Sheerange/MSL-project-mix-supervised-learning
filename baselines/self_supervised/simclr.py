"""
Baseline: SimCLR - A Simple Framework for Contrastive Learning (Chen et al., ICML 2020)
自监督对比学习方法
- 在训练集上进行监督学习
- 在验证集上进行自监督对比学习(无标签)
- 使用数据增强构建正样本对

训练逻辑：每个epoch = 训练集监督训练 + 验证集SimCLR对比学习

Paper: https://arxiv.org/abs/2002.05709
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
parser.add_argument("--temperature", type=float, default=0.5)  # 对比学习的温度参数
parser.add_argument("--feature_dim", type=int, default=128)  # 投影头输出维度
parser.add_argument("--lambda_u", type=float, default=1.0)  # 无标签损失权重
args = parser.parse_args()


class GaussianBlur:
    """高斯模糊增强"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class SimCLRTransform:
    """SimCLR的数据增强策略"""
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
        # 返回两个不同的增强视图
        return self.transform(x), self.transform(x)


class SimCLRDataset(torch.utils.data.Dataset):
    """包装数据集以返回两个增强视图"""
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.dataset[index]
        # 对PIL图像应用SimCLR转换
        img1, img2 = self.transform(img)
        return img1, img2, label

    def __len__(self):
        return len(self.dataset)


class ProjectionHead(nn.Module):
    """SimCLR的投影头(MLP)"""
    def __init__(self, in_dim, hidden_dim=2048, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


def nt_xent_loss(z_i, z_j, temperature=0.5):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent)
    z_i, z_j: [batch_size, feature_dim] - 两个增强视图的特征
    """
    batch_size = z_i.shape[0]

    # 归一化
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)

    # 拼接所有样本: [2*batch_size, feature_dim]
    z = torch.cat([z_i, z_j], dim=0)

    # 计算相似度矩阵: [2*batch_size, 2*batch_size]
    sim = torch.mm(z, z.T) / temperature

    # 创建正样本对的mask
    # 对于样本i，它的正样本是i+batch_size (或i-batch_size)
    sim_i_j = torch.diag(sim, batch_size)  # [batch_size]
    sim_j_i = torch.diag(sim, -batch_size)  # [batch_size]

    # 正样本的相似度
    positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0).reshape(2 * batch_size, 1)

    # 负样本的相似度 (移除对角线)
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    negative_samples = sim[~mask].reshape(2 * batch_size, -1)

    # 构建logits
    logits = torch.cat([positive_samples, negative_samples], dim=1)

    # labels: 正样本在索引0
    labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)

    loss = F.cross_entropy(logits, labels)
    return loss


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
    print(f"Baseline: SimCLR (Self-Supervised Contrastive Learning)")
    print(f"batch_size: {batch_size}")
    print(f"Dataset: {dataset}, Model: {model_name}")
    print(f"Temperature: {args.temperature}")
    print(f"Feature dim: {args.feature_dim}")
    print(f"Lambda_u: {args.lambda_u}")
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

    # 验证集（用于SimCLR对比学习，需要PIL格式）
    val_dataset_base = datasets.ImageFolder(root=os.path.join(image_path, 'val'))
    val_dataset_simclr = SimCLRDataset(val_dataset_base, SimCLRTransform())
    val_loader_simclr = torch.utils.data.DataLoader(
        val_dataset_simclr, batch_size=batch_size,
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

    # 初始化编码器（用于监督学习）
    encoder = EfficientNet.from_name('efficientnet-b3', num_classes=1000).to('cuda')
    encoder.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    encoder._fc = nn.Linear(channel, num_classes).to('cuda')
    encoder.aug = False

    # 添加投影头（用于SimCLR对比学习）
    projection_head = ProjectionHead(channel, hidden_dim=2048, out_dim=args.feature_dim).to('cuda')

    # 创建临时的特征提取器（encoder去掉分类头）
    encoder_feat = EfficientNet.from_name('efficientnet-b3', num_classes=1000).to('cuda')
    encoder_feat.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    encoder_feat._fc = nn.Identity()
    encoder_feat.aug = False

    criterion = nn.CrossEntropyLoss()

    # 优化器
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

    # SimCLR优化器（优化encoder特征提取部分和投影头）
    simclr_optimizer = torch.optim.Adam(
        list(encoder_feat.parameters()) + list(projection_head.parameters()),
        lr=args.lr * 0.1  # SimCLR学习率设置小一点
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

        # 同步encoder_feat的权重（除了分类头）
        with torch.no_grad():
            for (name_enc, param_enc), (name_feat, param_feat) in zip(
                encoder.named_parameters(), encoder_feat.named_parameters()
            ):
                if '_fc' not in name_enc:  # 跳过分类头
                    param_feat.data.copy_(param_enc.data)

        # ===== 阶段2: 验证集SimCLR对比学习 =====
        encoder_feat.train()
        projection_head.train()

        simclr_loss_epoch = []
        val_bar = tqdm(val_loader_simclr, desc=f"Epoch {epoch+1}/{epochs} - SimCLR")

        for img1, img2, _ in val_bar:
            img1 = img1.to('cuda')
            img2 = img2.to('cuda')

            simclr_optimizer.zero_grad()

            # 提取特征
            feat1, _ = encoder_feat(img1)
            feat2, _ = encoder_feat(img2)

            # 通过投影头
            z1 = projection_head(feat1)
            z2 = projection_head(feat2)

            # 计算NT-Xent损失
            loss = nt_xent_loss(z1, z2, temperature=args.temperature)

            simclr_loss_epoch.append(loss.item())
            loss.backward()
            simclr_optimizer.step()

            val_bar.set_description(
                f"Epoch {epoch+1}/{epochs} - SimCLR Loss: {loss.item():.3f}"
            )

        avg_simclr_loss = sum(simclr_loss_epoch) / len(simclr_loss_epoch)

        # 同步回encoder的权重（除了分类头）
        with torch.no_grad():
            for (name_enc, param_enc), (name_feat, param_feat) in zip(
                encoder.named_parameters(), encoder_feat.named_parameters()
            ):
                if '_fc' not in name_enc:
                    param_enc.data.copy_(param_feat.data)

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
        print(f"  Supervised Loss={avg_supervised_loss:.4f}, SimCLR Loss={avg_simclr_loss:.4f}")
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
