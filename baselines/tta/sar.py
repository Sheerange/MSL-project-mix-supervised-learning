"""
Baseline: SAR - Sharpness-Aware and Reliable Test-Time Adaptation (Niu et al., ICLR 2023)
改进TENT的测试时适应方法，通过SAM优化器提升泛化性
- 使用Sharpness-Aware Minimization (SAM)找到平坦的最小值
- 可靠性估计：过滤低置信度样本
- 重要性加权：对高置信度预测给予更多权重

训练逻辑：每个epoch = 训练集监督训练 + 验证集SAR适应

Paper: https://arxiv.org/abs/2301.13001
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
import copy

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--nw", type=int, default=20)
parser.add_argument("--lr", type=float, default=5e-3)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--dataset", type=str, default='CUB200')
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--sar_lr", type=float, default=1e-3)  # SAR的学习率
parser.add_argument("--rho", type=float, default=0.05)  # SAM的扰动半径
parser.add_argument("--ent_threshold", type=float, default=0.7)  # 熵阈值（过滤不确定样本）
args = parser.parse_args()


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization优化器"""
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """第一步: 计算扰动并应用"""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # 爬到"陡峭"的点
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """第二步: 恢复原始参数并更新"""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # 回到原始点
        self.base_optimizer.step()  # 用"陡峭"点的梯度更新
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError("SAR should use first_step() and second_step()")

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm


def configure_model_for_sar(model):
    """配置模型用于SAR: BatchNorm设置为训练模式"""
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def collect_bn_params(model):
    """收集BatchNorm的affine参数"""
    params = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            for p in m.parameters():
                if p.requires_grad:
                    params.append(p)
    return params


def entropy_loss(logits, reduction='mean'):
    """计算预测的熵"""
    probs = F.softmax(logits, dim=1)
    probs = torch.clamp(probs, min=1e-7)
    entropy = -torch.sum(probs * torch.log(probs), dim=1)
    if reduction == 'mean':
        return entropy.mean()
    elif reduction == 'none':
        return entropy
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


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
    print(f"Baseline: SAR (Sharpness-Aware Test-Time Adaptation)")
    print(f"batch_size: {batch_size}")
    print(f"Dataset: {dataset}, Model: {model_name}")
    print(f"SAR learning rate: {args.sar_lr}")
    print(f"SAM rho: {args.rho}")
    print(f"Entropy threshold: {args.ent_threshold}")
    print('*' * 50)

    # 加载数据
    if dataset == 'tiny_ImageNet':
        data_root = os.path.abspath(os.path.join(os.getcwd(), '../datasets/tiny-imagenet-200'))
    elif dataset == 'CUB200':
        data_root = os.path.abspath(os.path.join(os.getcwd(), '../datasets/CUB-200(+test)'))

    image_path = os.path.join(data_root)

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'train'),
                                       transform=data_transform['train'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=nw)

    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                      transform=data_transform['val'])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=nw)

    # 用于SAR适应的验证集（带数据增强）
    val_dataset_aug = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                          transform=data_transform['train'])
    val_loader_aug = torch.utils.data.DataLoader(val_dataset_aug, batch_size=batch_size,
                                                 shuffle=True, num_workers=nw)

    test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                       transform=data_transform['val'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=nw)

    print(f"Training: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # 初始化模型
    net = EfficientNet.from_name('efficientnet-b3', num_classes=1000).to('cuda')
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    net._fc = nn.Linear(channel, num_classes).to('cuda')
    net.aug = False

    criterion = nn.CrossEntropyLoss()

    # 主优化器（用于训练集监督学习）
    optimizer = torch.optim.SGD(net.parameters(), lr=base_lr,
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

    # SAR优化器（只优化BatchNorm参数，用于验证集）
    bn_params = collect_bn_params(net)
    sar_optimizer = SAM(bn_params, torch.optim.Adam, lr=args.sar_lr, rho=args.rho)

    best_acc_val = 0.0
    best_acc_test = 0.0

    for epoch in range(epochs):
        # ===== 阶段1: 训练集监督训练 =====
        net.train()

        supervised_loss_epoch = []
        num_correct = 0
        num_total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Supervised")

        for imgs, labels in train_bar:
            imgs = imgs.to('cuda')
            labels = labels.to('cuda')

            optimizer.zero_grad()
            outputs, _ = net(imgs)
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

        # ===== 阶段2: 验证集SAR适应 =====
        # 配置模型为SAR模式
        net = configure_model_for_sar(net)

        sar_loss_epoch = []
        filter_ratio_list = []
        val_bar = tqdm(val_loader_aug, desc=f"Epoch {epoch+1}/{epochs} - SAR")

        for imgs, _ in val_bar:
            imgs = imgs.to('cuda')

            # SAR核心: 使用SAM进行两步优化
            # 第一步: 计算扰动
            outputs, _ = net(imgs)
            ent_per_sample = entropy_loss(outputs, reduction='none')

            # 可靠性过滤: 只适应低熵(高置信度)的样本
            max_ent = np.log(num_classes)
            ent_normalized = ent_per_sample / max_ent
            reliable_mask = (ent_normalized < args.ent_threshold).float()

            # 计算加权熵损失
            if reliable_mask.sum() > 0:
                loss = (ent_per_sample * reliable_mask).sum() / reliable_mask.sum()
                loss.backward()
                sar_optimizer.first_step(zero_grad=True)

                # 第二步: 在扰动点计算梯度并更新
                outputs2, _ = net(imgs)
                ent_per_sample2 = entropy_loss(outputs2, reduction='none')
                loss2 = (ent_per_sample2 * reliable_mask).sum() / reliable_mask.sum()
                loss2.backward()
                sar_optimizer.second_step(zero_grad=True)

                sar_loss_epoch.append(loss.item())
            else:
                sar_loss_epoch.append(0.0)

            filter_ratio_list.append(reliable_mask.mean().item())

            val_bar.set_description(
                f"Epoch {epoch+1}/{epochs} - SAR Loss: {loss.item() if reliable_mask.sum() > 0 else 0:.3f}, "
                f"Filter: {reliable_mask.mean().item():.2f}"
            )

        avg_sar_loss = sum(sar_loss_epoch) / len(sar_loss_epoch) if sar_loss_epoch else 0
        avg_filter_ratio = sum(filter_ratio_list) / len(filter_ratio_list) if filter_ratio_list else 0

        # 评估
        val_acc, val_acc5 = _accuracy_(net, val_loader)
        test_acc, test_acc5 = _accuracy_(net, test_loader)

        if val_acc > best_acc_val:
            print(f"\n{'='*50}")
            print("Better model found!")
            print(f"{'='*50}")
            best_acc_val = val_acc
            best_acc_test = test_acc

        print(f"\nEpoch {epoch+1}:")
        print(f"  Supervised Loss={avg_supervised_loss:.4f}, SAR Loss={avg_sar_loss:.4f}")
        print(f"  Filter Ratio (reliable samples)={avg_filter_ratio:.2f}")
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
