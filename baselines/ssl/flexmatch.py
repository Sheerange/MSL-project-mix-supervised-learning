"""
Baseline: FlexMatch (Zhang et al., NeurIPS 2021)
FlexMatch改进了FixMatch，使用Curriculum Pseudo Labeling (CPL)
- 动态调整每个类别的置信度阈值
- 根据学习状态自适应调整阈值
- 对于难学习的类别使用更低的阈值

Paper: https://arxiv.org/abs/2110.08263
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
from PIL import Image, ImageEnhance, ImageOps
import random

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--nw", type=int, default=20)
parser.add_argument("--lr", type=float, default=5e-3)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--dataset", type=str, default='CUB200')
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--thresh_warmup", type=float, default=0.95)  # warmup阶段的阈值
parser.add_argument("--lambda_u", type=float, default=1.0)
parser.add_argument("--warmup_epochs", type=int, default=0)
args = parser.parse_args()


class RandAugment:
    """简化版的RandAugment用于强数据增强"""
    def __init__(self, n=2, m=10):
        self.n = n
        self.m = m

    def __call__(self, img):
        ops = [
            self.autocontrast,
            self.equalize,
            self.rotate,
            self.solarize,
            self.color,
            self.contrast,
            self.brightness,
            self.sharpness,
        ]

        for _ in range(self.n):
            op = random.choice(ops)
            img = op(img)
        return img

    def autocontrast(self, img):
        return ImageOps.autocontrast(img)

    def equalize(self, img):
        return ImageOps.equalize(img)

    def rotate(self, img):
        degrees = (self.m / 10) * 30
        if random.random() > 0.5:
            degrees = -degrees
        return img.rotate(degrees)

    def solarize(self, img):
        threshold = int((self.m / 10) * 256)
        return ImageOps.solarize(img, threshold)

    def color(self, img):
        factor = (self.m / 10) * 1.8 + 0.1
        return ImageEnhance.Color(img).enhance(factor)

    def contrast(self, img):
        factor = (self.m / 10) * 1.8 + 0.1
        return ImageEnhance.Contrast(img).enhance(factor)

    def brightness(self, img):
        factor = (self.m / 10) * 1.8 + 0.1
        return ImageEnhance.Brightness(img).enhance(factor)

    def sharpness(self, img):
        factor = (self.m / 10) * 1.8 + 0.1
        return ImageEnhance.Sharpness(img).enhance(factor)


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
    # 弱增强
    weak_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.49139968, 0.48215841, 0.44653091],
                           [0.24703223, 0.24348513, 0.26158784])
    ])

    # 强增强
    strong_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        RandAugment(n=2, m=10),
        transforms.ToTensor(),
        transforms.Normalize([0.49139968, 0.48215841, 0.44653091],
                           [0.24703223, 0.24348513, 0.26158784])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.49139968, 0.48215841, 0.44653091],
                           [0.24703223, 0.24348513, 0.26158784])
    ])

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
    print(f"Baseline: FlexMatch (Curriculum Pseudo Labeling)")
    print(f"batch_size: {batch_size}")
    print(f"Dataset: {dataset}, Model: {model_name}")
    print(f"Warmup threshold: {args.thresh_warmup}")
    print(f"Lambda_u: {args.lambda_u}")
    print('*' * 50)

    # 加载数据
    if dataset == 'tiny_ImageNet':
        data_root = os.path.abspath(os.path.join(os.getcwd(), '../datasets/tiny-imagenet-200'))
    elif dataset == 'CUB200':
        data_root = os.path.abspath(os.path.join(os.getcwd(), '../datasets/CUB-200(+test)'))

    image_path = os.path.join(data_root)

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'train'),
                                       transform=weak_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=nw)

    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                      transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=nw)

    val_dataset_weak = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                           transform=weak_transform)
    val_loader_weak = torch.utils.data.DataLoader(val_dataset_weak, batch_size=batch_size,
                                                  shuffle=True, num_workers=nw)

    val_dataset_strong = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                             transform=strong_transform)
    val_loader_strong = torch.utils.data.DataLoader(val_dataset_strong, batch_size=batch_size,
                                                    shuffle=True, num_workers=nw)

    test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                       transform=val_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=nw)

    print(f"Training: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # 初始化模型
    net = EfficientNet.from_name('efficientnet-b3', num_classes=1000).to('cuda')
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    net._fc = nn.Linear(channel, num_classes).to('cuda')
    net.aug = False

    criterion = nn.CrossEntropyLoss()
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

    best_acc_val = 0.0
    best_acc_test = 0.0

    # FlexMatch: 为每个类别维护一个动态阈值
    class_thresholds = torch.ones(num_classes).cuda() * args.thresh_warmup
    # 记录每个类别的选中次数（用于CPL）
    classwise_acc = torch.zeros(num_classes).cuda()

    for epoch in range(epochs):
        net.train()

        supervised_loss_epoch = []
        unsupervised_loss_epoch = []
        total_loss_epoch = []
        num_correct = 0
        num_total = 0
        mask_ratio_epoch = []

        # 创建迭代器
        train_iter = iter(train_loader)
        val_weak_iter = iter(val_loader_weak)
        val_strong_iter = iter(val_loader_strong)

        steps_per_epoch = min(len(train_loader), len(val_loader_weak))
        train_bar = tqdm(range(steps_per_epoch))

        # 计算当前的lambda_u
        if epoch < args.warmup_epochs:
            current_lambda_u = args.lambda_u * (epoch / max(args.warmup_epochs, 1))
        else:
            current_lambda_u = args.lambda_u

        for step in train_bar:
            # 获取有标签数据
            try:
                labeled_imgs, labeled_labels = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                labeled_imgs, labeled_labels = next(train_iter)

            # 获取无标签数据（弱增强）
            try:
                unlabeled_imgs_weak, _ = next(val_weak_iter)
            except StopIteration:
                val_weak_iter = iter(val_loader_weak)
                unlabeled_imgs_weak, _ = next(val_weak_iter)

            # 获取无标签数据（强增强）
            try:
                unlabeled_imgs_strong, _ = next(val_strong_iter)
            except StopIteration:
                val_strong_iter = iter(val_loader_strong)
                unlabeled_imgs_strong, _ = next(val_strong_iter)

            labeled_imgs = labeled_imgs.to('cuda')
            labeled_labels = labeled_labels.to('cuda')
            unlabeled_imgs_weak = unlabeled_imgs_weak.to('cuda')
            unlabeled_imgs_strong = unlabeled_imgs_strong.to('cuda')

            optimizer.zero_grad()

            # 1. 监督损失
            logits_labeled, _ = net(labeled_imgs)
            loss_supervised = criterion(logits_labeled, labeled_labels)

            # 2. 无监督损失（FlexMatch with CPL）
            with torch.no_grad():
                logits_weak, _ = net(unlabeled_imgs_weak)
                probs_weak = F.softmax(logits_weak, dim=1)
                max_probs, pseudo_labels = torch.max(probs_weak, dim=1)

                # FlexMatch: 使用每个类别的动态阈值
                mask = torch.zeros_like(max_probs)
                for c in range(num_classes):
                    class_mask = (pseudo_labels == c)
                    mask[class_mask] = (max_probs[class_mask] >= class_thresholds[c]).float()

            # 强增强预测
            logits_strong, _ = net(unlabeled_imgs_strong)
            loss_unsupervised = (F.cross_entropy(logits_strong, pseudo_labels,
                                                 reduction='none') * mask).mean()

            # 总损失
            loss = loss_supervised + current_lambda_u * loss_unsupervised

            # 统计
            _, pred = torch.max(logits_labeled, dim=1)
            num_correct += torch.sum(pred == labeled_labels.detach_())
            num_total += labeled_labels.size(0)

            supervised_loss_epoch.append(loss_supervised.item())
            unsupervised_loss_epoch.append(loss_unsupervised.item())
            total_loss_epoch.append(loss.item())
            mask_ratio_epoch.append(mask.mean().item())

            # 反向传播
            loss.backward()
            optimizer.step()
            schedule.step()

            # FlexMatch CPL: 更新每个类别的阈值
            with torch.no_grad():
                # 计算每个类别的预测准确率（使用伪标签的mask）
                for c in range(num_classes):
                    class_mask = (pseudo_labels == c) & (mask > 0)
                    if class_mask.sum() > 0:
                        # 使用移动平均更新
                        classwise_acc[c] = 0.999 * classwise_acc[c] + 0.001 * class_mask.sum().float()

                # 根据学习进度调整阈值（学习慢的类别降低阈值）
                # 使用beta分布计算动态阈值
                max_count = classwise_acc.max()
                if max_count > 0:
                    for c in range(num_classes):
                        # 学习进度 = 该类被选中次数 / 最多被选中次数
                        learning_progress = classwise_acc[c] / (max_count + 1e-6)
                        # 动态阈值: 学习慢的类别用更低的阈值
                        class_thresholds[c] = learning_progress * args.thresh_warmup + (1 - learning_progress) * 0.5

            train_bar.set_description(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Loss: {loss.item():.3f} "
                f"Mask: {mask.mean().item():.2f}"
            )

        train_acc = num_correct.detach().cpu().numpy() * 100 / num_total
        avg_total_loss = sum(total_loss_epoch) / len(total_loss_epoch)
        avg_supervised_loss = sum(supervised_loss_epoch) / len(supervised_loss_epoch)
        avg_unsupervised_loss = sum(unsupervised_loss_epoch) / len(unsupervised_loss_epoch)
        avg_mask_ratio = sum(mask_ratio_epoch) / len(mask_ratio_epoch)

        # 评估
        val_acc, val_acc5 = _accuracy_(net, val_loader)
        test_acc, test_acc5 = _accuracy_(net, test_loader)

        if val_acc > best_acc_val:
            print(f"\n{'='*50}")
            print("Better model found!")
            print(f"{'='*50}")
            best_acc_val = val_acc
            best_acc_test = test_acc

        # 显示阈值统计
        avg_threshold = class_thresholds.mean().item()
        min_threshold = class_thresholds.min().item()
        max_threshold = class_thresholds.max().item()

        print(f"\nEpoch {epoch+1}:")
        print(f"  Train Loss={avg_total_loss:.4f} "
              f"(Sup={avg_supervised_loss:.4f}, Unsup={avg_unsupervised_loss:.4f}, "
              f"Lambda_u={current_lambda_u:.3f})")
        print(f"  Mask ratio={avg_mask_ratio:.2f}, "
              f"Thresholds: avg={avg_threshold:.2f}, min={min_threshold:.2f}, max={max_threshold:.2f}")
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
