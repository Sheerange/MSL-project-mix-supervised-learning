"""
Baseline: Mean Teacher (Tarvainen & Valpola, 2017)
经典的半监督学习方法
- 维护一个教师模型(EMA)和学生模型
- 学生模型在有标签数据上进行监督学习
- 学生模型和教师模型在无标签数据(验证集)上保持一致性
- 教师模型通过EMA更新

Paper: https://arxiv.org/abs/1703.01780
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
parser.add_argument("--ema_decay", type=float, default=0.999)  # EMA衰减率
parser.add_argument("--consistency_weight", type=float, default=1.0)  # 一致性损失权重
parser.add_argument("--consistency_rampup", type=int, default=5)  # 一致性损失权重线性增长的epoch数
args = parser.parse_args()


def update_ema_variables(model, ema_model, alpha):
    """更新EMA模型参数"""
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def get_current_consistency_weight(epoch, max_weight, rampup_length):
    """获取当前epoch的一致性损失权重（线性增长）"""
    if epoch < rampup_length:
        return max_weight * (epoch / rampup_length)
    else:
        return max_weight


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

            # Top-5 accuracy
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
    print(f"Baseline: Mean Teacher")
    print(f"batch_size: {batch_size}")
    print(f"Dataset: {dataset}, Model: {model_name}")
    print(f"EMA decay: {args.ema_decay}")
    print(f"Consistency weight: {args.consistency_weight}")
    print(f"Consistency rampup: {args.consistency_rampup} epochs")
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

    # 用于一致性训练的验证集（带数据增强）
    val_dataset_aug = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                          transform=data_transform['train'])
    val_loader_aug = torch.utils.data.DataLoader(val_dataset_aug, batch_size=batch_size,
                                                shuffle=True, num_workers=nw)

    test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                       transform=data_transform['val'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=nw)

    print(f"Training: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # 初始化学生模型
    student_model = EfficientNet.from_name('efficientnet-b3', num_classes=1000).to('cuda')
    student_model.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    student_model._fc = nn.Linear(channel, num_classes).to('cuda')
    student_model.aug = False

    # 初始化教师模型（EMA模型）
    teacher_model = EfficientNet.from_name('efficientnet-b3', num_classes=1000).to('cuda')
    teacher_model.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    teacher_model._fc = nn.Linear(channel, num_classes).to('cuda')
    teacher_model.aug = False

    # 复制学生模型的参数到教师模型
    for param_t, param_s in zip(teacher_model.parameters(), student_model.parameters()):
        param_t.data.copy_(param_s.data)

    # 教师模型不需要梯度
    for param in teacher_model.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    consistency_criterion = nn.MSELoss()  # 一致性损失
    optimizer = torch.optim.SGD(student_model.parameters(), lr=base_lr,
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

    for epoch in range(epochs):
        student_model.train()
        teacher_model.train()  # 虽然不更新梯度，但需要保持train模式

        train_loss_epoch = []
        supervised_loss_epoch = []
        consistency_loss_epoch = []
        num_correct = 0
        num_total = 0

        # 获取当前epoch的一致性损失权重
        consistency_weight = get_current_consistency_weight(
            epoch, args.consistency_weight, args.consistency_rampup
        )

        # 创建训练集和验证集的迭代器
        train_iter = iter(train_loader)
        val_iter = iter(val_loader_aug)

        # 取两者中较小的长度作为epoch的步数
        steps_per_epoch = min(len(train_loader), len(val_loader_aug))

        train_bar = tqdm(range(steps_per_epoch))

        for step in train_bar:
            # 获取有标签数据（训练集）
            try:
                labeled_imgs, labeled_labels = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                labeled_imgs, labeled_labels = next(train_iter)

            # 获取无标签数据（验证集）
            try:
                unlabeled_imgs, _ = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader_aug)
                unlabeled_imgs, _ = next(val_iter)

            labeled_imgs = labeled_imgs.to('cuda')
            labeled_labels = labeled_labels.to('cuda')
            unlabeled_imgs = unlabeled_imgs.to('cuda')

            optimizer.zero_grad()

            # 1. 监督损失（有标签数据）
            student_outputs_labeled, _ = student_model(labeled_imgs)
            supervised_loss = criterion(student_outputs_labeled, labeled_labels)

            # 2. 一致性损失（无标签数据）
            student_outputs_unlabeled, _ = student_model(unlabeled_imgs)
            with torch.no_grad():
                teacher_outputs_unlabeled, _ = teacher_model(unlabeled_imgs)

            # 使用MSE计算一致性损失
            consistency_loss = consistency_criterion(
                F.softmax(student_outputs_unlabeled, dim=1),
                F.softmax(teacher_outputs_unlabeled, dim=1)
            )

            # 总损失
            loss = supervised_loss + consistency_weight * consistency_loss

            # 统计
            _, pred = torch.max(student_outputs_labeled, dim=1)
            num_correct += torch.sum(pred == labeled_labels.detach_())
            num_total += labeled_labels.size(0)

            train_loss_epoch.append(loss.item())
            supervised_loss_epoch.append(supervised_loss.item())
            consistency_loss_epoch.append(consistency_loss.item())

            # 反向传播
            loss.backward()
            optimizer.step()
            schedule.step()

            # 更新教师模型（EMA）
            update_ema_variables(student_model, teacher_model, args.ema_decay)

            train_bar.set_description(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Loss: {loss.item():.3f} "
                f"Sup: {supervised_loss.item():.3f} "
                f"Cons: {consistency_loss.item():.3f}"
            )

        train_acc = num_correct.detach().cpu().numpy() * 100 / num_total
        avg_train_loss = sum(train_loss_epoch) / len(train_loss_epoch)
        avg_supervised_loss = sum(supervised_loss_epoch) / len(supervised_loss_epoch)
        avg_consistency_loss = sum(consistency_loss_epoch) / len(consistency_loss_epoch)

        # 评估（使用学生模型）
        val_acc, val_acc5 = _accuracy_(student_model, val_loader)
        test_acc, test_acc5 = _accuracy_(student_model, test_loader)

        # 也评估教师模型
        teacher_val_acc, teacher_val_acc5 = _accuracy_(teacher_model, val_loader)
        teacher_test_acc, teacher_test_acc5 = _accuracy_(teacher_model, test_loader)

        if val_acc > best_acc_val:
            print(f"\n{'='*50}")
            print("Better model found!")
            print(f"{'='*50}")
            best_acc_val = val_acc
            best_acc_test = test_acc

        print(f"\nEpoch {epoch+1}:")
        print(f"  Train Loss={avg_train_loss:.4f} (Sup={avg_supervised_loss:.4f}, "
              f"Cons={avg_consistency_loss:.4f}, Weight={consistency_weight:.3f})")
        print(f"  Train Acc={train_acc:.2f}%")
        print(f"  Student - Val: {val_acc:.2f}%, Test: {test_acc:.2f}%, "
              f"Val Top5: {val_acc5:.2f}%, Test Top5: {test_acc5:.2f}%")
        print(f"  Teacher - Val: {teacher_val_acc:.2f}%, Test: {teacher_test_acc:.2f}%, "
              f"Val Top5: {teacher_val_acc5:.2f}%, Test Top5: {teacher_test_acc5:.2f}%")

    print(f"\n{'='*50}")
    print(f"Training Finished!")
    print(f"Best Student Val Acc: {best_acc_val:.2f}%")
    print(f"Best Student Test Acc: {best_acc_test:.2f}%")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
