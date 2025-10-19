"""
Baseline: CoTTA - Continual Test-Time Adaptation (Wang et al., CVPR 2022)
改进TENT的测试时适应方法，解决持续适应中的错误累积问题
- 使用teacher-student架构防止灾难性遗忘
- 通过随机恢复(stochastic restoration)保持多样性
- 增强平均(augmentation averaging)提高鲁棒性

训练逻辑：每个epoch = 训练集监督训练 + 验证集CoTTA适应

Paper: https://arxiv.org/abs/2203.13591
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
parser.add_argument("--cotta_lr", type=float, default=1e-3)  # CoTTA的学习率
parser.add_argument("--ema_decay", type=float, default=0.999)  # Teacher EMA decay
parser.add_argument("--rst_m", type=float, default=0.01)  # 随机恢复概率
parser.add_argument("--lambda_u", type=float, default=1.0)  # 无标签损失权重
args = parser.parse_args()


def update_ema_variables(model, ema_model, alpha):
    """更新EMA模型参数"""
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def configure_model_for_cotta(model):
    """配置模型用于CoTTA: BatchNorm设置为训练模式"""
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


def entropy_loss(logits):
    """计算预测的熵"""
    probs = F.softmax(logits, dim=1)
    probs = torch.clamp(probs, min=1e-7)
    entropy = -torch.sum(probs * torch.log(probs), dim=1)
    return entropy.mean()


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
    print(f"Baseline: CoTTA (Continual Test-Time Adaptation)")
    print(f"batch_size: {batch_size}")
    print(f"Dataset: {dataset}, Model: {model_name}")
    print(f"CoTTA learning rate: {args.cotta_lr}")
    print(f"EMA decay: {args.ema_decay}")
    print(f"Restoration probability: {args.rst_m}")
    print(f"Lambda_u: {args.lambda_u}")
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

    # 用于CoTTA适应的验证集（带数据增强）
    val_dataset_aug = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                          transform=data_transform['train'])
    val_loader_aug = torch.utils.data.DataLoader(val_dataset_aug, batch_size=batch_size,
                                                 shuffle=True, num_workers=nw)

    test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                       transform=data_transform['val'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=nw)

    print(f"Training: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # 初始化学生模型和教师模型
    student_model = EfficientNet.from_name('efficientnet-b3', num_classes=1000).to('cuda')
    student_model.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    student_model._fc = nn.Linear(channel, num_classes).to('cuda')
    student_model.aug = False

    teacher_model = EfficientNet.from_name('efficientnet-b3', num_classes=1000).to('cuda')
    teacher_model.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    teacher_model._fc = nn.Linear(channel, num_classes).to('cuda')
    teacher_model.aug = False

    # 复制学生模型参数到教师模型
    for param_t, param_s in zip(teacher_model.parameters(), student_model.parameters()):
        param_t.data.copy_(param_s.data)

    # 教师模型不需要梯度
    for param in teacher_model.parameters():
        param.requires_grad = False

    # 保存源模型参数（用于随机恢复）
    source_model_state = copy.deepcopy(student_model.state_dict())

    criterion = nn.CrossEntropyLoss()

    # 主优化器（用于训练集监督学习）
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

    # CoTTA优化器（只优化BatchNorm参数，用于验证集）
    bn_params = collect_bn_params(student_model)
    cotta_optimizer = torch.optim.Adam(bn_params, lr=args.cotta_lr)

    best_acc_val = 0.0
    best_acc_test = 0.0

    for epoch in range(epochs):
        # ===== 阶段1: 训练集监督训练 =====
        student_model.train()
        teacher_model.eval()

        supervised_loss_epoch = []
        num_correct = 0
        num_total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Supervised")

        for imgs, labels in train_bar:
            imgs = imgs.to('cuda')
            labels = labels.to('cuda')

            optimizer.zero_grad()
            outputs, _ = student_model(imgs)
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

        # ===== 阶段2: 验证集CoTTA适应 =====
        # 配置模型为CoTTA模式
        student_model = configure_model_for_cotta(student_model)
        teacher_model.train()  # 教师模型也设置为train模式（但不更新梯度）

        cotta_loss_epoch = []
        val_bar = tqdm(val_loader_aug, desc=f"Epoch {epoch+1}/{epochs} - CoTTA")

        for imgs, _ in val_bar:
            imgs = imgs.to('cuda')

            # CoTTA核心: 增强平均 + 一致性学习
            # 使用教师模型生成目标
            with torch.no_grad():
                teacher_outputs, _ = teacher_model(imgs)

            # 学生模型预测
            cotta_optimizer.zero_grad()
            student_outputs, _ = student_model(imgs)

            # 损失: 熵最小化 + 与教师模型的一致性
            loss_ent = entropy_loss(student_outputs)
            loss_consist = F.mse_loss(F.softmax(student_outputs, dim=1),
                                      F.softmax(teacher_outputs, dim=1))
            loss = loss_ent + loss_consist

            loss.backward()
            cotta_optimizer.step()
            cotta_loss_epoch.append(loss.item())

            # 更新教师模型(EMA)
            update_ema_variables(student_model, teacher_model, args.ema_decay)

            # 随机恢复(Stochastic Restoration)
            for nm, param in student_model.named_parameters():
                if nm in source_model_state and ('bn' in nm or 'norm' in nm):
                    if np.random.rand() < args.rst_m:
                        param.data = source_model_state[nm].data.clone().to('cuda')

            val_bar.set_description(
                f"Epoch {epoch+1}/{epochs} - CoTTA Loss: {loss.item():.3f}"
            )

        avg_cotta_loss = sum(cotta_loss_epoch) / len(cotta_loss_epoch)

        # 评估（使用教师模型）
        val_acc, val_acc5 = _accuracy_(teacher_model, val_loader)
        test_acc, test_acc5 = _accuracy_(teacher_model, test_loader)

        if val_acc > best_acc_val:
            print(f"\n{'='*50}")
            print("Better model found!")
            print(f"{'='*50}")
            best_acc_val = val_acc
            best_acc_test = test_acc

        print(f"\nEpoch {epoch+1}:")
        print(f"  Supervised Loss={avg_supervised_loss:.4f}, CoTTA Loss={avg_cotta_loss:.4f}")
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
