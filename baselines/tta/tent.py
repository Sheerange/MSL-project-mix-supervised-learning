"""
Baseline: TENT - Test-Time Entropy Minimization (Wang et al., ICLR 2021)
测试时适应(Test-Time Adaptation)方法
- 先在训练集上进行标准监督训练
- 在测试/验证集上通过熵最小化进行在线适应
- 只更新BatchNorm参数

论文: https://arxiv.org/abs/2006.10726
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
import torch.nn.init as init
import copy

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--nw", type=int, default=20)
parser.add_argument("--lr", type=float, default=5e-3)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--dataset", type=str, default='cifar10_tiny')
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--tent_lr", type=float, default=1e-3)  # TENT的学习率
parser.add_argument("--tent_steps", type=int, default=1)  # 每个batch的TENT更新步数
args = parser.parse_args()


def configure_model(model):
    """配置模型用于TENT: 设置为eval模式但启用BatchNorm的训练模式"""
    model.eval()
    # 启用BatchNorm的训练模式
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.train()
            # 强制使用batch统计而不是running统计
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def collect_params(model):
    """收集BatchNorm的affine参数（gamma和beta）"""
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # affine parameters
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def entropy_loss(logits):
    """计算预测的熵"""
    probs = F.softmax(logits, dim=1)
    # 避免log(0)
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
    print(f"Baseline: TENT (Test-Time Entropy Minimization)")
    print(f"batch_size: {batch_size}")
    print(f"Dataset: {dataset}, Model: {model_name}")
    print(f"TENT learning rate: {args.tent_lr}")
    print(f"TENT steps per batch: {args.tent_steps}")
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

    # TENT阶段使用的验证集加载器（shuffle=True用于在线适应）
    val_dataset_tent = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                           transform=data_transform['train'])
    val_loader_tent = torch.utils.data.DataLoader(val_dataset_tent, batch_size=batch_size,
                                                  shuffle=True, num_workers=nw)

    test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                       transform=data_transform['val'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=nw)

    print(f"Training: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # ===== 阶段1: 标准监督训练 =====
    print(f"\n{'='*50}")
    print("Phase 1: Standard Supervised Training")
    print(f"{'='*50}\n")

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

    best_acc_val_supervised = 0.0
    best_model_state = None

    for epoch in range(epochs):
        net.train()
        train_loss_epoch = []
        num_correct = 0
        num_total = 0

        train_bar = tqdm(train_loader)

        for imgs, labels in train_bar:
            imgs = imgs.to('cuda')
            labels = labels.to('cuda')

            optimizer.zero_grad()
            outputs, _ = net(imgs)
            loss = criterion(outputs, labels)

            _, pred = torch.max(outputs, dim=1)
            num_correct += torch.sum(pred == labels.detach_())
            num_total += labels.size(0)

            train_loss_epoch.append(loss.item())
            loss.backward()
            optimizer.step()
            schedule.step()

            train_bar.set_description(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.3f}")

        train_acc = num_correct.detach().cpu().numpy() * 100 / num_total
        avg_train_loss = sum(train_loss_epoch) / len(train_loss_epoch)

        val_acc, val_acc5 = _accuracy_(net, val_loader)
        test_acc, test_acc5 = _accuracy_(net, test_loader)

        if val_acc > best_acc_val_supervised:
            print(f"\nBetter model found in supervised training!")
            best_acc_val_supervised = val_acc
            best_model_state = copy.deepcopy(net.state_dict())

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, "
              f"Val Acc={val_acc:.2f}%, Test Acc={test_acc:.2f}%")

    print(f"\n{'='*50}")
    print(f"Supervised Training Finished!")
    print(f"Best Val Acc (Supervised): {best_acc_val_supervised:.2f}%")
    print(f"{'='*50}\n")

    # ===== 阶段2: TENT (Test-Time Adaptation) =====
    print(f"\n{'='*50}")
    print("Phase 2: TENT - Test-Time Adaptation on Validation Set")
    print(f"{'='*50}\n")

    # 加载最佳模型
    net.load_state_dict(best_model_state)

    # 配置模型用于TENT
    net = configure_model(net)

    # 只优化BatchNorm参数
    params, param_names = collect_params(net)
    print(f"Optimizing {len(params)} BatchNorm parameters")
    print(f"Parameters: {param_names[:5]}... (showing first 5)")

    tent_optimizer = torch.optim.Adam(params, lr=args.tent_lr)

    # TENT适应: 在验证集上边适应边预测
    # 重要: 每个batch适应后收集预测，但不累积更新（防止灾难性遗忘）
    tent_bar = tqdm(val_loader, desc="TENT Adaptation")
    tent_loss_epoch = []

    all_preds = []
    all_labels = []

    # 保存初始模型状态（每隔一定步数重置，防止偏移过大）
    initial_state = copy.deepcopy(net.state_dict())
    reset_interval = 10  # 每10个batch重置一次

    for batch_idx, (imgs, labels) in enumerate(tent_bar):
        imgs = imgs.to('cuda')
        labels = labels.to('cuda')

        # TENT: 在当前batch上进行适应
        for _ in range(args.tent_steps):
            tent_optimizer.zero_grad()
            outputs, _ = net(imgs)
            loss = entropy_loss(outputs)
            loss.backward()
            tent_optimizer.step()
            tent_loss_epoch.append(loss.item())

        # 适应后预测
        with torch.no_grad():
            outputs, _ = net(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        tent_bar.set_description(f"TENT Loss: {loss.item():.4f}")

        # 定期重置模型，防止累积偏移
        if (batch_idx + 1) % reset_interval == 0:
            net.load_state_dict(initial_state)

    avg_tent_loss = sum(tent_loss_epoch) / len(tent_loss_epoch) if tent_loss_epoch else 0
    print(f"\nAverage TENT Loss: {avg_tent_loss:.4f}")

    # 计算TENT后的验证集准确率
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    val_acc_tent = 100.0 * (all_preds == all_labels).sum().item() / len(all_labels)

    # 在测试集上评估（使用相同的TENT过程）
    # 重置模型
    net.load_state_dict(initial_state)
    test_bar = tqdm(test_loader, desc="TENT on Test Set")
    test_preds = []
    test_labels_list = []

    for batch_idx, (imgs, labels) in enumerate(test_bar):
        imgs = imgs.to('cuda')
        labels = labels.to('cuda')

        # TENT适应
        for _ in range(args.tent_steps):
            tent_optimizer.zero_grad()
            outputs, _ = net(imgs)
            loss = entropy_loss(outputs)
            loss.backward()
            tent_optimizer.step()

        # 预测
        with torch.no_grad():
            outputs, _ = net(imgs)
            _, preds = torch.max(outputs, 1)
            test_preds.append(preds.cpu())
            test_labels_list.append(labels.cpu())

        # 定期重置
        if (batch_idx + 1) % reset_interval == 0:
            net.load_state_dict(initial_state)

    test_preds = torch.cat(test_preds)
    test_labels_all = torch.cat(test_labels_list)
    test_acc_tent = 100.0 * (test_preds == test_labels_all).sum().item() / len(test_labels_all)

    print(f"\n{'='*50}")
    print("Final Results:")
    print(f"{'='*50}")
    print(f"After Supervised Training:")
    print(f"  Best Val Acc: {best_acc_val_supervised:.2f}%")
    print(f"\nAfter TENT Adaptation (with periodic reset):")
    print(f"  Val Acc: {val_acc_tent:.2f}%")
    print(f"  Test Acc: {test_acc_tent:.2f}%")
    print(f"  Improvement: {val_acc_tent - best_acc_val_supervised:+.2f}%")
    print(f"{'='*50}")
    print("Final Results:")
    # 可选: 保存TENT后的模型
    # torch.save(net.state_dict(), './weights/baseline_tent.pth')


if __name__ == '__main__':
    main()
