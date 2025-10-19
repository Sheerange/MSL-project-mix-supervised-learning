"""
Baseline: Pseudo-Labeling (Lee et al., 2013)
简单的伪标签半监督学习方法
- 在训练集上进行监督训练
- 定期使用模型对验证集生成伪标签
- 将高置信度的伪标签样本加入训练
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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import functional as F
import torch.nn.init as init
from torch.utils.data import Dataset

class TensorDatasetWithTransform(Dataset):
    """自定义Dataset类，使TensorDataset与ImageFolder兼容"""
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 返回已经是tensor的数据和label（与ImageFolder + transform后的格式一致）
        return self.data[idx], self.labels[idx].item()  # .item()将tensor转为int

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--nw", type=int, default=20)
parser.add_argument("--lr", type=float, default=5e-3)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--dataset", type=str, default='CUB200')
parser.add_argument("--confidence", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--pseudo_start_epoch", type=int, default=5)  # 开始使用伪标签的epoch
parser.add_argument("--pseudo_interval", type=int, default=1)  # 每隔几个epoch更新伪标签
args = parser.parse_args()

def init_weights_kaiming(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, nonlinearity='relu')
        init.zeros_(m.bias)

def init_weights_xavier(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        init.zeros_(m.bias)

def generate_pseudo_labels(model, dataloader, device, threshold=0.9):
    """生成伪标签"""
    model.eval()
    pseudo_data = []
    pseudo_labels = []
    true_labels = []

    corrects = 0
    total_pseudo = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs,_ = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, dim=1)

            # 选择高置信度样本
            high_conf_mask = confidence > threshold
            pseudo_data.append(inputs[high_conf_mask])
            pseudo_labels.append(predicted[high_conf_mask])
            true_labels.append(labels[high_conf_mask])

            corrects += (predicted[high_conf_mask] == labels[high_conf_mask]).sum().item()
            total_pseudo += high_conf_mask.sum().item()

    print(f"Pseudo-labels: {corrects}/{total_pseudo} correct ({100*corrects/max(total_pseudo,1):.2f}%)")

    if pseudo_data:
        pseudo_data = torch.cat(pseudo_data)
        pseudo_labels = torch.cat(pseudo_labels)
        true_labels = torch.cat(true_labels)
    else:
        pseudo_data = torch.empty((0, 3, 224, 224), device=device)
        pseudo_labels = torch.empty((0,), dtype=torch.long, device=device)
        true_labels = torch.empty((0,), dtype=torch.long, device=device)

    return pseudo_data, pseudo_labels, true_labels

def _accuracy_(net, data_loader):
    net.eval()
    num_total = 0
    num_acc = 0
    num_acc_top5 = 0
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to('cuda')
            labels = labels.to('cuda')

            output,_ = net(imgs)
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
    print(f"Baseline: Pseudo-Labeling")
    print(f"batch_size: {batch_size}")
    print(f"Dataset: {dataset}, Model: {model_name}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Pseudo-label start epoch: {args.pseudo_start_epoch}")
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

    # 用于生成伪标签的验证集加载器（带数据增强）
    val_dataset_aug = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                          transform=data_transform['train'])

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
    optimizer = torch.optim.SGD(net.parameters(), lr=base_lr, weight_decay=weight_decay, momentum=0.9)

    # 使用CosineAnnealingLR替代OneCycleLR，避免伪标签导致的步数不匹配问题
    schedule = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=base_lr/1000)

    best_acc_val = 0.0
    best_acc_test = 0.0

    for epoch in range(epochs):
        net.train()
        train_loss_epoch = []
        num_correct = 0
        num_total = 0

        # 判断是否使用伪标签
        use_pseudo = (epoch >= args.pseudo_start_epoch) and (epoch % args.pseudo_interval == 0)

        # 如果需要，生成伪标签
        if use_pseudo:
            print(f"\n{'='*50}")
            print(f"Generating pseudo-labels from validation set...")
            pseudo_data, pseudo_labels, true_labels = generate_pseudo_labels(
                net, val_loader, 'cuda', threshold=args.confidence
            )

            if len(pseudo_data) > 0:
                # 合并训练集和伪标签数据
                # 使用自定义Dataset类以确保与ImageFolder兼容
                pseudo_dataset = TensorDatasetWithTransform(
                    pseudo_data.cpu(), pseudo_labels.cpu()
                )
                combined_dataset = torch.utils.data.ConcatDataset([train_dataset, pseudo_dataset])
                combined_loader = torch.utils.data.DataLoader(
                    combined_dataset, batch_size=batch_size, shuffle=True, num_workers=nw
                )
                print(f"Combined dataset size: {len(combined_dataset)}")
                train_bar = tqdm(combined_loader)
            else:
                print("No high-confidence pseudo-labels generated, using original training set")
                train_bar = tqdm(train_loader)
        else:
            train_bar = tqdm(train_loader)

        # 训练
        for imgs, labels in train_bar:
            imgs = imgs.to('cuda')
            labels = labels.to('cuda')

            optimizer.zero_grad()
            outputs,_ = net(imgs)
            loss = criterion(outputs, labels)

            _, pred = torch.max(outputs, dim=1)
            num_correct += torch.sum(pred == labels.detach_())
            num_total += labels.size(0)

            train_loss_epoch.append(loss.item())
            loss.backward()
            optimizer.step()

            train_bar.set_description(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.3f}")

        # 每个epoch结束后更新学习率
        schedule.step()

        train_acc = num_correct.detach().cpu().numpy() * 100 / num_total
        avg_train_loss = sum(train_loss_epoch) / len(train_loss_epoch)

        # 评估
        val_acc, val_acc5 = _accuracy_(net, val_loader)
        test_acc, test_acc5 = _accuracy_(net, test_loader)

        if val_acc > best_acc_val:
            print(f"\n{'='*50}")
            print("Better model found!")
            print(f"{'='*50}")
            best_acc_val = val_acc
            best_acc_test = test_acc
            # torch.save(net.state_dict(), './weights/baseline_pseudo_label.pth')

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, "
              f"Val Acc={val_acc:.2f}%, Test Acc={test_acc:.2f}%, "
              f"Val Top5={val_acc5:.2f}%, Test Top5={test_acc5:.2f}%")

    print(f"\n{'='*50}")
    print(f"Training Finished!")
    print(f"Best Val Acc: {best_acc_val:.2f}%")
    print(f"Best Test Acc: {best_acc_test:.2f}%")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()
