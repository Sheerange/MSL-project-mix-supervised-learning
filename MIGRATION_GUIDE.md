# Quick Migration Guide

## 重构完成！✅

项目已成功重构，所有文件已重新组织，TCO已全部重命名为MSL。

## 新的项目结构

```
📁 msl/                    # MSL主方法
📁 baselines/              # 所有baseline
  ├── ssl/                 # 半监督学习方法 (5个)
  ├── tta/                 # 测试时适应方法 (3个)
  └── self_supervised/     # 自监督学习方法 (2个)
📁 models/                 # 所有模型架构
📁 utils/                  # 工具函数
📁 legacy/                 # 旧代码（参考用）
📁 docs/                   # 文档
```

## 如何使用新结构

### 1. 训练MSL模型

**旧命令：**
```bash
python DRC_TCO.py --tco 1 --dataset CUB200
```

**新命令：**
```bash
python msl/DRC_MSL.py --msl 1 --dataset CUB200
```

### 2. 训练Baseline方法

**示例 - Pseudo-Labeling:**
```bash
python baselines/ssl/pseudo_label.py --dataset CUB200 --batch_size 64
```

**示例 - TENT:**
```bash
python baselines/tta/tent.py --dataset CUB200 --tent_lr 1e-3
```

**示例 - SimCLR:**
```bash
python baselines/self_supervised/simclr.py --dataset CUB200
```

### 3. 所有Baseline脚本路径

#### Semi-Supervised Learning (SSL)
- `baselines/ssl/pseudo_label.py`
- `baselines/ssl/fixmatch.py`
- `baselines/ssl/flexmatch.py`
- `baselines/ssl/freematch.py`
- `baselines/ssl/mean_teacher.py`

#### Test-Time Adaptation (TTA)
- `baselines/tta/tent.py`
- `baselines/tta/cotta.py`
- `baselines/tta/sar.py`

#### Self-Supervised Learning
- `baselines/self_supervised/simclr.py`
- `baselines/self_supervised/moco.py`

## 主要变化

### ✅ 已完成的修改

1. **目录结构重组** - 按功能分类
2. **TCO→MSL重命名** - 所有代码中的TCO已改为MSL
3. **Import路径更新** - 所有import语句已更新
4. **README更新** - 完整的使用文档
5. **文档整理** - 所有文档移至docs/

### 📝 命名变化

| 旧名称 | 新名称 |
|--------|--------|
| `--tco 1` | `--msl 1` |
| `tco_switch` | `msl_switch` |
| `Tco_mode` | `Msl_mode` |
| `DRC_TCO.py` | `msl/DRC_MSL.py` |

## 清理旧文件（可选）

验证新结构正常工作后，可以删除根目录下的旧文件：

```bash
# 删除旧baseline文件
rm baseline_*.py

# 删除旧模型文件
rm model.py model_dense.py resnet.py resnest.py swin_transformer.py
rm transnext.py poolformer.py pvt.py pyramid_vig.py cls_cvt.py splat.py

# 删除旧工具文件
rm attention_cuda.py attention_native.py random_augment.py utils.py

# 删除旧主文件
rm DRC_TCO.py wcy_method.py weight_init.py

# 删除旧目录
rm -rf efficientnet_pytorch/ gcn_lib/
```

## 验证安装

测试新结构是否工作：

```bash
# 检查MSL脚本
python msl/DRC_MSL.py --help

# 检查baseline脚本
python baselines/ssl/pseudo_label.py --help
```

## 需要帮助？

- 查看 `README.md` 获取完整文档
- 查看 `docs/RESTRUCTURING_SUMMARY.md` 了解重构细节
- 旧代码保存在 `legacy/` 目录中供参考

---

**重构完成时间：** 2025-10-19
**状态：** ✅ 所有任务完成
