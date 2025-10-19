# Git 使用教程

## ✅ 已完成的工作

1. ✅ 清理了旧文件
2. ✅ 初始化了Git仓库
3. ✅ 创建了.gitignore文件
4. ✅ 完成了第一次提交

当前提交ID: `a848894`

---

## Git基础概念

Git是一个版本控制系统，可以帮你：
- 📝 记录代码的每次修改
- ⏮️ 回退到任何历史版本
- 🌿 创建分支进行实验
- 👥 与他人协作开发

### 三个重要区域

```
工作区              暂存区              本地仓库
(Working)    -->    (Staging)    -->   (Repository)
  修改文件      git add .         git commit -m "..."
```

---

## 常用Git命令

### 1. 查看状态

```bash
# 查看当前仓库状态
git status

# 简洁模式
git status -s
```

**说明**: 显示哪些文件被修改、新增或删除

### 2. 添加文件到暂存区

```bash
# 添加所有文件
git add .

# 添加特定文件
git add msl/DRC_MSL.py

# 添加某个目录
git add baselines/
```

**说明**: 将修改的文件添加到暂存区，准备提交

### 3. 提交更改

```bash
# 提交并写提交信息
git commit -m "修改了XXX功能"

# 提交并写详细说明
git commit -m "标题" -m "详细说明第一行" -m "详细说明第二行"
```

**说明**: 将暂存区的文件提交到本地仓库

### 4. 查看历史

```bash
# 查看提交历史（简洁）
git log --oneline

# 查看详细历史
git log

# 查看最近3次提交
git log -3

# 查看某个文件的修改历史
git log -- msl/DRC_MSL.py
```

### 5. 查看修改内容

```bash
# 查看工作区的修改（还未add）
git diff

# 查看暂存区的修改（已add但未commit）
git diff --staged

# 查看某次提交的修改
git show a848894
```

### 6. 撤销修改

```bash
# 撤销工作区的修改（危险！会丢失修改）
git checkout -- msl/DRC_MSL.py

# 从暂存区移除文件（不删除修改）
git restore --staged msl/DRC_MSL.py

# 撤销最后一次提交（保留修改）
git reset --soft HEAD~1

# 撤销最后一次提交（丢弃修改，危险！）
git reset --hard HEAD~1
```

---

## 实际工作流程

### 场景1: 修改代码后提交

```bash
# 1. 修改代码（在编辑器中修改文件）

# 2. 查看修改了什么
git status
git diff

# 3. 添加到暂存区
git add .

# 4. 提交
git commit -m "Fix: 修复了MSL中的bug"

# 5. 查看提交记录
git log --oneline
```

### 场景2: 创建新功能分支

```bash
# 1. 创建并切换到新分支
git checkout -b feature-new-baseline

# 2. 修改代码并提交
git add .
git commit -m "Add: 添加新的baseline方法"

# 3. 切换回主分支
git checkout main

# 4. 合并新功能
git merge feature-new-baseline

# 5. 删除功能分支
git branch -d feature-new-baseline
```

### 场景3: 查看和比较版本

```bash
# 查看所有提交
git log --oneline --graph --all

# 查看某个文件在两次提交之间的差异
git diff a848894 HEAD -- msl/DRC_MSL.py

# 回到某个历史版本（临时查看，不修改）
git checkout a848894

# 返回最新版本
git checkout main
```

---

## 常见问题

### Q1: 如何修改最后一次提交信息？

```bash
git commit --amend -m "新的提交信息"
```

### Q2: 如何查看某个文件是谁修改的？

```bash
git blame msl/DRC_MSL.py
```

### Q3: 如何暂存当前修改（不提交）？

```bash
# 暂存当前修改
git stash

# 查看暂存列表
git stash list

# 恢复最近的暂存
git stash pop
```

### Q4: 如何忽略某些文件？

编辑 `.gitignore` 文件，添加要忽略的文件/目录：

```
*.log
__pycache__/
*.pyc
```

### Q5: 不小心提交了大文件怎么办？

```bash
# 从最后一次提交中移除
git rm --cached efficientnet-b3.pth
git commit --amend -m "Remove large file"
```

---

## 与GitHub/GitLab协作

### 第一次推送到远程仓库

```bash
# 1. 在GitHub/GitLab上创建空仓库

# 2. 添加远程仓库
git remote add origin https://github.com/yourusername/msl-project.git

# 3. 推送到远程
git push -u origin main
```

### 日常推送和拉取

```bash
# 推送本地修改到远程
git push

# 从远程拉取最新代码
git pull

# 仅拉取不合并
git fetch
```

---

## 你的当前仓库信息

```bash
# 查看所有分支
git branch -a

# 查看远程仓库
git remote -v

# 查看提交历史
git log --oneline --graph

# 查看当前状态
git status
```

---

## 推荐的提交信息格式

```
类型: 简短描述（不超过50字）

详细说明（如果需要）
- 第一点
- 第二点

相关Issue: #123
```

**类型标签:**
- `Feat`: 新功能
- `Fix`: 修复bug
- `Docs`: 文档修改
- `Style`: 代码格式修改
- `Refactor`: 代码重构
- `Test`: 测试相关
- `Chore`: 构建/工具链修改

**示例:**
```bash
git commit -m "Feat: 添加SAR baseline方法

- 实现SAR的核心算法
- 添加entropy filtering机制
- 更新README文档

相关Issue: #15"
```

---

## 快速参考卡片

```bash
# 常用命令速查
git status              # 查看状态
git add .              # 添加所有文件
git commit -m "msg"    # 提交
git log --oneline      # 查看历史
git diff               # 查看修改
git push               # 推送到远程
git pull               # 从远程拉取
```

---

## 下一步

建议你实践以下操作：

1. ✅ **已完成**: 初始化仓库和第一次提交
2. 📝 **接下来**: 修改一个文件，然后提交
3. 🌿 **进阶**: 创建一个分支进行实验
4. 🌐 **可选**: 推送到GitHub

试试看：
```bash
# 查看当前状态
git status

# 查看提交历史
git log --oneline

# 查看第一次提交的详情
git show a848894
```

Good luck! 🚀
