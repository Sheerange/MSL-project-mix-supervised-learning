# 如何将项目上传到GitHub

## 步骤1: 在GitHub上创建仓库

### 1.1 登录GitHub
- 打开浏览器，访问 https://github.com
- 登录你的账号

### 1.2 创建新仓库
1. 点击右上角的 `+` 号
2. 选择 `New repository`
3. 填写信息：
   - **Repository name**: `msl-fine-grained` (或你喜欢的名字)
   - **Description**: "MSL: Mixed Supervised Learning for Fine-Grained Image Classification"
   - **Public/Private**: 选择 Public（公开）或 Private（私密）
   - **❌ 不要勾选** "Add a README file"
   - **❌ 不要勾选** "Add .gitignore"
   - **❌ 不要勾选** "Choose a license"
4. 点击 `Create repository` 按钮

### 1.3 记录仓库地址
创建后，GitHub会显示一个地址，类似：
```
https://github.com/你的用户名/msl-fine-grained.git
```
**把这个地址复制下来！**

---

## 步骤2: 连接本地仓库到GitHub

在终端中执行以下命令：

### 2.1 添加远程仓库
```bash
# 替换下面的URL为你刚才复制的地址
git remote add origin https://github.com/你的用户名/msl-fine-grained.git
```

### 2.2 验证连接
```bash
git remote -v
```
应该看到：
```
origin  https://github.com/你的用户名/msl-fine-grained.git (fetch)
origin  https://github.com/你的用户名/msl-fine-grained.git (push)
```

---

## 步骤3: 推送代码到GitHub

### 3.1 推送主分支
```bash
git push -u origin main
```

**如果遇到错误说 `main` 分支不存在，尝试：**
```bash
# 先检查当前分支名
git branch

# 如果是 master，则推送 master
git push -u origin master

# 或者重命名分支为 main
git branch -M main
git push -u origin main
```

### 3.2 输入GitHub账号密码

**重要提示：** GitHub现在不支持密码登录，需要使用 **Personal Access Token (PAT)**

#### 如何创建Token：

1. 登录GitHub
2. 点击右上角头像 → `Settings`
3. 左侧菜单最下方 → `Developer settings`
4. 左侧选择 `Personal access tokens` → `Tokens (classic)`
5. 点击 `Generate new token` → `Generate new token (classic)`
6. 填写信息：
   - **Note**: "MSL Project Upload"
   - **Expiration**: 90 days（或选择其他期限）
   - **Select scopes**: 勾选 `repo` (所有子选项会自动勾选)
7. 点击底部 `Generate token`
8. **立即复制token！** (只显示一次，保存到安全的地方)

#### 使用Token推送：

```bash
git push -u origin main
```

当提示输入密码时：
- **Username**: 你的GitHub用户名
- **Password**: 粘贴刚才复制的Token（不是GitHub密码！）

---

## 步骤4: 验证上传成功

1. 打开浏览器
2. 访问 `https://github.com/你的用户名/msl-fine-grained`
3. 应该能看到所有文件和README.md

---

## 完整操作流程（复制粘贴版）

**在终端依次执行：**

```bash
# 1. 检查当前分支
git branch

# 2. 如果不是main分支，重命名为main
git branch -M main

# 3. 添加远程仓库（替换为你的仓库地址）
git remote add origin https://github.com/你的用户名/msl-fine-grained.git

# 4. 验证连接
git remote -v

# 5. 推送到GitHub
git push -u origin main
```

---

## 之后如何更新代码到GitHub？

每次修改代码后：

```bash
# 1. 查看修改了什么
git status

# 2. 添加修改
git add .

# 3. 提交到本地
git commit -m "描述你的修改"

# 4. 推送到GitHub（之后只需要这一个命令）
git push
```

---

## 常见问题

### Q1: 推送时提示 "fatal: remote origin already exists"
```bash
# 删除已有的远程仓库配置
git remote remove origin

# 重新添加
git remote add origin https://github.com/你的用户名/msl-fine-grained.git
```

### Q2: 推送时提示 "Updates were rejected"
```bash
# 先拉取远程代码（如果GitHub上有其他提交）
git pull origin main --rebase

# 再推送
git push -u origin main
```

### Q3: Token过期了怎么办？
重复"创建Token"的步骤，生成新的Token，下次推送时使用新Token

### Q4: 我忘记保存Token了
重新生成一个新的Token即可

### Q5: 不想每次都输入Token
使用SSH密钥（稍复杂，但更安全方便）：

```bash
# 1. 生成SSH密钥
ssh-keygen -t ed25519 -C "your_email@example.com"

# 2. 将公钥添加到GitHub
# Settings → SSH and GPG keys → New SSH key
# 粘贴 ~/.ssh/id_ed25519.pub 的内容

# 3. 修改远程仓库URL为SSH格式
git remote set-url origin git@github.com:你的用户名/msl-fine-grained.git

# 4. 之后推送不需要输入密码
git push
```

---

## 使用SSH方式（推荐）

如果你觉得每次输入Token麻烦，可以用SSH：

### 生成SSH密钥
```bash
ssh-keygen -t ed25519 -C "你的邮箱@example.com"
# 一路回车（使用默认设置）
```

### 复制公钥
```bash
cat ~/.ssh/id_ed25519.pub
# 复制输出的内容
```

### 添加到GitHub
1. GitHub → Settings → SSH and GPG keys
2. 点击 "New SSH key"
3. Title: "我的电脑"
4. Key: 粘贴刚才复制的公钥
5. 点击 "Add SSH key"

### 使用SSH URL
```bash
# 移除HTTPS远程仓库
git remote remove origin

# 添加SSH远程仓库
git remote add origin git@github.com:你的用户名/msl-fine-grained.git

# 推送（不需要密码）
git push -u origin main
```

---

## 图形化工具（可选）

如果觉得命令行太复杂，可以使用：

1. **GitHub Desktop**
   - 下载: https://desktop.github.com/
   - 图形界面，拖拽操作

2. **VS Code Git插件**
   - 如果你用VS Code编辑器
   - 内置Git功能，点击按钮就能提交

---

## 推送后的优势

✅ 云端备份，不怕电脑坏了
✅ 可以在任何电脑上访问代码
✅ 可以分享给导师/同事查看
✅ 可以在GitHub展示你的项目
✅ 支持团队协作

---

## 需要帮助？

如果在操作过程中遇到问题，告诉我具体的错误信息，我会帮你解决！

常见错误：
- "Permission denied" → Token权限不足或过期
- "Repository not found" → URL错误或仓库不存在
- "Updates were rejected" → 需要先pull再push
