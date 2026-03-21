# 公网演示版部署说明

这份项目可以直接部署到 Streamlit Community Cloud，生成一个可公开访问的网址。

## 目录说明
- `app.py`：主程序
- `requirements.txt`：依赖
- `sample_template.csv`：示例模板
- `.streamlit/config.toml`：部署时的主题和上传限制配置

## 第一步：上传到 GitHub
1. 登录 GitHub，新建一个仓库，例如：`ai-teaching-dashboard-public`
2. 把本文件夹内所有文件上传到仓库根目录
3. 确保仓库里至少有：
   - `app.py`
   - `requirements.txt`
   - `.streamlit/config.toml`

## 第二步：部署到 Streamlit Community Cloud
1. 打开 Streamlit Community Cloud
2. 用 GitHub 账号登录
3. 点击 `Create app`
4. 选择你的 GitHub 仓库
5. `Branch` 选 `main`
6. `Main file path` 填 `app.py`
7. 可自定义 `App URL`
8. 点击 `Deploy`

## 第三步：演示前建议
1. 使用脱敏数据，不要直接公开学生真实姓名、学号、电话等
2. 把上传数据替换成演示样例
3. 可在页面标题中写上学校、学科、班级、考试名称

## 常见问题
### 1. 部署后打不开
检查 `requirements.txt` 是否完整，重新部署。

### 2. 中文乱码
优先上传 `.xlsx`，不要反复编辑 CSV。

### 3. 上传文件过大
当前配置里 `maxUploadSize = 50`，表示单文件 50MB。

## 说明
这是“公开演示版”。后续如果你需要“公网可访问但必须输入密码”，可以在此基础上再加一层访问控制。
