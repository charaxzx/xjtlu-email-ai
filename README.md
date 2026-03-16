# 📧 XJTLU 邮件智能助手

> 一个基于 Playwright + LLM 的 XJTLU 邮箱自动总结工具，支持关键词搜索、智能提取正文并生成中文摘要。

## ✨ 功能特性

- 🔍 **关键词检索** — 搜索收件箱，或直接获取最新邮件
- 📄 **正文提取** — 自动点击并提取10封邮件全文
- 🤖 **AI 总结** — 调用 LLM（支持 OpenAI / DeepSeek 等 API）生成智能摘要
- 🌐 **Web 界面** — 基于 FastAPI 的浏览器操作界面
- 🍪 **Cookie 登录** — 支持 OWA 双因素认证环境

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r src/requirements.txt
playwright install msedge
```

### 2. 配置

复制模板并填入你的信息：

```bash
cp src/config.example.json src/config.json
```

编辑 `src/config.json`（已经默认指向 XJTLU）：

```json
{
  "ai": {
    "base_url": "https://api.openai.com/v1",
    "api_key": "你的 API Key",
    "model": "gpt-4o-mini"
  },
  "email": {
    "url": "https://mail.xjtlu.edu.cn/owa",
    "login_type": "cookie",
    "username": "",
    "password": ""
  }
}
```

### 3. 获取 Cookie

1. 在 Edge/Chrome 中安装 **Cookie-Editor** 插件
2. 登录你的学校邮箱
3. 点击 Cookie-Editor → **Export as JSON**
4. 将导出的 JSON 粘贴到 `config.json` 的 `cookies` 字段，**或**保存为 `src/cookies.txt` 文件并设置 `"cookie_file": "cookies.txt"`

### 4. 启动

**Windows（推荐）**：双击 `run_app.bat`

**命令行**：
```bash
# Web 界面模式
python src/app.py
# 然后访问 http://localhost:8001

# 命令行模式
python src/main.py
```

---

## ⚙️ 技术栈

| 模块 | 技术 |
|------|------|
| 浏览器自动化 | Playwright (Microsoft Edge) |
| Web 服务 | FastAPI + Uvicorn |
| AI 接口 | OpenAI 兼容 API |
| 正文解析 | BeautifulSoup4 |

---

## 📝 配置文件说明

| 字段 | 说明 |
|------|------|
| `ai.base_url` | LLM API 地址（支持 OpenAI / DeepSeek 等） |
| `ai.api_key` | API Key |
| `ai.model` | 使用的模型名称 |
| `email.url` | 邮箱 URL（默认 XJTLU） |
| `email.cookie_file` | Netscape 格式 Cookie 文件路径 |
| `email.cookies` | Cookie JSON 数组（与 cookie_file 二选一） |
| `selectors.*` | 页面 CSS 选择器（如 OWA 更新 UI 后调整） |

---

## ⚠️ 注意事项

- **Cookie 有效期**：Cookie 通常在几天到数周后失效，失效后需重新导出
- **隐私安全**：`config.json` 和 `cookies.txt` 已在 `.gitignore` 中，**请勿提交到公开仓库**
- **选择器兼容性**：如 Outlook OWA 更新 UI，可能需要在 `config.json` 中更新 `selectors` 字段

---

## 📄 License

MIT License
