version2026.4.10 — runtime bundle (email assistant)

Quick start (Windows):
1. Python 3.10+ installed, with PATH.
2. Unzip this folder anywhere.
3. Open Command Prompt in this folder:
   python -m venv .venv
   .venv\Scripts\pip install -r src\requirements.txt
   .venv\Scripts\playwright install chromium
4. Copy src\config.example.json to src\config.json and edit if needed.
5. Double-click run_app.bat (or: cd src && ..\.venv\Scripts\python app.py)

Optional: copy .env.example to .env for secrets.

---
版本包说明（中文）
1. 需要 Python 3.10+。
2. 解压后在本目录创建虚拟环境并安装依赖（见上）。
3. playwright install chromium 用于浏览器自动化。
4. 将 src\config.example.json 复制为 src\config.json 并按需填写。
5. 运行 run_app.bat 启动，浏览器访问 http://localhost:8001

本包不含：用户数据库 user.db、config.json、Cookie 导出等敏感文件，需自行配置。
