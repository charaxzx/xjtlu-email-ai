import asyncio
import sys

# Windows-specific event loop policy for Playwright compatibility
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
from typing import List, Optional
import os
import json
import logging
import traceback
from contextlib import asynccontextmanager

# Import functions from main.py without modifying it
from main import load_config, get_browser_page, search_emails, extract_full_body, call_llm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state to hold browser instance and search results
class AppState:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.email_results = []  # Stores full email objects including locators
        self.config = {}
        # ====== 改进二：进度状态追踪 ======
        self.current_stage = "idle"          # idle / connecting / searching / extracting / summarizing / done / error
        self.stage_detail = ""               # 当前阶段的详细信息
        self.extracted_count = 0             # 已提取的邮件数
        self.total_count = 0                 # 待提取的邮件总数

state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    # Load config on startup
    config_path = Path("config.json")
    if not config_path.exists():
        # Fallback if running from a different directory
        config_path = Path(__file__).parent / "config.json"
    
    state.config = load_config(config_path)
    print("Config loaded.")
    
    yield
    
    # Shutdown logic
    # Cleanup browser resources
    if state.context:
        await state.context.close()
    if state.browser:
        await state.browser.close()
    if state.playwright:
        await state.playwright.stop()
    print("Browser resources released.")

app = FastAPI(lifespan=lifespan)

# Global Exception Handler to ensure consistent JSON responses
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": str(exc)},
    )

# Setup templates
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# Pydantic models for request bodies
class SearchRequest(BaseModel):
    keyword: str

class ExecuteRequest(BaseModel):
    keyword: str
    instruction: str


def load_cookies_for_check(email_config: dict) -> list:
    """Load cookies from config JSON array and optional Netscape cookie file."""
    cookies = list(email_config.get("cookies", []))
    cookie_file = email_config.get("cookie_file")
    if not cookie_file:
        return cookies

    cookie_path = Path(__file__).parent / cookie_file
    if not cookie_path.exists():
        logger.warning(f"Cookie file not found: {cookie_path}")
        return cookies

    try:
        with open(cookie_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 7:
                    continue
                cookies.append(
                    {
                        "name": parts[5],
                        "value": parts[6],
                        "domain": parts[0],
                        "path": parts[2],
                        "secure": parts[3] == "TRUE",
                    }
                )
    except Exception as e:
        logger.warning(f"Failed reading cookie file {cookie_path}: {e}")

    return cookies


# ====== 改进二：带重试的浏览器初始化 ======
async def ensure_browser(max_retries: int = 2):
    """Ensure browser is running and page is ready, with automatic retry."""
    if state.page and not state.page.is_closed():
        return  # 浏览器已就绪

    last_error = None
    for attempt in range(1, max_retries + 1):
        logger.info(f"Initializing browser... (attempt {attempt}/{max_retries})")
        state.current_stage = "connecting"
        state.stage_detail = f"正在启动浏览器 (第 {attempt} 次尝试)" if attempt > 1 else "正在启动浏览器..."
        try:
            # 先清理上一次失败遗留的资源
            try:
                if state.context:
                    await state.context.close()
                if state.browser:
                    await state.browser.close()
                if state.playwright:
                    await state.playwright.stop()
            except Exception:
                pass

            state.playwright, state.browser, state.context, state.page = await get_browser_page(state.config)
            logger.info("Browser initialized successfully.")
            return  # 成功
        except Exception as e:
            last_error = e
            error_msg = f"Browser init attempt {attempt} failed: {repr(e)}"
            logger.warning(error_msg)
            if attempt < max_retries:
                await asyncio.sleep(2)  # 等待 2 秒后重试

    # 所有重试都失败了
    state.current_stage = "error"
    state.stage_detail = f"浏览器启动失败: {repr(last_error)}"
    raise HTTPException(status_code=500, detail=f"Failed to launch browser after {max_retries} attempts: {repr(last_error)}")


# ====== 改进二：进度状态 API ======
@app.get("/api/status")
async def get_status():
    """返回后端当前处理阶段，供前端轮询。"""
    return {
        "stage": state.current_stage,
        "detail": state.stage_detail,
        "extracted_count": state.extracted_count,
        "total_count": state.total_count,
    }

@app.post("/api/check_cookie")
async def check_cookie():
    """检查 Cookie 状态并更新 config"""
    logger.info("Checking cookie status...")
    
    # 1. 重新加载 config.json 以获取最新内容
    config_path = Path("config.json")
    if not config_path.exists():
        config_path = Path(__file__).parent / "config.json"
    
    try:
        new_config = load_config(config_path)
        state.config = new_config
        logger.info("Config reloaded for cookie check.")
    except Exception as e:
        logger.error(f"Failed to reload config: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": f"加载配置失败: {e}"})

    # 2. 清理可能存在的旧浏览器资源，确保使用新 Cookie 启动
    try:
        if state.context:
            await state.context.close()
        if state.browser:
            await state.browser.close()
        if state.playwright:
            await state.playwright.stop()
        state.context = None
        state.browser = None
        state.playwright = None
        state.page = None
    except Exception as e:
        logger.warning(f"Error while cleaning up old browser resources: {e}")

    # 3. 启动无头浏览器并测试访问
    test_playwright = None
    test_browser = None
    test_context = None
    test_page = None
    try:
        from playwright.async_api import async_playwright
        test_playwright = await async_playwright().start()
        # 使用无头模式在后台静默测试
        test_browser = await test_playwright.chromium.launch(headless=True, channel="msedge")
        test_context = await test_browser.new_context()
        
        email_config = state.config.get("email", {})
        
        # Load cookies from both config JSON and optional cookie_file.
        cookies = load_cookies_for_check(email_config)
        if cookies:
            await test_context.add_cookies(cookies)
        else:
            return {"status": "invalid", "message": "未读取到任何 Cookie，请检查 config.json 的 cookies 或 cookie_file 配置。"}
            
        test_page = await test_context.new_page()
        
        # 访问邮箱页面
        url_to_test = email_config.get("url", "https://mail.xjtlu.edu.cn/owa")
        if not url_to_test:
            # Although standard configuration handles it, keeping fallback keeps experience seamless for internal targets
            pass
        await test_page.goto(url_to_test, wait_until="networkidle", timeout=30000)
        await test_page.wait_for_timeout(2000)
        
        # 查找是否存在搜索框或者能够成功进入邮箱界面（而不是停留在 login 页面）
        current_url = test_page.url
        if "login" in current_url.lower() or "adfs" in current_url.lower():
            return {"status": "invalid", "message": "Cookie 已过期或无效，被重定向至登录页。"}
            
        # 进一步确认是否有邮件列表或搜索框存在
        target_frame = test_page
        for f in test_page.frames:
            try:
                if await f.locator("input").count() > 3:
                    target_frame = f
                    break
            except:
                pass
                
        # 简单检查是否有输入框或信封等元素
        input_count = await target_frame.locator("input").count()
        if input_count > 0:
            return {"status": "valid", "message": "Cookie 有效！成功访问邮箱。"}
        else:
             return {"status": "warning", "message": "已进入页面，但未找到预期的邮箱元素，可能 Cookie 即将过期或页面加载不完全。"}
             
    except Exception as e:
        logger.error(f"Cookie check failed: {e}")
        return {"status": "error", "message": f"检查过程中发生网络或超时错误: {str(e)[:100]}"}
    finally:
        # 清理测试用的浏览器
        try:
            if test_context: await test_context.close()
            if test_browser: await test_browser.close()
            if test_playwright: await test_playwright.stop()
        except:
            pass


@app.post("/api/execute")
async def execute_task(request: ExecuteRequest):
    try:
        await ensure_browser()
    except HTTPException:
        raise
    except Exception as e:
        state.current_stage = "error"
        state.stage_detail = str(e)
        raise
    
    keyword = request.keyword.strip()
        
    instruction = request.instruction.strip()
    if not instruction:
        instruction = "请用自然语气总结这10封邮件，重点关注活动、课程作业和重要事项。"
    
    print(f"Executing task: search='{keyword or '(最新邮件)'}', instruction='{instruction}'")
    
    try:
        # 1. Search
        state.current_stage = "searching"
        state.stage_detail = f"正在搜索关键词: {keyword}" if keyword else "正在获取最新邮件..."
        print(f"Searching for: {keyword}")
        try:
            emails = await search_emails(state.page, keyword, config=state.config)
        except RuntimeError as e:
            state.current_stage = "error"
            state.stage_detail = str(e)
            return JSONResponse(status_code=200, content={
                "status": "cookie_expired",
                "message": str(e),
            })
        state.email_results = emails # Store in state just in case
        
        # 2. Extract bodies — 改进二：单封邮件级别的错误容忍
        state.current_stage = "extracting"
        state.total_count = len(emails)
        state.extracted_count = 0
        state.stage_detail = f"正在提取 {len(emails)} 封邮件正文..."
        print(f"Extracting bodies for {len(emails)} emails...")
        full_bodies = []
        failed_count = 0
        for i, email in enumerate(emails):
            state.extracted_count = i + 1
            state.stage_detail = f"正在提取第 {i+1}/{len(emails)} 封邮件..."
            print(f"Extracting body for email {i+1}...")
            try:
                body = await extract_full_body(state.page, email["locator"])
            except Exception as e:
                logger.warning(f"Failed to extract email {i+1} body: {repr(e)}")
                body = f"[正文提取失败: {str(e)[:100]}]"
                failed_count += 1

            full_bodies.append(f"=== 邮件 {i+1}: {email['subject']} ===\n日期: {email['date']}\n\n{body}\n\n{'='*80}\n")
            await asyncio.sleep(0.2)

        if failed_count > 0:
            logger.info(f"{failed_count} email(s) failed body extraction, continuing with available data.")
            
        aggregated_text = "\n".join(full_bodies)
        
        # 3. Summarize — 智能分块 + 以用户指令为核心
        state.current_stage = "summarizing"
        from datetime import datetime
        loop = asyncio.get_event_loop()

        # ====== 动态分块：按字符数自动决定调用次数 ======
        CHARS_PER_CHUNK = 6000  # 每块约 6000 字符（约 2000-3000 token）
        aggregated_text = "\n".join(full_bodies)
        total_chars = len(aggregated_text)

        if total_chars <= CHARS_PER_CHUNK:
            # 文本较短，一次调用即可
            state.stage_detail = "AI 正在分析并生成回复..."
            summary_prompt = f"""
当前日期：{datetime.now().strftime('%Y-%m-%d')}

【用户指令（最高优先级，必须严格遵守）】：
{instruction}

请严格按照用户指令处理以下邮件。用户指令决定了你应该关注哪些邮件、输出什么内容、用什么风格。

【严禁编造】你只能基于下方提供的邮件原文内容作答。绝对不允许编造、臆测或虚构任何邮件标题、发件人或正文。如果某封邮件正文显示"提取失败"或内容为空，你必须如实说明，不得凭空捏造内容。

如果用户指令没有指定输出格式，则默认对涉及的每封邮件按以下格式输出：
- 邮件标题：[标题]
- 发件人：[发件人]
- 内容总结：[简练的总结]
- 重要程度：[1-5星，如 ★★★☆☆]

以下是全部 {len(full_bodies)} 封邮件的内容（供你根据用户指令选择性处理）：
{aggregated_text}
"""
            print(f"Calling LLM (single call, {total_chars} chars)...")
            summary = await loop.run_in_executor(None, call_llm, summary_prompt, state.config)
        else:
            # 文本较长，按字符数动态分块
            state.stage_detail = "AI 正在分批分析邮件..."
            chunks = []
            current_chunk = []
            current_chars = 0
            for body in full_bodies:
                body_len = len(body)
                if current_chars + body_len > CHARS_PER_CHUNK and current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_chars = 0
                current_chunk.append(body)
                current_chars += body_len
            if current_chunk:
                chunks.append("\n".join(current_chunk))

            print(f"Smart chunking: {total_chars} chars → {len(chunks)} chunks")

            chunk_summaries = []
            for idx, chunk_text in enumerate(chunks):
                state.stage_detail = f"AI 正在分析第 {idx+1}/{len(chunks)} 批邮件..."
                print(f"Calling LLM for chunk {idx+1}/{len(chunks)} ({len(chunk_text)} chars)...")
                chunk_prompt = f"""
当前日期：{datetime.now().strftime('%Y-%m-%d')}

【用户指令（最高优先级）】：
{instruction}

这是第 {idx+1}/{len(chunks)} 批邮件。请根据用户指令处理这批邮件。
【严禁编造】只能基于下方邮件原文作答，不得虚构任何内容。提取失败的邮件如实说明。
如果用户指令没有指定输出格式，则默认对涉及的每封邮件按以下格式输出：
- 邮件标题：[标题]
- 发件人：[发件人]
- 内容总结：[简练的总结]
- 重要程度：[1-5星，如 ★★★☆☆]

邮件内容：
{chunk_text}
"""
                chunk_summary = await loop.run_in_executor(None, call_llm, chunk_prompt, state.config)
                chunk_summaries.append(f"【第 {idx+1} 批分析结果】\n{chunk_summary}")
                
            state.stage_detail = "AI 正在生成最终总结..."
            print("Calling LLM for final summary...")
            aggregated_summaries = "\n\n".join(chunk_summaries)
            final_prompt = f"""
当前日期：{datetime.now().strftime('%Y-%m-%d')}

【用户指令（最高优先级，必须严格遵守）】：
{instruction}

前面我们已经分批处理了邮件并得到了各批次的分析结果。
请严格按照用户指令，汇总输出最终结果。如果用户只问了某封邮件，你只需要回答那封。
【严禁编造】不得虚构任何邮件信息，所有内容必须来自前面的分析结果。
如果用户指令没有指定输出格式，则默认对涉及的每封邮件按以下格式输出：
- 邮件标题：[标题]
- 发件人：[发件人]
- 内容总结：[简练的总结]
- 重要程度：[1-5星，如 ★★★☆☆]

各批次分析结果：
{aggregated_summaries}
"""
            summary = await loop.run_in_executor(None, call_llm, final_prompt, state.config)
        
        # Prepare sanitized email list for frontend display
        sanitized_emails = []
        for i, email in enumerate(emails):
            sanitized_emails.append({
                "id": i,
                "subject": email.get("subject", ""),
                "date": email.get("date", ""),
                "href": email.get("href", "")
            })

        state.current_stage = "done"
        state.stage_detail = "分析完成"

        return {
            "status": "success", 
            "summary": summary,
            "emails": sanitized_emails,
            "failed_extractions": failed_count,
        }
        
    except Exception as e:
        state.current_stage = "error"
        state.stage_detail = f"任务执行失败: {str(e)}"
        error_msg = f"Task execution failed: {repr(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    mail_url = state.config.get("email", {}).get("url", "https://mail.xjtlu.edu.cn/owa")
    return templates.TemplateResponse("index.html", {"request": request, "mail_url": mail_url})

@app.post("/api/search")
async def search(request: SearchRequest):
    return {"status": "deprecated", "message": "Use /api/execute instead"}

@app.post("/api/summarize")
async def summarize(request: Request): # Changed to Request to avoid undefined model
    return {"status": "deprecated", "message": "Use /api/execute instead"}

if __name__ == "__main__":
    # Disable reload to avoid Windows subprocess loop issues
    # Use port 8001 to avoid conflicts with hung processes
    uvicorn.run("app:app", host="127.0.0.1", port=8001, reload=False, loop="asyncio")
