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
from pydantic import BaseModel, Field
import uvicorn
from typing import List, Optional
import os
import json
import logging
import traceback
from contextlib import asynccontextmanager

# Import functions from main.py without modifying it
from main import (
    load_config,
    get_browser_page,
    search_emails,
    extract_full_body,
    call_llm,
    build_per_email_analysis_prompt,
    build_final_merge_prompt,
    format_human_email_fragment,
    LLM_PARALLEL_BATCH_SIZE,
    normalize_parallel_llm_result,
    total_extracted_body_words,
)

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
    email_count: int = Field(default=10, ge=1, le=10)

class ConfigUpdateRequest(BaseModel):
    ai_base_url: str
    ai_api_key: str
    ai_model: str
    email_url: str
    email_cookies: str



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


def _normalize_cookie_dicts(items: list) -> list:
    """Ensure each cookie dict is suitable for config + Playwright add_cookies."""
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if name is None:
            continue
        val = item.get("value")
        if val is None:
            val = ""
        elif not isinstance(val, str):
            val = str(val)
        domain = item.get("domain")
        path = item.get("path")
        c = {
            "name": str(name),
            "value": val,
            "domain": "" if domain is None else str(domain),
            "path": "/" if not path else str(path),
        }
        sec = item.get("secure")
        if sec is True or (isinstance(sec, str) and sec.lower() == "true"):
            c["secure"] = True
        ho = item.get("httpOnly")
        if ho is True or (isinstance(ho, str) and ho.lower() == "true"):
            c["httpOnly"] = True
        exp = item.get("expires")
        if exp is None:
            exp = item.get("expirationDate")
        if exp is not None:
            try:
                c["expires"] = float(exp)
            except (TypeError, ValueError):
                pass
        ss = item.get("sameSite")
        if isinstance(ss, str) and ss in ("Strict", "Lax", "None"):
            c["sameSite"] = ss
        out.append(c)
    return out


def _parse_netscape_cookie_text(text: str) -> list:
    cookies = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 7:
            cookies.append(
                {
                    "name": parts[5],
                    "value": parts[6],
                    "domain": parts[0],
                    "path": parts[2],
                    "secure": parts[3].upper() == "TRUE",
                }
            )
    return cookies


def parse_email_cookies_blob(raw: Optional[str]) -> tuple[list, Optional[str]]:
    """
    Cookie-Editor：JSON 数组；或 Netscape / curl 导出的制表符分隔多行。
    返回 (cookies_list, error_message)；成功时 error_message 为 None。
    """
    if raw is None:
        return [], None
    s = raw.strip()
    if not s:
        return [], None
    if s.startswith("\ufeff"):
        s = s.lstrip("\ufeff").strip()

    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            normalized = _normalize_cookie_dicts(parsed)
            if not normalized and len(parsed) > 0:
                return [], "Cookie JSON 中未找到带 name 字段的有效条目。"
            return normalized, None
        if isinstance(parsed, dict):
            return [], "Cookies 必须是 JSON 数组（用方括号 [] 包裹多条 cookie）。"
        return [], "Cookies 必须是一个 JSON 数组。"
    except json.JSONDecodeError:
        pass

    netscape = _normalize_cookie_dicts(_parse_netscape_cookie_text(s))
    if netscape:
        return netscape, None

    return [], (
        "无法解析 Cookie。请粘贴 Cookie-Editor 导出的 JSON 数组，"
        "或 Netscape 格式的 cookie 文本（含制表符的多行）。"
    )


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

@app.get("/api/config")
async def get_config():
    """获取当前配置供前端显示"""
    # 重新加载以确保最新
    config_path = Path("config.json")
    if not config_path.exists():
        config_path = Path(__file__).parent / "config.json"
    
    current_config = load_config(config_path)
    
    # 将 cookie list 转成带格式的 JSON 字符串供前端多行文本框显示
    cookies_list = current_config.get("email", {}).get("cookies", [])
    cookies_str = json.dumps(cookies_list, indent=2, ensure_ascii=False) if cookies_list else ""
    
    return {
        "ai_base_url": current_config.get("ai", {}).get("base_url", ""),
        "ai_api_key": current_config.get("ai", {}).get("api_key", ""),
        "ai_model": current_config.get("ai", {}).get("model", "gpt-4o-mini"),
        "email_url": current_config.get("email", {}).get("url", "https://mail.xjtlu.edu.cn/owa"),
        "email_cookies": cookies_str
    }

@app.post("/api/config")
async def update_config(req: ConfigUpdateRequest):
    """保存用户在前端修改的配置"""
    config_path = Path("config.json")
    if not config_path.exists():
        config_path = Path(__file__).parent / "config.json"
        
    current_config = load_config(config_path)

    cookies_list, cookie_err = parse_email_cookies_blob(req.email_cookies)
    if cookie_err:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": cookie_err},
        )

    # 更新 AI 配置
    if "ai" not in current_config:
        current_config["ai"] = {}
    current_config["ai"]["base_url"] = req.ai_base_url.strip()
    current_config["ai"]["api_key"] = req.ai_api_key.strip()
    current_config["ai"]["model"] = req.ai_model.strip()

    # 更新 Email 配置
    if "email" not in current_config:
        current_config["email"] = {}
    current_config["email"]["url"] = req.email_url.strip()
    # 确保 login_type 是 cookie
    current_config["email"]["login_type"] = "cookie"
    current_config["email"]["cookies"] = cookies_list

    # 写回文件
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(current_config, f, indent=4, ensure_ascii=False)
        
        # 更新内存态
        state.config = current_config

        # 保存设置后清理浏览器状态；清理失败不应导致用户看到“保存失败”（文件已写入）
        try:
            if state.context:
                await state.context.close()
            if state.browser:
                await state.browser.close()
            if state.playwright:
                await state.playwright.stop()
        except Exception as cleanup_err:
            logger.warning("Browser cleanup after config save: %s", cleanup_err)
        finally:
            state.context = None
            state.browser = None
            state.playwright = None
            state.page = None

        return {"status": "success", "message": "配置已保存。"}
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": f"保存配置时出错: {e}"})


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
        instruction = "请用自然语气总结这些邮件，重点关注活动、课程作业和重要事项。"
    
    print(f"Executing task: search='{keyword or '(最新邮件)'}', instruction='{instruction}'")
    
    try:
        # 1. Search
        state.current_stage = "searching"
        state.stage_detail = f"正在搜索关键词: {keyword}" if keyword else "正在获取最新邮件..."
        print(f"Searching for: {keyword}")
        try:
            emails = await search_emails(
                state.page,
                keyword,
                config=state.config,
                max_emails=request.email_count,
            )
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
        extracted_items = []
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

            extracted_items.append(
                {
                    "index": i + 1,
                    "subject": email.get("subject", ""),
                    "date": email.get("date", ""),
                    "body": body,
                }
            )
            await asyncio.sleep(0.2)

        if failed_count > 0:
            logger.info(f"{failed_count} email(s) failed body extraction, continuing with available data.")
            
        # 3. Summarize — 每封邮件并行 1 次 LLM，再 1 次汇总
        state.current_stage = "summarizing"
        from datetime import datetime
        loop = asyncio.get_running_loop()

        sanitized_emails = []
        for i, email in enumerate(emails):
            sanitized_emails.append({
                "id": i,
                "subject": email.get("subject", ""),
                "date": email.get("date", ""),
                "href": email.get("href", "")
            })

        if not extracted_items:
            state.current_stage = "done"
            state.stage_detail = "分析完成"
            return {
                "status": "success",
                "summary": "没有可分析的邮件。",
                "emails": sanitized_emails,
                "failed_extractions": failed_count,
            }

        n = len(extracted_items)
        body_words = total_extracted_body_words(extracted_items)
        today = datetime.now().strftime("%Y-%m-%d")
        total_batches = (n + LLM_PARALLEL_BATCH_SIZE - 1) // LLM_PARALLEL_BATCH_SIZE
        logger.info(
            f"邮件正文总词数: {body_words}；每批并发 {LLM_PARALLEL_BATCH_SIZE} 次，"
            f"共 {total_batches} 批单封 LLM + 1 次汇总 LLM"
        )

        async def run_llm_task(prompt: str) -> str:
            return await loop.run_in_executor(None, call_llm, prompt, state.config)

        state.stage_detail = f"准备分 {total_batches} 批并行分析 {n} 封邮件…"
        prompts = []
        for i, item in enumerate(extracted_items):
            human = format_human_email_fragment(
                str(item.get("subject", "") or ""),
                str(item.get("date", "") or ""),
                str(item.get("body") or ""),
            )
            prompts.append(
                build_per_email_analysis_prompt(
                    today=today,
                    instruction=instruction,
                    email_human_text=human,
                    email_index=i + 1,
                    email_total=n,
                )
            )

        normalized = []
        for batch_index, start in enumerate(range(0, n, LLM_PARALLEL_BATCH_SIZE), start=1):
            batch_prompts = prompts[start:start + LLM_PARALLEL_BATCH_SIZE]
            state.stage_detail = (
                f"并行分析第 {batch_index}/{total_batches} 批邮件"
                f"（本批 {len(batch_prompts)} 封）…"
            )
            print(
                f"Parallel LLM batch {batch_index}/{total_batches}: "
                f"{len(batch_prompts)} per-email calls…"
            )
            parallel_out = await asyncio.gather(
                *[run_llm_task(p) for p in batch_prompts],
                return_exceptions=True,
            )
            normalized.extend(normalize_parallel_llm_result(r) for r in parallel_out)
        per_email_sections = "\n\n".join(
            f"—— 第 {i + 1} 封邮件的初步分析 ——\n{t}" for i, t in enumerate(normalized)
        )
        final_prompt = build_final_merge_prompt(
            today=today,
            instruction=instruction,
            per_email_sections=per_email_sections,
        )
        state.stage_detail = "AI 正在生成最终总结…"
        print("Calling LLM for final summary…")
        try:
            summary = await run_llm_task(final_prompt)
        except Exception as exc:
            logger.exception("Final LLM call failed")
            summary = f"最终汇总失败：{exc}"

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
