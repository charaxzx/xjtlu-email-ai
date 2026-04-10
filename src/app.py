import asyncio
import sys

# Windows-specific event loop policy for Playwright compatibility
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import uvicorn
from typing import List, Optional, Tuple
import os
import json
import base64
import logging
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from starlette.middleware.sessions import SessionMiddleware

import auth_db
from digest_schedule import compute_next_run_hint, merge_digest_defaults

# Import functions from main.py without modifying it
from main import (
    load_config,
    get_browser_page,
    search_emails,
    extract_full_body,
    _LIST_ITEM_SELECTOR,
    _prepare_owa_mail_list_frame,
    _reset_owa_mail_list_scroll,
    _scroll_owa_mail_list_step,
    call_llm,
    build_per_email_analysis_prompt,
    build_final_merge_prompt,
    format_human_email_fragment,
    LLM_PARALLEL_BATCH_SIZE,
    normalize_parallel_llm_result,
    total_extracted_body_words,
    classify_email,
    parse_email_date_for_filter,
    EMAIL_CATEGORY_LABELS,
    merge_list_and_pane_datetime,
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
        # idle / connecting / searching / extracting / summarizing / done / error
        self.current_stage = "idle"
        self.stage_detail = ""               # 当前阶段的详细信息
        self.extracted_count = 0             # 已提取的邮件数
        self.total_count = 0                 # 待提取的邮件总数
        # 服务上线后首次进入首页时自动做一次无头 Cookie 检查（与手动「检查 Cookie」一致）
        # pending|checking|valid|invalid|warning|error|skipped
        self.auto_cookie_status = "pending"
        self.auto_cookie_message = ""
        self.auto_cookie_checked_at: Optional[str] = None
        self._auto_cookie_check_ran = False  # 是否已安排过「上线自动检查一次」
        # 当前预启动浏览器对应的用户（避免 A 登录却用 B 的 Cookie 打开邮箱）
        self.prelaunch_user_id: Optional[int] = None
        # 交互式邮箱登录：弹窗浏览器未注入 Cookie，用户手动登录后点「我已登录」抓取会话
        self.interactive_login_user_id: Optional[int] = None
        # 深度扫描：与 /api/execute 的 indices 子集对应，用于校验关键词一致
        self.deep_scan_keyword: str = ""
        # 最近一次深度扫描的完整导出（正文+分类），与 deep_scan_result.json 同步
        self.deep_scan_export: Optional[dict] = None


state = AppState()

# 多用户本地服务时串行执行邮件任务，避免全局 state.config 被并发覆盖
execute_lock = asyncio.Lock()
# 预启动仅按「当前登录用户」串行执行，避免重复任务
prelaunch_lock = asyncio.Lock()
# 首次首页自动 Cookie 检查只触发一次
auto_cookie_lock = asyncio.Lock()


def _session_user_id(request: Request) -> Optional[int]:
    raw = request.session.get("user_id")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def require_user_id(request: Request) -> int:
    uid = _session_user_id(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="请先登录")
    return uid


async def probe_mail_session_on_page(page) -> dict:
    """
    在已加载的邮箱页面上判断会话是否像有效 OWA（与手动「检查 Cookie」同一套启发式）。
    返回 {"status": "valid"|"invalid"|"warning"|"error", "message": str}
    """
    try:
        current_url = page.url or ""
    except Exception as exc:
        return {"status": "error", "message": f"无法读取页面: {exc}"}

    lu = current_url.lower()
    if "login" in lu or "adfs" in lu:
        return {
            "status": "invalid",
            "message": "Cookie 已过期或无效，被重定向至登录页。",
        }

    target_frame = page
    for f in page.frames:
        try:
            if await f.locator("input").count() > 3:
                target_frame = f
                break
        except Exception:
            continue

    try:
        input_count = await target_frame.locator("input").count()
    except Exception as exc:
        return {"status": "error", "message": f"检测页面元素失败: {str(exc)[:80]}"}

    if input_count > 0:
        return {
            "status": "valid",
            "message": "Cookie 有效。",
        }
    return {
        "status": "warning",
        "message": "已进入页面，但未找到预期的邮箱元素，Cookie 可能即将失效。",
    }


async def _prelaunch_browser():
    """后台预启动可见浏览器（仅打开邮箱，不再做预启动页 Cookie 检测）。"""
    try:
        logger.info("预启动浏览器中...")
        await ensure_browser(max_retries=2)
        logger.info("浏览器预启动完成。")
    except Exception as e:
        logger.warning(f"浏览器预启动失败（不影响手动使用）: {repr(e)}")


async def close_browser_resources() -> None:
    """关闭 Playwright 浏览器实例（切换登录用户或退出时调用）。"""
    try:
        # 释放交互式登录加锁标记，防止触发 page.on("close") 时重复触发清理死锁
        state.interactive_login_user_id = None
        if state.context:
            await state.context.close()
        if state.browser:
            await state.browser.close()
        if state.playwright:
            await state.playwright.stop()
    except Exception as exc:
        logger.warning("close_browser_resources: %s", exc)
    finally:
        state.context = None
        state.browser = None
        state.playwright = None
        state.page = None
        state.email_results = []
        state.prelaunch_user_id = None
        state.interactive_login_user_id = None


async def cleanup_on_page_close(uid: int):
    """Playwright 页面关闭事件的单独异步处理器。"""
    async with prelaunch_lock:
        if state.interactive_login_user_id == uid:
            logger.info(f"检测到用户 {uid} 手动关闭了浏览器页面，正在释放资源。")
            await close_browser_resources()


async def cleanup_on_visible_mail_page_close() -> None:
    """预启动或分析任务使用的可见邮箱窗口被关闭时释放资源（交互式登录由 cleanup_on_page_close 处理）。"""
    async with prelaunch_lock:
        if state.interactive_login_user_id is not None:
            return
        logger.info("可见邮箱浏览器窗口已关闭，正在释放 Playwright 资源。")
        await close_browser_resources()


def _playwright_cookies_to_config_list(raw: list) -> list:
    """将 Playwright context.cookies() 结果转为可写入 config 的 cookie 列表。"""
    out = []
    for c in raw or []:
        if not isinstance(c, dict) or c.get("name") is None:
            continue
        d = {
            "name": str(c["name"]),
            "value": "" if c.get("value") is None else str(c["value"]),
            "domain": "" if c.get("domain") is None else str(c["domain"]),
            "path": "/" if not c.get("path") else str(c["path"]),
        }
        if c.get("secure"):
            d["secure"] = True
        if c.get("httpOnly"):
            d["httpOnly"] = True
        ss = c.get("sameSite")
        if isinstance(ss, str) and ss in ("Strict", "Lax", "None"):
            d["sameSite"] = ss
        exp = c.get("expires")
        if exp is not None and float(exp) > 0:
            try:
                d["expires"] = float(exp)
            except (TypeError, ValueError):
                pass
        out.append(d)
    return _normalize_cookie_dicts(out)


async def schedule_prelaunch_for_user(user_id: int) -> None:
    """
    仅使用「当前登录用户」的配置预启动可见浏览器（不做 Cookie 预检测）。
    """
    async with prelaunch_lock:
        cfg = auth_db.load_user_config(user_id)
        email_cfg = cfg.get("email", {})
        cookies = load_cookies_for_check(email_cfg)

        # 若其他用户正在交互登录，先关闭（全局仅一个 Playwright 实例）
        if state.interactive_login_user_id is not None and state.interactive_login_user_id != user_id:
            await close_browser_resources()
        # 当前用户正在交互登录向导时，不要预启动带 Cookie 的浏览器以免顶替窗口
        if state.interactive_login_user_id == user_id:
            return

        # 切换登录用户时先关掉旧浏览器，避免沿用上一账号的会话
        if state.prelaunch_user_id is not None and state.prelaunch_user_id != user_id:
            await close_browser_resources()
            state.prelaunch_user_id = None

        if not cfg.get("browser", {}).get("prelaunch", False):
            return

        if not cookies:
            return

        if (
            state.prelaunch_user_id == user_id
            and state.page
            and not state.page.is_closed()
        ):
            return

        await close_browser_resources()
        state.prelaunch_user_id = None
        state.config = cfg

        try:
            await _prelaunch_browser()
        except Exception as exc:
            logger.warning("schedule_prelaunch_for_user: %s", exc)

        if state.page and not state.page.is_closed():
            state.prelaunch_user_id = user_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    auth_db.init_db()
    auth_db.ensure_seed_user_and_migrate_legacy()

    # 不在启动时加载「某一用户」的配置到全局浏览器，避免多账号串 Cookie
    state.config = {}
    state.prelaunch_user_id = None
    state.interactive_login_user_id = None
    state.auto_cookie_status = "pending"
    state.auto_cookie_message = "打开首页后将自动检查 Cookie 一次（无头检测）。"
    state.auto_cookie_checked_at = None
    state._auto_cookie_check_ran = False
    print("DB ready. Auto cookie check runs once on first home page load.")

    yield

    # Shutdown logic
    await close_browser_resources()
    print("Browser resources released.")

app = FastAPI(lifespan=lifespan)

_session_secret = os.environ.get("SESSION_SECRET", "dev-local-change-me")
app.add_middleware(
    SessionMiddleware,
    secret_key=_session_secret,
    max_age=14 * 24 * 3600,
    same_site="lax",
    https_only=False,
)

# Global Exception Handler to ensure consistent JSON responses


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": str(exc)},
    )

# Setup templates
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# 深度模式：列表与单次分析上限（深度扫描、正文提取、前端文案一致，最多 100 封）
DEEP_MAX_EMAILS = 100
# 深度扫描落盘：完整正文 + 分类，供深度分析离线读取（不碰浏览器）
DEEP_SCAN_RESULT_JSON = Path(
    __file__).resolve().parent / "deep_scan_result.json"
# 与快照同名的 JSON/PNG 导出目录（见前端提示 deep_scan_exports/）
DEEP_SCAN_EXPORTS_DIR = Path(__file__).resolve().parent / "deep_scan_exports"


def _deep_scan_export_stem_from_iso(exported_at: str) -> str:
    s = (exported_at or "").strip()
    if not s:
        return "deep_scan_export"
    safe = s.replace(":", "-").replace("T", "_")[:80]
    return f"deep_scan_{safe}"


def _load_deep_scan_export_from_disk() -> Optional[dict]:
    """从 deep_scan_result.json 读回最近一次深度扫描结果（服务重启后仍可用）。"""
    try:
        if not DEEP_SCAN_RESULT_JSON.is_file():
            return None
        with DEEP_SCAN_RESULT_JSON.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.exception("读取 %s 失败", DEEP_SCAN_RESULT_JSON)
        return None


def _save_deep_scan_result_json(export_doc: dict) -> None:
    with DEEP_SCAN_RESULT_JSON.open("w", encoding="utf-8") as f:
        json.dump(export_doc, f, ensure_ascii=False, indent=2)


DAILY_MAX_EMAILS = 10


async def dev_style_locator_for_convid_from_list_top(page, keyword: str, config: dict, cvid: str, fallback):
    """
    与 /api/dev/extract_sample_bodies 一致：从邮件列表顶重置后向下慢滚，
    直到 data-convid 匹配的行进入视口再返回。避免虚拟列表下缓存 locator 失效导致正文像预览/截断。
    """
    if not (cvid or "").strip():
        return fallback
    try:
        target_frame = await _prepare_owa_mail_list_frame(page, keyword, config)
        await _reset_owa_mail_list_scroll(target_frame)
        await page.wait_for_timeout(280)

        max_steps = 240
        for step in range(max_steps):
            rows = target_frame.locator(_LIST_ITEM_SELECTOR)
            n = await rows.count()
            for i in range(min(n, 500)):
                row = rows.nth(i)
                v = await row.get_attribute("data-convid")
                if v == cvid:
                    await row.scroll_into_view_if_needed()
                    await page.wait_for_timeout(45)
                    return row
            await _scroll_owa_mail_list_step(
                target_frame,
                container_fraction=0.36,
                min_delta=56,
            )
            await page.wait_for_timeout(150)
            if (step + 1) % 20 == 0:
                logger.info(
                    "dev_style_locator_for_convid step=%s/%s cvid=%s",
                    step + 1,
                    max_steps,
                    cvid[:48],
                )
    except Exception:
        logger.warning(
            "dev_style_locator_for_convid_from_list_top failed for %s",
            (cvid or "")[:48],
        )
    return fallback


async def search_deep_emails_for_extraction(page, keyword: str, config: dict, *, max_emails: int):
    """
    深度正文提取统一走开发者测试同款列表拉取参数。
    深度正文提取与 deep_scan 均受 DEEP_MAX_EMAILS 约束。
    """
    return await search_emails(
        page,
        keyword,
        config=config,
        max_emails=max(1, min(int(max_emails), DEEP_MAX_EMAILS)),
        mode="deep",
        sort_by_date=False,
        dedupe_convid=True,
        list_scroll_pause_ms=200,
        list_scroll_step_fraction=0.38,
        deep_list_initial_wait_ms=480,
        deep_stagnation_pause_ms=900,
        deep_stagnation_limit=12,
    )


async def dev_style_deep_extract_to_export(
    page,
    keyword: str,
    config: dict,
    *,
    max_list_emails: int,
    ui_label: str = "〔开发者〕",
    write_debug_artifacts: bool = True,
) -> Tuple[dict, list]:
    """
    与 /api/dev/extract_sample_bodies 完全同一套：拉深度列表 → 准备列表帧 →
    顺序 convid 定位 → fast_activation 抽正文 → 组装 export 文档。
    不写磁盘；由调用方决定是否落盘。返回 (export_doc, emails)。
    """
    kw = (keyword or "").strip()
    pool = max(1, min(int(max_list_emails), DEEP_MAX_EMAILS))
    state.current_stage = "searching"
    emails = await search_deep_emails_for_extraction(
        page, kw, config, max_emails=pool
    )
    n = len(emails)
    state.current_stage = "extracting"
    samples: list = []
    extract_indices = range(1, n + 1)

    debug_log_path = Path(__file__).resolve().parent / \
        "dev_extract_debug.jsonl"
    screenshot_dir = Path(__file__).resolve().parent / "debug_screenshots"
    if write_debug_artifacts:
        debug_log_path.write_text("", encoding="utf-8")
        screenshot_dir.mkdir(parents=True, exist_ok=True)

    target_frame = None
    try:
        target_frame = await _prepare_owa_mail_list_frame(page, kw, config)
        await _reset_owa_mail_list_scroll(target_frame)
        await page.wait_for_timeout(400)
    except Exception as e:
        logger.error(
            "dev_style_deep_extract_to_export: prepare target_frame: %s", repr(e))
        target_frame = None

    for ord_ in extract_indices:
        if ord_ > 1 and (ord_ - 1) % 20 == 0:
            logger.info(
                "dev_style_deep_extract batch cooling: sleeping 5s at ord=%s", ord_)
            state.stage_detail = f"{ui_label}防热保护休眠 5秒 (已完成 {ord_ - 1} 封)…"
            await asyncio.sleep(5)

        state.stage_detail = f"{ui_label}正在提取第 {ord_} 封正文…"
        em = emails[ord_ - 1]
        cvid = (em.get("convid") or "").strip()
        subj = (em.get("subject") or "").strip()

        loc = await dev_style_locator_sequential(target_frame, cvid)
        dt = (em.get("date") or "").strip()
        d_disp = (em.get("date_display") or "").strip()
        body = ""
        err = None
        pane_disp, pane_sort = "", ""
        if loc is None:
            err = "列表项无 locator"
            body = f"[正文提取失败: {err}]"
        else:
            try:
                body, pane_disp, pane_sort = await extract_full_body(
                    page,
                    loc,
                    expected_subject=subj,
                    fast_activation=True,
                )
            except Exception as exc:
                logger.warning(
                    "dev_style_deep_extract ord=%s: %s", ord_, repr(exc))
                err = str(exc)[:200]
                body = f"[正文提取失败: {err}]"

        merged_disp, merged_sort = merge_list_and_pane_datetime(
            dt, d_disp, pane_sort, pane_disp
        )
        final_date = merged_sort or dt
        final_disp = merged_disp or merged_sort or dt
        sender = (em.get("sender") or "").strip()
        href = (em.get("href") or "").strip()
        res_item = {
            "index": ord_,
            "subject": subj,
            "date": final_date,
            "date_display": final_disp,
            "sender": sender,
            "convid": cvid or None,
            "href": href,
            "body": body,
            "body_chars": len(body or ""),
            "ok": err is None
            and bool(body)
            and not str(body).startswith("[正文提取失败"),
        }
        samples.append(res_item)

        if write_debug_artifacts:
            try:
                with debug_log_path.open("a", encoding="utf-8") as df:
                    log_entry = {
                        "ts": datetime.now().isoformat(),
                        "ord": ord_,
                        "subj": subj,
                        "ok": res_item["ok"],
                        "chars": res_item["body_chars"],
                        "error": err,
                    }
                    df.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                if not res_item["ok"]:
                    ss_path = screenshot_dir / f"error_{ord_}.png"
                    await page.screenshot(path=str(ss_path))
            except Exception:
                pass
        await asyncio.sleep(0.04)

    exported_at = datetime.now().isoformat(timespec="seconds")
    export_doc = {
        "format": "dev_extract_sample_bodies",
        "version": 1,
        "exported_at": exported_at,
        "keyword": kw,
        "list_count": n,
        "indices": list(extract_indices),
        "samples": samples,
    }
    return export_doc, emails


async def dev_style_locator_sequential(target_frame, cvid: str):
    """
    单向流式向下寻找指定的 convid，依序遍历。
    如果向下找了 80 步没找到（可能由于跳题或滚动过快被抛在上面），则执行一次全局回顶重新往下找。
    """
    if not (cvid or "").strip() or target_frame is None:
        return None

    async def _scan_down(max_s):
        for step in range(max_s):
            rows = target_frame.locator(_LIST_ITEM_SELECTOR)
            n = await rows.count()
            for i in range(min(n, 200)):
                row = rows.nth(i)
                v = await row.get_attribute("data-convid")
                if v == cvid:
                    precise_loc = target_frame.locator(
                        f'[data-convid="{cvid}"]').first
                    await precise_loc.scroll_into_view_if_needed()
                    await asyncio.sleep(0.045)
                    return precise_loc
            await _scroll_owa_mail_list_step(
                target_frame,
                container_fraction=0.45,
                min_delta=56,
            )
            await asyncio.sleep(0.18)
        return None

    try:
        # 第一阶段：顺流向下找
        found = await _scan_down(80)
        if found:
            return found

        # 第二阶段：没找到，说明可能在上面，或者刚才没刷出来，兜底重置回顶再找一遍
        logger.warning(
            f"dev_style_locator_sequential missed {cvid[:20]}, triggering fallback to top...")
        await _reset_owa_mail_list_scroll(target_frame)
        await asyncio.sleep(0.5)
        return await _scan_down(80)

    except Exception as e:
        logger.warning("dev_style_locator_sequential failed: %s", repr(e))

    return None


# Pydantic models for request bodies
class SearchRequest(BaseModel):
    keyword: str


class ExecuteRequest(BaseModel):
    keyword: str = ""
    instruction: str
    mode: str = Field(default="daily", description="daily or deep")
    email_count: int = Field(default=10, ge=1, le=DEEP_MAX_EMAILS)
    indices: Optional[List[int]] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None


class DeepScanRequest(BaseModel):
    keyword: str = ""


class DeepScanChartSaveRequest(BaseModel):
    export_stem: str = ""
    png_base64: str = ""


class DevExtractSampleBodiesRequest(BaseModel):
    """开发者：仅提取指定序号的完整正文，不调 LLM。"""
    keyword: str = ""


class DevExtractDailyBodiesRequest(BaseModel):
    """开发者测试2：与日常模式相同流程（daily search_emails + 逐封 extract_full_body），不调 LLM。"""
    keyword: str = ""
    email_count: int = Field(default=DAILY_MAX_EMAILS,
                             ge=1, le=DAILY_MAX_EMAILS)


class ConfigUpdateRequest(BaseModel):
    ai_base_url: str
    ai_api_key: str
    ai_model: str
    email_url: str
    email_cookies: str


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class UpdateUsernameRequest(BaseModel):
    new_username: str


class DigestSettingsRequest(BaseModel):
    enabled: bool = False
    cadence: str = "daily"
    local_time: str = "08:00"
    timezone: str = ""
    weekday: int = 0
    keyword: str = ""
    instruction: str = "请用自然语气总结这些邮件，重点关注活动、课程作业和重要事项。"
    mode: str = "daily"
    email_count: int = Field(default=10, ge=1, le=10)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if _session_user_id(request) is not None:
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(request, "login.html", {"request": request})


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    if _session_user_id(request) is not None:
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(request, "register.html", {"request": request})


@app.post("/api/auth/login")
async def api_login(request: Request, body: LoginRequest):
    user = auth_db.verify_login(body.email, body.password)
    if not user:
        return JSONResponse(
            status_code=401,
            content={"status": "error", "message": "邮箱或密码错误"},
        )
    request.session["user_id"] = user["id"]
    request.session["username"] = user["username"]
    request.session["email"] = user["email"]
    asyncio.create_task(schedule_prelaunch_for_user(user["id"]))
    return {"status": "success", "user": user}


@app.post("/api/auth/register")
async def api_register(request: Request, body: RegisterRequest):
    ok, msg = auth_db.create_user(body.username, body.email, body.password)
    if not ok:
        return JSONResponse(status_code=400, content={"status": "error", "message": msg})
    user = auth_db.verify_login(body.email, body.password)
    if user:
        request.session["user_id"] = user["id"]
        request.session["username"] = user["username"]
        request.session["email"] = user["email"]
        asyncio.create_task(schedule_prelaunch_for_user(user["id"]))
    return {"status": "success", "message": "注册成功", "user": user}


@app.post("/api/auth/logout")
async def api_logout(request: Request):
    async with prelaunch_lock:
        await close_browser_resources()
        state.prelaunch_user_id = None
    request.session.clear()
    return {"status": "success"}


@app.get("/api/auth/me")
async def api_me(request: Request):
    uid = _session_user_id(request)
    if uid is None:
        return JSONResponse(status_code=401, content={"status": "error", "message": "未登录"})
    u = auth_db.get_user_by_id(uid)
    if not u:
        request.session.clear()
        return JSONResponse(status_code=401, content={"status": "error", "message": "会话无效"})
    return {
        "status": "success",
        "id": u["id"],
        "username": u["username"],
        "email": u["email"],
    }


@app.post("/api/auth/update_username")
async def api_update_username(request: Request, body: UpdateUsernameRequest):
    uid = require_user_id(request)
    ok, msg = auth_db.update_username(uid, body.new_username)
    if not ok:
        return JSONResponse(status_code=400, content={"status": "error", "message": msg})
    request.session["username"] = body.new_username
    return {"status": "success", "message": "用户名已更新", "username": body.new_username}


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


async def cookie_check_headless(email_config: dict) -> dict:
    """
    独立无头浏览器访问邮箱并探测 Cookie（与手动「检查 Cookie」逻辑一致）。
    返回 {"status": str, "message": str}，不含 HTTP 包装。
    """
    test_playwright = None
    test_browser = None
    test_context = None
    test_page = None
    try:
        from playwright.async_api import async_playwright

        test_playwright = await async_playwright().start()
        test_browser = await test_playwright.chromium.launch(
            headless=True, channel="msedge"
        )
        test_context = await test_browser.new_context()

        cookies = load_cookies_for_check(email_config)
        if not cookies:
            return {
                "status": "invalid",
                "message": "未读取到任何 Cookie，请在配置中填写 cookies 或 cookie_file。",
            }

        await test_context.add_cookies(cookies)
        test_page = await test_context.new_page()

        url_to_test = email_config.get("url", "https://mail.xjtlu.edu.cn/owa") or (
            "https://mail.xjtlu.edu.cn/owa"
        )
        await test_page.goto(url_to_test, wait_until="networkidle", timeout=30000)
        await test_page.wait_for_timeout(2000)

        outcome = await probe_mail_session_on_page(test_page)
        if outcome["status"] == "valid":
            return {"status": "valid", "message": "Cookie 有效！成功访问邮箱。"}
        if outcome["status"] == "invalid":
            return {"status": "invalid", "message": outcome["message"]}
        if outcome["status"] == "warning":
            return {
                "status": "warning",
                "message": "已进入页面，但未找到预期的邮箱元素，可能 Cookie 即将过期或页面加载不完全。",
            }
        return {"status": "error", "message": outcome.get("message", "检测失败")}
    except Exception as e:
        logger.error(f"cookie_check_headless failed: {e}")
        return {
            "status": "error",
            "message": f"检查过程中发生网络或超时错误: {str(e)[:100]}",
        }
    finally:
        try:
            if test_context:
                await test_context.close()
            if test_browser:
                await test_browser.close()
            if test_playwright:
                await test_playwright.stop()
        except Exception:
            pass


async def run_auto_cookie_check_once(uid: int) -> None:
    """上线后首次进入首页时触发一次，与手动检查一致的无头检测。"""
    if state.interactive_login_user_id is not None:
        state.auto_cookie_status = "skipped"
        state.auto_cookie_message = "有会话正在通过浏览器连接邮箱，已跳过自动 Cookie 检查。"
        state.auto_cookie_checked_at = datetime.now().isoformat(timespec="seconds")
        return

    cfg = auth_db.load_user_config(uid)
    state.config = cfg
    email_cfg = cfg.get("email", {})

    if not load_cookies_for_check(email_cfg):
        state.auto_cookie_status = "skipped"
        state.auto_cookie_message = "未配置 Cookie，已跳过上线自动检查。"
        state.auto_cookie_checked_at = datetime.now().isoformat(timespec="seconds")
        return

    # 无头检测使用独立 Playwright 实例，勿关闭全局可见浏览器（否则会破坏预启动）

    state.auto_cookie_status = "checking"
    state.auto_cookie_message = "正在自动检查 Cookie…"
    try:
        result = await cookie_check_headless(email_cfg)
        state.auto_cookie_status = result["status"]
        state.auto_cookie_message = result["message"]
    except Exception as e:
        logger.exception("run_auto_cookie_check_once")
        state.auto_cookie_status = "error"
        state.auto_cookie_message = str(e)[:200]
    finally:
        state.auto_cookie_checked_at = datetime.now().isoformat(timespec="seconds")


# ====== 改进二：带重试的浏览器初始化 ======
async def ensure_browser(max_retries: int = 2):
    """Ensure browser is running and page is ready, with automatic retry."""
    if state.page and not state.page.is_closed():
        return  # 浏览器已就绪

    last_error = None
    for attempt in range(1, max_retries + 1):
        logger.info(
            f"Initializing browser... (attempt {attempt}/{max_retries})")
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

            async def _on_visible_page_close(_p) -> None:
                asyncio.create_task(cleanup_on_visible_mail_page_close())

            state.page.on("close", _on_visible_page_close)
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
    raise HTTPException(
        status_code=500, detail=f"Failed to launch browser after {max_retries} attempts: {repr(last_error)}")


# ====== 改进二：进度状态 API ======
@app.get("/api/status")
async def get_status(request: Request):
    """返回后端当前处理阶段，供前端轮询。"""
    require_user_id(request)
    return {
        "stage": state.current_stage,
        "detail": state.stage_detail,
        "extracted_count": state.extracted_count,
        "total_count": state.total_count,
        "auto_cookie_status": state.auto_cookie_status,
        "auto_cookie_message": state.auto_cookie_message,
        "auto_cookie_checked_at": state.auto_cookie_checked_at,
    }

# ====== 浏览器预启动开关 API ======


class PrelaunchRequest(BaseModel):
    enabled: bool


@app.get("/api/prelaunch")
async def get_prelaunch(request: Request):
    """获取浏览器预启动开关状态。"""
    uid = require_user_id(request)
    cfg = auth_db.load_user_config(uid)
    enabled = cfg.get("browser", {}).get("prelaunch", False)
    return {"enabled": enabled}


@app.post("/api/prelaunch")
async def set_prelaunch(request: Request, req: PrelaunchRequest):
    """切换浏览器预启动开关并写入当前用户配置。"""
    uid = require_user_id(request)
    current_config = auth_db.load_user_config(uid)
    if "browser" not in current_config:
        current_config["browser"] = {}
    current_config["browser"]["prelaunch"] = req.enabled
    try:
        auth_db.save_user_config(uid, current_config)
        if _session_user_id(request) == uid:
            state.config = current_config
        if req.enabled:
            asyncio.create_task(schedule_prelaunch_for_user(uid))
        else:

            async def _close_prelaunch_if_match() -> None:
                async with prelaunch_lock:
                    if state.prelaunch_user_id == uid:
                        await close_browser_resources()

            asyncio.create_task(_close_prelaunch_if_match())
        status_label = "开启" if req.enabled else "关闭"
        extra = "若已配置邮箱 Cookie，将在后台打开浏览器并登录邮箱。" if req.enabled else ""
        return {
            "status": "success",
            "message": (f"浏览器预启动已{status_label}。" + extra).strip(),
        }
    except Exception as e:
        logger.error(f"Failed to save prelaunch config: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": f"保存配置失败: {e}"})


@app.get("/api/config")
async def get_config(request: Request):
    """获取当前登录用户的配置供前端显示"""
    uid = require_user_id(request)
    current_config = auth_db.load_user_config(uid)

    cookies_list = current_config.get("email", {}).get("cookies", [])
    cookies_str = json.dumps(cookies_list, indent=2,
                             ensure_ascii=False) if cookies_list else ""

    return {
        "ai_base_url": current_config.get("ai", {}).get("base_url", ""),
        "ai_api_key": current_config.get("ai", {}).get("api_key", ""),
        "ai_model": current_config.get("ai", {}).get("model", "gpt-4o-mini"),
        "email_url": current_config.get("email", {}).get("url", "https://mail.xjtlu.edu.cn/owa"),
        "email_cookies": cookies_str,
    }


@app.post("/api/config")
async def update_config(request: Request, req: ConfigUpdateRequest):
    """保存当前登录用户的配置"""
    uid = require_user_id(request)
    current_config = auth_db.load_user_config(uid)

    cookies_list, cookie_err = parse_email_cookies_blob(req.email_cookies)
    if cookie_err:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": cookie_err},
        )

    if "ai" not in current_config:
        current_config["ai"] = {}
    current_config["ai"]["base_url"] = req.ai_base_url.strip()
    current_config["ai"]["api_key"] = req.ai_api_key.strip()
    current_config["ai"]["model"] = req.ai_model.strip()

    if "email" not in current_config:
        current_config["email"] = {}
    current_config["email"]["url"] = req.email_url.strip()
    current_config["email"]["login_type"] = "cookie"
    current_config["email"]["cookies"] = cookies_list

    try:
        auth_db.save_user_config(uid, current_config)
        state.config = current_config

        try:
            if state.context:
                await state.context.close()
            if state.browser:
                await state.browser.close()
            if state.playwright:
                await state.playwright.stop()
        except Exception as cleanup_err:
            logger.warning(
                "Browser cleanup after config save: %s", cleanup_err)
        finally:
            state.context = None
            state.browser = None
            state.playwright = None
            state.page = None
            state.prelaunch_user_id = None
            state.interactive_login_user_id = None

        return {"status": "success", "message": "配置已保存。"}
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": f"保存配置时出错: {e}"})


@app.get("/api/digest/settings")
async def get_digest_settings(request: Request):
    uid = require_user_id(request)
    cfg = auth_db.load_user_config(uid)
    digest = merge_digest_defaults(cfg.get("digest"))
    return {
        "status": "success",
        "digest": digest,
        "next_run_hint": compute_next_run_hint(digest),
        "server_note": "当前账号的定时摘要设置已读取与保存。",
    }


@app.put("/api/digest/settings")
async def put_digest_settings(request: Request, body: DigestSettingsRequest):
    uid = require_user_id(request)
    cfg = auth_db.load_user_config(uid)
    digest = merge_digest_defaults(cfg.get("digest"))
    digest.update(
        {
            "enabled": bool(body.enabled),
            "cadence": (body.cadence or "daily").strip().lower(),
            "local_time": (body.local_time or "08:00").strip() or "08:00",
            "timezone": (body.timezone or "").strip(),
            "weekday": max(0, min(6, int(body.weekday))),
            "keyword": (body.keyword or "").strip(),
            "instruction": (body.instruction or "").strip()
            or "请用自然语气总结这些邮件，重点关注活动、课程作业和重要事项。",
            "mode": "daily",
            "email_count": max(1, min(10, int(body.email_count))),
        }
    )
    cfg["digest"] = digest
    auth_db.save_user_config(uid, cfg)
    return {
        "status": "success",
        "digest": digest,
        "next_run_hint": compute_next_run_hint(digest),
        "message": "定时摘要设置已保存。",
    }


@app.get("/api/digest/history")
async def get_digest_history(request: Request, limit: int = 25):
    uid = require_user_id(request)
    items = auth_db.digest_list_artifacts(uid, limit=limit)
    return {"status": "success", "items": items}


@app.post("/api/check_cookie")
async def check_cookie(request: Request):
    """检查 Cookie 状态并更新 config"""
    logger.info("Checking cookie status...")
    uid = require_user_id(request)

    try:
        new_config = auth_db.load_user_config(uid)
        state.config = new_config
        logger.info("Config reloaded for cookie check (user %s).", uid)
    except Exception as e:
        logger.error(f"Failed to reload config: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": f"加载配置失败: {e}"})

    # cookie_check_headless 使用独立无头实例，不应先关闭预启动/任务用的可见浏览器

    email_config = state.config.get("email", {})
    result = await cookie_check_headless(email_config)
    st = result.get("status")
    if st in ("valid", "warning"):
        state.auto_cookie_status = st
        state.auto_cookie_message = result.get("message", "")
    elif st == "invalid":
        state.auto_cookie_status = "invalid"
        state.auto_cookie_message = result.get("message", "")
    elif st == "error":
        state.auto_cookie_status = "error"
        state.auto_cookie_message = result.get("message", "")
    else:
        state.auto_cookie_status = st if isinstance(st, str) else "error"
        state.auto_cookie_message = result.get("message", "")
    state.auto_cookie_checked_at = datetime.now().isoformat(timespec="seconds")
    return result


@app.post("/api/mail/interactive_login/start")
async def interactive_mail_login_start(request: Request):
    """打开未注入 Cookie 的可见浏览器，供用户在 OWA 中手动登录。"""
    uid = require_user_id(request)
    async with prelaunch_lock:
        await close_browser_resources()
        cfg = auth_db.load_user_config(uid)
        email_cfg = cfg.get("email", {}) or {}
        mail_url = (email_cfg.get("url")
                    or "https://mail.xjtlu.edu.cn/owa").strip()
        if not mail_url:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "请先在高级设置中填写邮箱 URL。"},
            )

        pw = browser = context = page = None
        try:
            from playwright.async_api import async_playwright

            pw = await async_playwright().start()
            browser = await pw.chromium.launch(
                headless=False, channel="msedge", args=["--start-maximized"]
            )
            context = await browser.new_context()
            page = await context.new_page()
            await page.goto(mail_url, wait_until="domcontentloaded", timeout=90000)
            await page.wait_for_timeout(600)
        except Exception as e:
            logger.exception("interactive_mail_login_start")
            try:
                if context:
                    await context.close()
            except Exception:
                pass
            try:
                if browser:
                    await browser.close()
            except Exception:
                pass
            try:
                if pw:
                    await pw.stop()
            except Exception:
                pass
            return JSONResponse(
                status_code=500,
                content={"status": "error",
                         "message": f"无法打开浏览器: {str(e)[:120]}"},
            )

        state.playwright = pw
        state.browser = browser
        state.context = context
        state.page = page
        state.config = cfg
        state.interactive_login_user_id = uid

        # 绑定页面关闭事件监听器
        async def handle_page_close(p):
            # 将清理任务丢给事件循环
            asyncio.create_task(cleanup_on_page_close(uid))

        page.on("close", handle_page_close)

    return {
        "status": "success",
        "message": "请在打开的窗口中登录邮箱，完成后点击「我已登录」。",
    }


@app.post("/api/mail/interactive_login/complete")
async def interactive_mail_login_complete(request: Request):
    """读取当前浏览器上下文的 Cookie 并写入当前用户的 email.cookies。"""
    uid = require_user_id(request)
    async with prelaunch_lock:
        if state.interactive_login_user_id != uid:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "请先在当前账号下点击「打开邮箱登录窗口」。",
                },
            )
        if not state.context or not state.page or state.page.is_closed():
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "浏览器会话已失效，请重新开始。"},
            )

        outcome = await probe_mail_session_on_page(state.page)
        if outcome["status"] in ("invalid", "error"):
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": outcome.get("message")
                    or ("仍在登录页，请完成登录后再试。" if outcome["status"] == "invalid" else "页面检测失败。"),
                },
            )

        try:
            raw = await state.context.cookies()
        except Exception as e:
            logger.exception("interactive_mail_login_complete cookies")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": str(e)[:120]},
            )

        normalized = _playwright_cookies_to_config_list(raw)
        if not normalized:
            return JSONResponse(
                status_code=400,
                content={"status": "error",
                         "message": "未能读取到任何 Cookie，请确认已登录成功。"},
            )

        cfg = auth_db.load_user_config(uid)
        if "email" not in cfg:
            cfg["email"] = {}
        em = cfg["email"]
        em["cookies"] = normalized
        em["login_type"] = "cookie"
        if not (em.get("url") or "").strip():
            em["url"] = "https://mail.xjtlu.edu.cn/owa"
        auth_db.save_user_config(uid, cfg)
        state.config = cfg

        warn = outcome["status"] == "warning"
        state.auto_cookie_status = "warning" if warn else "valid"
        state.auto_cookie_message = "已通过交互式登录保存会话。"

        await close_browser_resources()

    msg = "已保存邮箱会话。"
    if warn and outcome.get("message"):
        msg += " " + str(outcome["message"])
    return {
        "status": "success",
        "message": msg,
        "cookie_count": len(normalized),
        "warning": warn,
    }


@app.post("/api/mail/interactive_login/cancel")
async def interactive_mail_login_cancel(request: Request):
    uid = require_user_id(request)
    async with prelaunch_lock:
        if state.interactive_login_user_id != uid:
            return {"status": "success", "message": "没有进行中的交互登录。"}
        await close_browser_resources()
    return {"status": "success", "message": "已取消。"}


@app.get("/api/mail/interactive_login/status")
async def interactive_mail_login_status(request: Request):
    uid = require_user_id(request)
    active = (
        state.interactive_login_user_id == uid
        and state.page is not None
        and not state.page.is_closed()
    )
    return {
        "active": active,
        "interactive_user_id": state.interactive_login_user_id,
    }


def _parse_iso_date_boundary(s: Optional[str], *, end_of_day: bool = False) -> Optional[datetime]:
    if not s or not str(s).strip():
        return None
    try:
        d = datetime.strptime(str(s).strip()[:10], "%Y-%m-%d")
        if end_of_day:
            return d.replace(hour=23, minute=59, second=59)
        return d.replace(hour=0, minute=0, second=0, microsecond=0)
    except ValueError:
        return None


@app.post("/api/deep_scan")
async def deep_scan(http_request: Request, payload: DeepScanRequest):
    """
    深度扫描：与开发者测试同款拉列表 + 抽完整正文，正则分类，落盘 deep_scan_result.json，
    并缓存 state.deep_scan_export；不调 LLM。后续深度分析仅读本地 JSON，不碰浏览器。
    """
    uid = require_user_id(http_request)
    if state.interactive_login_user_id is not None:
        return JSONResponse(
            status_code=409,
            content={
                "status": "error",
                "message": "正在通过浏览器连接邮箱，请先完成「我已登录」或「取消」后再试。",
            },
        )
    if state.auto_cookie_status not in ("valid", "warning"):
        return JSONResponse(
            status_code=403,
            content={
                "status": "error",
                "message": "请先在步骤一中连接校园邮箱并确保验证成功。",
            },
        )

    kw = (payload.keyword or "").strip()
    export_stem = ""
    snapshot_saved = False

    async with execute_lock:
        state.config = auth_db.load_user_config(uid)
        try:
            await ensure_browser()
        except HTTPException:
            raise
        except Exception as e:
            state.current_stage = "error"
            state.stage_detail = str(e)
            raise

        state.current_stage = "searching"
        state.stage_detail = (
            f"〔深度扫描〕正在拉取并提取正文（最多 {DEEP_MAX_EMAILS} 封，与开发者测试同款）…"
        )
        try:
            deep_export_doc, emails = await dev_style_deep_extract_to_export(
                state.page,
                kw,
                state.config,
                max_list_emails=DEEP_MAX_EMAILS,
                ui_label="〔深度扫描〕",
                write_debug_artifacts=False,
            )
        except RuntimeError as e:
            state.current_stage = "idle"
            state.stage_detail = ""
            return JSONResponse(
                status_code=200,
                content={"status": "cookie_expired",
                         "message": str(e), "emails": []},
            )
        except Exception as e:
            logger.exception("deep_scan")
            state.current_stage = "error"
            state.stage_detail = str(e)
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": str(e), "emails": []},
            )

        for s in deep_export_doc["samples"]:
            preview = (s.get("body") or "")[:400]
            s["category"] = classify_email(
                s.get("sender") or "",
                s.get("subject") or "",
                preview,
            )

        export_to_save = {
            "format": "deep_scan_export",
            "version": 1,
            "exported_at": deep_export_doc.get("exported_at")
            or datetime.now().isoformat(timespec="seconds"),
            "keyword": deep_export_doc["keyword"],
            "list_count": deep_export_doc["list_count"],
            "indices": deep_export_doc.get("indices"),
            "samples": deep_export_doc["samples"],
        }
        export_stem = _deep_scan_export_stem_from_iso(
            str(export_to_save.get("exported_at") or ""))
        snapshot_saved = False
        try:
            _save_deep_scan_result_json(export_to_save)
            DEEP_SCAN_EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
            export_path = DEEP_SCAN_EXPORTS_DIR / f"{export_stem}.json"
            export_path.write_text(
                json.dumps(export_to_save, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            snapshot_saved = True
        except Exception:
            logger.exception("写入深度扫描快照失败")

        state.deep_scan_export = export_to_save
        state.email_results = emails
        state.deep_scan_keyword = kw
        state.current_stage = "idle"
        state.stage_detail = ""

    samples = (state.deep_scan_export or {}).get("samples") or []
    out_rows = []
    cat_counts = {c: 0 for c in EMAIL_CATEGORY_LABELS}
    parsed_dates: list[datetime] = []

    for s in samples:
        sender = s.get("sender") or ""
        subj = s.get("subject") or ""
        preview = (s.get("body") or "")[:400]
        cat = s.get("category") or classify_email(sender, subj, preview)
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        pd = parse_email_date_for_filter(s.get("date") or "")
        if pd:
            parsed_dates.append(pd)
        cv = s.get("convid")
        convid_str = (cv if isinstance(cv, str) else "")[:80]
        out_rows.append(
            {
                "index": int(s.get("index") or 0),
                "subject": subj,
                "date": s.get("date") or "",
                "sender": sender,
                "preview": preview,
                "category": cat,
                "convid": convid_str,
            }
        )

    date_min = ""
    date_max = ""
    if parsed_dates:
        mn = min(parsed_dates)
        mx = max(parsed_dates)
        date_min = mn.strftime("%Y-%m-%d")
        date_max = mx.strftime("%Y-%m-%d")

    n = len(samples)
    return {
        "status": "success",
        "message": f"已扫描 {n} 封邮件（列表顺序与收件箱一致，正文已缓存至本地）。",
        "list_count": n,
        "keyword": kw,
        "emails": out_rows,
        "categories": cat_counts,
        "date_range": {"min": date_min, "max": date_max},
        "export_stem": export_stem,
        "snapshot_saved": snapshot_saved,
    }


@app.post("/api/deep_scan/save_chart")
async def deep_scan_save_chart(request: Request, body: DeepScanChartSaveRequest):
    """将前端 ECharts 饼图导出的 PNG（data URL）写入 deep_scan_exports/{stem}.png"""
    require_user_id(request)
    stem = (body.export_stem or "").strip()
    if not stem or ".." in stem or "/" in stem or "\\" in stem:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "无效的 export_stem"},
        )
    raw = body.png_base64 or ""
    if "," in raw:
        raw = raw.split(",", 1)[1]
    try:
        data = base64.b64decode(raw)
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "无法解析图片数据"},
        )
    try:
        DEEP_SCAN_EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DEEP_SCAN_EXPORTS_DIR / f"{stem}.png"
        out_path.write_bytes(data)
    except Exception as exc:
        logger.exception("deep_scan_save_chart")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(exc)[:120]},
        )
    return {"status": "success", "filename": f"{stem}.png"}


@app.post("/api/execute")
async def execute_task(http_request: Request, payload: ExecuteRequest):
    uid = require_user_id(http_request)
    if state.interactive_login_user_id is not None:
        return JSONResponse(
            status_code=409,
            content={
                "status": "error",
                "message": "正在通过浏览器连接邮箱，请先完成「我已登录」保存会话，或点击「取消」后再开始分析。",
            },
        )

    if state.auto_cookie_status not in ("valid", "warning"):
        return JSONResponse(
            status_code=403,
            content={
                "status": "error",
                "message": "请先在“步骤一”中连接您的校园邮箱并确保验证成功，然后再执行 AI 总结。",
            },
        )

    async with execute_lock:
        state.config = auth_db.load_user_config(uid)

        keyword = (payload.keyword or "").strip()
        instruction = payload.instruction.strip()
        if not instruction:
            instruction = "请用自然语气总结这些邮件，重点关注活动、课程作业和重要事项。"

        use_indices = bool(payload.indices and len(payload.indices) > 0)

        if use_indices and payload.mode != "deep":
            state.current_stage = "idle"
            state.stage_detail = ""
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "按序号分析仅适用于深度模式，请先选择「深度模式」或清空选中序号。",
                },
            )

        export_doc: Optional[dict] = None
        if payload.mode == "deep":
            export_doc = state.deep_scan_export
            if export_doc is None:
                export_doc = _load_deep_scan_export_from_disk()
                if export_doc:
                    state.deep_scan_export = export_doc
            if export_doc is None or not export_doc.get("samples"):
                state.current_stage = "idle"
                state.stage_detail = ""
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "message": "没有可用的深度扫描结果。请先点击「深度扫描」拉取列表后再分析。",
                    },
                )
            state.deep_scan_keyword = (export_doc.get("keyword") or "").strip()
            if keyword != state.deep_scan_keyword:
                state.current_stage = "idle"
                state.stage_detail = ""
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "message": "搜索关键词与上次深度扫描不一致，请重新深度扫描或恢复相同关键词。",
                    },
                )
        else:
            try:
                await ensure_browser()
            except HTTPException:
                raise
            except Exception as e:
                state.current_stage = "error"
                state.stage_detail = str(e)
                raise

        print(
            f"Executing task: search='{keyword or '(最新邮件)'}', instruction='{instruction}', use_indices={use_indices}")

        try:
            limit_ceiling = (
                DEEP_MAX_EMAILS if payload.mode == "deep" else DAILY_MAX_EMAILS
            )
            safe_count = max(1, min(payload.email_count, limit_ceiling))

            extracted_items: list = []
            failed_count = 0
            sanitized_emails: list = []

            if payload.mode == "deep":
                # 从内存或 deep_scan_result.json 读取正文，不访问浏览器、不再调用 dev_style_deep_extract
                assert export_doc is not None
                d_lo = _parse_iso_date_boundary(
                    payload.date_from, end_of_day=False)
                d_hi = _parse_iso_date_boundary(
                    payload.date_to, end_of_day=True)
                idx_order: dict = {}
                idx_set: Optional[set] = None
                n_list = len(export_doc["samples"])

                if use_indices:
                    idxs_sorted = sorted(
                        {i for i in (payload.indices or [])
                         if isinstance(i, int)}
                    )
                    if not idxs_sorted:
                        state.current_stage = "idle"
                        state.stage_detail = ""
                        return JSONResponse(
                            status_code=400,
                            content={
                                "status": "error",
                                "message": "所选序号无效，请重新深度扫描后再选。",
                            },
                        )
                    for idx in idxs_sorted:
                        if idx < 1 or idx > n_list:
                            state.current_stage = "idle"
                            state.stage_detail = ""
                            return JSONResponse(
                                status_code=400,
                                content={
                                    "status": "error",
                                    "message": "所选序号超出上次深度扫描列表范围，请重新深度扫描后再选。",
                                },
                            )
                    if max(idxs_sorted) > DEEP_MAX_EMAILS:
                        state.current_stage = "idle"
                        state.stage_detail = ""
                        return JSONResponse(
                            status_code=400,
                            content={
                                "status": "error",
                                "message": (
                                    f"深度模式仅支持前 {DEEP_MAX_EMAILS} 封。请改选前 "
                                    f"{DEEP_MAX_EMAILS} 封邮件后再试。"
                                ),
                            },
                        )
                    if len(idxs_sorted) > limit_ceiling:
                        state.current_stage = "idle"
                        state.stage_detail = ""
                        return JSONResponse(
                            status_code=400,
                            content={
                                "status": "error",
                                "message": f"一次最多分析 {limit_ceiling} 封邮件，请缩小筛选范围。",
                            },
                        )
                    idx_set = set(idxs_sorted)
                    idx_order = {idx: pos for pos,
                                 idx in enumerate(idxs_sorted)}

                state.stage_detail = (
                    f"〔深度分析〕从本地缓存读取正文（共 {n_list} 封，不访问浏览器）…"
                )

                if n_list == 0:
                    state.current_stage = "idle"
                    state.stage_detail = ""
                    return JSONResponse(
                        status_code=400,
                        content={
                            "status": "error",
                            "message": "当前未找到任何邮件，无法分析。",
                        },
                    )

                samples = [dict(s) for s in export_doc["samples"]]
                for s in samples:
                    if "category" not in s:
                        preview = (s.get("body") or "")[:400]
                        s["category"] = classify_email(
                            s.get("sender") or "",
                            s.get("subject") or "",
                            preview,
                        )

                if not use_indices:
                    samples = samples[:safe_count]
                if use_indices and idx_set is not None:
                    samples = [s for s in samples if s["index"] in idx_set]
                    found = {s["index"] for s in samples}
                    missing = sorted(idx_set - found)
                    if missing:
                        state.current_stage = "idle"
                        state.stage_detail = ""
                        return JSONResponse(
                            status_code=400,
                            content={
                                "status": "error",
                                "message": (
                                    f"本次拉取结果中未包含序号 {missing}，请重新深度扫描后再选。"
                                ),
                            },
                        )
                    samples.sort(key=lambda x: idx_order[x["index"]])

                if d_lo or d_hi:
                    kept = []
                    for s in samples:
                        ed = parse_email_date_for_filter(s.get("date") or "")
                        if ed is None:
                            kept.append(s)
                            continue
                        if d_lo and ed < d_lo:
                            continue
                        if d_hi and ed > d_hi:
                            continue
                        kept.append(s)
                    samples = kept

                if len(samples) > limit_ceiling:
                    state.current_stage = "idle"
                    state.stage_detail = ""
                    return JSONResponse(
                        status_code=400,
                        content={
                            "status": "error",
                            "message": f"一次最多分析 {limit_ceiling} 封邮件，当前筛选结果 {len(samples)} 封，请缩小分类或日期范围。",
                        },
                    )

                state.total_count = len(samples)
                state.extracted_count = len(samples)
                for i, s in enumerate(samples):
                    if not s.get("ok"):
                        failed_count += 1
                    extracted_items.append(
                        {
                            "index": i + 1,
                            "subject": s.get("subject") or "",
                            "date": s.get("date") or "",
                            "sender": s.get("sender") or "",
                            "body": s.get("body") or "",
                        }
                    )
                    sanitized_emails.append(
                        {
                            "id": i,
                            "subject": s.get("subject") or "",
                            "date": s.get("date") or "",
                            "href": s.get("href") or "",
                        }
                    )

            else:
                # 日常模式
                state.current_stage = "searching"
                state.stage_detail = (
                    f"正在搜索关键词: {keyword}" if keyword else "正在获取最新邮件..."
                )
                print(f"Searching for: {keyword}")
                try:
                    emails = await search_emails(
                        state.page,
                        keyword,
                        config=state.config,
                        max_emails=safe_count,
                        mode="daily",
                    )
                except RuntimeError as e:
                    state.current_stage = "error"
                    state.stage_detail = str(e)
                    return JSONResponse(
                        status_code=200,
                        content={
                            "status": "cookie_expired",
                            "message": str(e),
                        },
                    )
                state.email_results = emails

                state.current_stage = "extracting"
                state.total_count = len(emails)
                state.extracted_count = 0
                state.stage_detail = f"正在提取 {len(emails)} 封邮件正文..."
                print(f"Extracting bodies for {len(emails)} emails...")

                for i, email in enumerate(emails):
                    state.extracted_count = i + 1
                    state.stage_detail = f"正在提取第 {i+1}/{len(emails)} 封邮件..."
                    print(f"Extracting body for email {i+1}...")
                    loc = email.get("locator")
                    try:
                        body, pane_disp, pane_sort = await extract_full_body(
                            state.page,
                            loc,
                            expected_subject=email.get("subject", ""),
                            fast_activation=False,
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to extract email {i+1} body: {repr(e)}")
                        body = f"[正文提取失败: {str(e)[:100]}]"
                        pane_disp, pane_sort = "", ""
                        failed_count += 1
                    merged_disp, merged_sort = merge_list_and_pane_datetime(
                        str(email.get("date", "") or ""),
                        str(email.get("date_display", "") or ""),
                        pane_sort,
                        pane_disp,
                    )
                    final_sort = merged_sort or email.get("date", "")
                    final_disp = merged_disp or merged_sort or email.get(
                        "date", "")
                    extracted_items.append(
                        {
                            "index": i + 1,
                            "subject": email.get("subject", ""),
                            "date": final_sort,
                            "date_display": final_disp,
                            "sender": email.get("sender", ""),
                            "body": body,
                        }
                    )
                    sanitized_emails.append(
                        {
                            "id": i,
                            "subject": email.get("subject", ""),
                            "date": final_sort,
                            "date_display": final_disp,
                            "href": email.get("href", ""),
                        }
                    )
                    await asyncio.sleep(0.2)

            if failed_count > 0:
                logger.info(
                    f"{failed_count} email(s) failed body extraction, continuing with available data."
                )

            # 3. Summarize — 每封邮件并行 1 次 LLM，再 1 次汇总
            state.current_stage = "summarizing"
            from datetime import datetime

            loop = asyncio.get_running_loop()

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
            total_batches = (n + LLM_PARALLEL_BATCH_SIZE -
                             1) // LLM_PARALLEL_BATCH_SIZE
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
                    sender=str(item.get("sender", "") or ""),
                    date_display=str(item.get("date_display", "") or ""),
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
                batch_prompts = prompts[start: start + LLM_PARALLEL_BATCH_SIZE]
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
                normalized.extend(normalize_parallel_llm_result(r)
                                  for r in parallel_out)
            per_email_sections = "\n\n".join(
                f"—— 第 {i + 1} 封邮件的初步分析 ——\n{t}" for i, t in enumerate(normalized)
            )
            final_prompt = build_final_merge_prompt(
                today=today,
                instruction=instruction,
                per_email_sections=per_email_sections,
                email_count=n,
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


@app.post("/api/dev/extract_sample_bodies")
async def dev_extract_sample_bodies(http_request: Request, payload: DevExtractSampleBodiesRequest):
    """
    开发者：深度模式拉取列表（最多 DEEP_MAX_EMAILS 封），对全部条目执行 extract_full_body，
    不调 LLM；成功后将结果写入 src 下 JSON（便于脚本/后续处理）。
    """
    uid = require_user_id(http_request)
    if state.interactive_login_user_id is not None:
        return JSONResponse(
            status_code=409,
            content={
                "status": "error",
                "message": "正在通过浏览器连接邮箱，请先完成「我已登录」或「取消」后再试。",
            },
        )
    if state.auto_cookie_status not in ("valid", "warning"):
        return JSONResponse(
            status_code=403,
            content={
                "status": "error",
                "message": "请先在步骤一中连接校园邮箱并确保验证成功。",
            },
        )

    kw = (payload.keyword or "").strip()

    async with execute_lock:
        state.config = auth_db.load_user_config(uid)
        try:
            await ensure_browser()
        except HTTPException:
            raise
        except Exception as e:
            state.current_stage = "error"
            state.stage_detail = str(e)
            raise

        state.current_stage = "searching"
        state.stage_detail = (
            f"〔开发者〕深度拉取列表（最多 {DEEP_MAX_EMAILS} 封，较快节奏）…"
        )
        try:
            export_doc, emails = await dev_style_deep_extract_to_export(
                state.page,
                kw,
                state.config,
                max_list_emails=DEEP_MAX_EMAILS,
                ui_label="〔开发者〕",
                write_debug_artifacts=True,
            )
        except RuntimeError as e:
            state.current_stage = "idle"
            state.stage_detail = ""
            return JSONResponse(
                status_code=200,
                content={"status": "cookie_expired",
                         "message": str(e), "samples": []},
            )
        except Exception as e:
            logger.exception("dev_extract_sample_bodies")
            state.current_stage = "error"
            state.stage_detail = str(e)
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": str(e), "samples": []},
            )

        n = export_doc["list_count"]
        samples = export_doc["samples"]
        extract_indices = export_doc["indices"]
        exported_at = export_doc["exported_at"]
        if n == 0:
            state.current_stage = "idle"
            state.stage_detail = ""
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "当前未找到任何邮件，无法提取。",
                    "list_count": n,
                    "samples": [],
                },
            )

        state.current_stage = "idle"
        state.stage_detail = ""
        state.email_results = emails

        out_fname = "dev_extract_bodies_ALL.json"
        out_path = Path(__file__).resolve().parent / out_fname
        out_path.write_text(
            json.dumps(export_doc, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("dev_extract_sample_bodies wrote %s", out_path)

        return {
            "status": "success",
            "message": f"已成功提取全量 {n} 封正文（未调用 LLM），已写入 JSON。",
            "list_count": n,
            "keyword": kw,
            "indices": list(extract_indices),
            "samples": samples,
            "exported_at": exported_at,
            "export": export_doc,
            "output_file": out_fname,
        }


@app.post("/api/dev/extract_daily_bodies_no_llm")
async def dev_extract_daily_bodies_no_llm(
    http_request: Request, payload: DevExtractDailyBodiesRequest
):
    """
    开发者测试2：与日常模式一致 — search_emails(mode=daily) 后按序 extract_full_body，
    不调 LLM；结果仅通过接口返回并在前端展示。
    """
    uid = require_user_id(http_request)
    if state.interactive_login_user_id is not None:
        return JSONResponse(
            status_code=409,
            content={
                "status": "error",
                "message": "正在通过浏览器连接邮箱，请先完成「我已登录」或「取消」后再试。",
            },
        )
    if state.auto_cookie_status not in ("valid", "warning"):
        return JSONResponse(
            status_code=403,
            content={
                "status": "error",
                "message": "请先在步骤一中连接校园邮箱并确保验证成功。",
            },
        )

    kw = (payload.keyword or "").strip()
    n_req = max(1, min(int(payload.email_count), DAILY_MAX_EMAILS))

    async with execute_lock:
        state.config = auth_db.load_user_config(uid)
        try:
            await ensure_browser()
        except HTTPException:
            raise
        except Exception as e:
            state.current_stage = "error"
            state.stage_detail = str(e)
            raise

        state.current_stage = "searching"
        state.stage_detail = f"〔开发者·日常流〕正在搜索/拉取列表（最多 {n_req} 封）…"
        try:
            emails = await search_emails(
                state.page,
                kw,
                config=state.config,
                max_emails=n_req,
                mode="daily",
            )
        except RuntimeError as e:
            state.current_stage = "idle"
            state.stage_detail = ""
            return JSONResponse(
                status_code=200,
                content={
                    "status": "cookie_expired",
                    "message": str(e),
                    "items": [],
                    "text": "",
                },
            )
        except Exception as e:
            logger.exception("dev_extract_daily_bodies_no_llm search")
            state.current_stage = "error"
            state.stage_detail = str(e)
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": str(
                    e), "items": [], "text": ""},
            )

        state.email_results = emails
        state.current_stage = "extracting"
        state.total_count = len(emails)
        state.extracted_count = 0

        items = []
        text_blocks = []
        for i, email in enumerate(emails):
            state.extracted_count = i + 1
            state.stage_detail = f"〔开发者·日常流〕正在提取第 {i + 1}/{len(emails)} 封正文…"
            subj = (email.get("subject") or "").strip()
            dt = (email.get("date") or "").strip()
            d_disp = (email.get("date_display") or "").strip()
            sender = (email.get("sender") or "").strip()
            body = ""
            err = None
            pane_disp, pane_sort = "", ""
            try:
                body, pane_disp, pane_sort = await extract_full_body(
                    state.page,
                    email.get("locator"),
                    expected_subject=subj,
                )
            except Exception as exc:
                logger.warning(
                    "dev_extract_daily_bodies_no_llm %s: %s", i + 1, repr(exc))
                err = str(exc)[:200]
                body = f"[正文提取失败: {err}]"

            merged_disp, merged_sort = merge_list_and_pane_datetime(
                dt, d_disp, pane_sort, pane_disp
            )
            final_date = merged_sort or dt
            final_disp = merged_disp or merged_sort or dt
            ok = err is None and bool(body) and not str(
                body).startswith("[正文提取失败")
            items.append(
                {
                    "index": i + 1,
                    "subject": subj,
                    "date": final_date,
                    "date_display": final_disp,
                    "sender": sender,
                    "body": body,
                    "body_chars": len(body or ""),
                    "ok": ok,
                }
            )
            text_blocks.append(
                f"========== 第 {i + 1} 封 ==========\n"
                f"主题: {subj}\n发件人: {sender or '(无)'}\n日期: {final_disp or final_date or '(无)'}\n\n{body}\n"
            )
            await asyncio.sleep(0.2)

        state.current_stage = "idle"
        state.stage_detail = ""

        header = (
            f"# dev_extract_daily_bodies_no_llm {datetime.now().isoformat(timespec='seconds')}\n"
            f"# 与日常模式相同流程（daily），未调用 LLM\n"
            f"# keyword: {kw or '(empty)'}\n"
            f"# requested_count: {n_req}  list_count: {len(emails)}\n\n"
        )
        full_text = header + "\n".join(text_blocks)

        return {
            "status": "success",
            "message": f"日常流程已提取 {len(emails)} 封正文（未调用 LLM）。",
            "keyword": kw,
            "requested_count": n_req,
            "list_count": len(emails),
            "items": items,
            "text": full_text,
        }


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    uid = _session_user_id(request)
    if uid is None:
        return RedirectResponse("/login", status_code=302)
    cfg = auth_db.load_user_config(uid)
    mail_url = cfg.get("email", {}).get("url", "https://mail.xjtlu.edu.cn/owa")
    u = auth_db.get_user_by_id(uid)
    asyncio.create_task(schedule_prelaunch_for_user(uid))
    async with auto_cookie_lock:
        if not state._auto_cookie_check_ran:
            state._auto_cookie_check_ran = True
            state.auto_cookie_status = "suggest_check"
            state.auto_cookie_message = "请先验证邮箱可访问再查询哦"
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "mail_url": mail_url,
            "current_username": u["username"] if u else "",
            "current_email": u["email"] if u else "",
        },
    )


@app.post("/api/search")
async def search(request: SearchRequest):
    return {"status": "deprecated", "message": "Use /api/execute instead"}


@app.post("/api/summarize")
async def summarize(request: Request):  # Changed to Request to avoid undefined model
    return {"status": "deprecated", "message": "Use /api/execute instead"}

if __name__ == "__main__":
    # Disable reload to avoid Windows subprocess loop issues
    # Use port 8001 to avoid conflicts with hung processes
    uvicorn.run("app:app", host="127.0.0.1", port=8001,
                reload=False, loop="asyncio")
