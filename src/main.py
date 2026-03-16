# requirements.txt 依赖由 pip install -r requirements.txt 安装
from pathlib import Path
import asyncio
import json
import os

import requests
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from datetime import datetime


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        config_path.write_text("{}", encoding="utf-8")
        config = {}
    else:
        try:
            content = config_path.read_text(encoding="utf-8").strip()
            if not content:
                config_path.write_text("{}", encoding="utf-8")
                config = {}
            else:
                loaded = json.loads(content)
                config = loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            config_path.write_text("{}", encoding="utf-8")
            config = {}

    required_fields = {
        "ai": ["base_url", "api_key", "model"],
        "email": ["url", "login_type", "username", "password", "cookies"],
        "selectors": [
            "search_box",
            "email_list",
            "email_date",
            "email_subject",
            "email_body",
        ],
    }

    for section, keys in required_fields.items():
        section_value = config.get(section)
        if not isinstance(section_value, dict):
            print(f"配置缺少: {section}")
            continue
        for key in keys:
            value = section_value.get(key)
            if key == "cookies":
                # Now we accept either a cookies array or a cookie_file string
                if not isinstance(value, list) and not section_value.get("cookie_file"):
                    print(f"配置缺少: {section}.{key} 或 {section}.cookie_file")
                continue
            if value is None:
                print(f"配置缺少: {section}.{key}")

    return config


def call_llm(prompt: str, config: dict) -> str:
    ai_config = config.get("ai", {})
    base_url = ai_config.get("base_url") or os.getenv("OPENAI_BASE_URL")
    api_key = ai_config.get("api_key") or os.getenv("OPENAI_API_KEY")
    if not base_url or not api_key:
        return "缺少 base_url 或 api_key"

    normalized_base_url = base_url.rstrip("/")
    if normalized_base_url.endswith("/v1"):
        url = f"{normalized_base_url}/chat/completions"
    else:
        url = f"{normalized_base_url}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": ai_config.get("model") or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": "你是一个邮件分析助手。严格基于用户提供的邮件原文作答，绝对禁止编造、臆测或虚构任何邮件标题、发件人、正文内容。如果邮件正文提取失败或内容为空，你必须如实说明'该邮件未提取到有效内容'，不得凭空捏造。只返回纯文本，不要 Markdown 标记。"},
            {"role": "user", "content": prompt},
        ],
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            return f"LLM 调用失败 (HTTP {response.status_code}): {response.text}"
            
        data = response.json()
        return (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
    except Exception as exc:
        return f"LLM 调用失败: {exc}"


async def get_browser_page(config: dict):
    email_config = config.get("email", {})
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=False, channel="msedge", args=["--start-maximized"])
    context = await browser.new_context()
    if email_config.get("login_type") == "cookie":
        cookie_file = email_config.get("cookie_file")
        cookies = email_config.get("cookies", [])
        
        if cookie_file:
            cookie_path = Path(__file__).parent / cookie_file
            if cookie_path.exists():
                with open(cookie_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.split('\t')
                            if len(parts) >= 7:
                                cookies.append({
                                    "name": parts[5],
                                    "value": parts[6],
                                    "domain": parts[0],
                                    "path": parts[2],
                                    "secure": parts[3] == 'TRUE'
                                })
                print(f"✓ 成功从 {cookie_file} 加载 {len(cookies)} 个 Cookie")
            else:
                print(f"⚠️ 找不到 Cookie 文件: {cookie_path}")
                
        if cookies:
            await context.add_cookies(cookies)
    page = await context.new_page()
    await page.goto(
        email_config.get("url"),
        wait_until="networkidle",
        timeout=60000,
    )
    await page.wait_for_timeout(500)
    return playwright, browser, context, page


async def search_emails(page, keyword: str, config: dict = None) -> list:
    if keyword:
        # 有关键词：跳转到搜索页面
        base_url = (config or {}).get("email", {}).get("url", "").rstrip("/")
        # 兼容有无 #path 的 OWA URL
        if "#" in base_url:
            base_url = base_url.split("#")[0].rstrip("/")
        await page.goto(f"{base_url}/#path=/mail/search", wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(2000)
        print("已跳转到搜索页面")
    else:
        # 无关键词：直接查看收件箱最新邮件
        base_url = (config or {}).get("email", {}).get("url", "").rstrip("/")
        if "#" in base_url:
            base_url = base_url.split("#")[0].rstrip("/")
        await page.goto(f"{base_url}/#path=/mail", wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(2000)
        print("已跳转到收件箱（无关键词，查看最新邮件）")
    
    target_frame = page
    for f in page.frames:
        try:
            if await f.locator("input").count() > 3:
                target_frame = f
                print(f"✓ 使用 frame: {f.url[:100]}...")
                break
        except Exception:
            continue

    if keyword:
        # 有关键词时才需要搜索框 —— 优先使用 config 选择器，减少冗余尝试
        primary_selectors = []
        config_sel = (config or {}).get("selectors", {}).get("search_box")
        if config_sel:
            primary_selectors.append(target_frame.locator(config_sel))
        primary_selectors.append(target_frame.get_by_role("searchbox"))

        fallback_selectors = [
            target_frame.locator('input[aria-label="搜索"]'),
            target_frame.locator('input[placeholder*="Search"]'),
        ]

        search_box = None
        # 先尝试主选择器（3s 超时）
        for loc in primary_selectors:
            try:
                await loc.wait_for(state="visible", timeout=3000)
                search_box = loc
                print("✓ 成功定位搜索框（主选择器）")
                break
            except Exception:
                continue
        # 主选择器未命中，尝试备选（2s 超时）
        if not search_box:
            for loc in fallback_selectors:
                try:
                    await loc.wait_for(state="visible", timeout=2000)
                    search_box = loc
                    print("✓ 成功定位搜索框（备选选择器）")
                    break
                except Exception:
                    continue

        if not search_box:
            print("\n⚠️  无法定位搜索框，可能是 cookie 已过期，请更新 config.json 中的 cookies 后重试！")
            await page.screenshot(path="debug_searchbox_final.png")
            raise RuntimeError("无法定位搜索框，可能是 cookie 已过期，请更新 config.json 中的 cookies 后重试。")

        await search_box.click()
        await search_box.fill(keyword)
        await search_box.press("Enter")
        await page.wait_for_timeout(1500)

    print("开始滚动加载...")
    for i in range(5):
        await page.evaluate("window.scrollBy(0, window.innerHeight * 0.8)")
        await page.wait_for_timeout(300)
        print(f"滚动 {i+1}/5 次")

    # 提取邮件（最终稳定版）
    all_items = await target_frame.locator('div[role="option"][data-convid], ._lvv_w[data-convid], [data-convid]').all()
    print(f"找到 {len(all_items)} 个邮件列表项")

    # 提取前20个（足够覆盖最近邮件）
    items = all_items[:20]

    mail_list = []
    import re

    for idx, item in enumerate(items):
        try:
            raw_text = await item.inner_text(timeout=2000)

            lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

            subject = lines[0] if lines else ""

            # 智能日期提取（搜索所有行）
            date_str = ""
            date_pattern = r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})'
            time_pattern = r'(\d{1,2}:\d{2}(\s*[APap][Mm])?)'
            for line in lines:
                match = re.search(date_pattern, line)
                if match:
                    date_str = match.group(1)
                    break
            # 如果没匹配到完整日期，尝试匹配时间（当天邮件只显示时间）
            if not date_str:
                for line in lines:
                    if re.search(time_pattern, line):
                        date_str = datetime.now().strftime('%Y-%m-%d')
                        break
            if not date_str and len(lines) > 1:
                date_str = lines[1]

            href = ""
            try:
                href = await item.locator("a").first.get_attribute("href", timeout=2000)
                if href and not href.startswith("http"):
                    mail_base = (config or {}).get("email", {}).get("url", "").split("#")[0].rstrip("/")
                    # Extract scheme + host only
                    from urllib.parse import urlparse
                    parsed = urlparse(mail_base)
                    mail_origin = f"{parsed.scheme}://{parsed.netloc}"
                    href = mail_origin + href
            except:
                pass

            if subject.strip():
                mail_list.append({
                    "subject": subject.strip(),
                    "date_str": date_str,
                    "href": href,
                    "raw_date": date_str,
                    "locator": item  # 保存 locator 对象
                })
                print(f"邮件项 {idx+1} 提取成功: {subject.strip()[:70]} | 日期: {date_str}")
        except Exception as e:
            print(f"邮件项 {idx+1} 处理失败: {e}")
            continue

    # 使用 datetime 精确排序（从新到旧）
    def parse_date(d):
        if not d:
            return datetime(1900, 1, 1)
        try:
            # 支持 2025/12/22、2025-12-22、2025/12/2 等格式
            d = d.replace('/', '-').strip()
            if len(d.split('-')) == 3:
                return datetime.strptime(d, '%Y-%m-%d')
            # 如果包含时间格式（当天邮件），视为今天
            elif re.search(r'\d{1,2}:\d{2}', d):
                return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                return datetime(1900, 1, 1)
        except:
            return datetime(1900, 1, 1)

    mail_list.sort(key=lambda x: parse_date(x["raw_date"]), reverse=True)
    final_items = mail_list[:10]

    emails = []
    for m in final_items:
        emails.append({
            "subject": m["subject"], 
            "date": m["date_str"], 
            "href": m["href"],
            "locator": m["locator"]
        })
        print(f"最终前10封邮件 {len(emails)}: {m['subject'][:75]} | 日期: {m['date_str']}")

    print(f"最终提取到最近前10封邮件")
    return emails


async def extract_full_body(page, item_locator) -> str:
    try:
        # 点击邮件列表项
        await item_locator.click()
        # 等待阅读窗格加载
        await page.wait_for_timeout(500)
        
        # 寻找包含正文的 frame
        frames = page.frames
        content_frame = page
        
        # 尝试找到 ReadingPane 或正文容器
        # 策略：遍历所有 frame，看哪个包含典型的邮件正文结构
        found_frame = False
        
        # 1. 尝试直接在当前 page 查找（非 iframe 模式）
        # Outlook OWA 可能会使用 iframe，也可能直接渲染
        
        # 定义可能的正文选择器
        body_selectors = [
            'div[role="document"]',
            'div.gs div.ii',
            'div[role="main"]',
            'div[aria-label*="正文"]',
            'div[aria-label*="Body"]',
            'div.AllowTextSelection',
            'div.rps_5055' # 常见的随机类名，可能不稳定，但有时有用
        ]
        
        # 遍历 frames 寻找
        for f in frames:
            for sel in body_selectors:
                try:
                    if await f.locator(sel).count() > 0:
                        content_frame = f
                        found_frame = True
                        break
                except:
                    continue
            if found_frame:
                break
        
        body_text = ""
        for sel in body_selectors:
            try:
                locator = content_frame.locator(sel).first
                if await locator.is_visible(timeout=2000):
                    body_text = await locator.inner_text(timeout=2000)
                    if len(body_text) > 50:
                        break
            except:
                continue
                
        # 如果还是没找到，尝试提取所有文本
        if not body_text:
             body_text = await content_frame.inner_text()

        # 清理正文
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(body_text, 'html.parser')
        for tag in soup(["script", "style", "header", "footer", "nav", "button"]):
            tag.decompose()
        clean_text = soup.get_text(separator="\n", strip=True)
        
        # 返回列表页（如果需要，比如手机版布局，但在桌面版 OWA 通常不需要显式返回，点击下一个即可）
        # 这里为了稳健，不执行显式后退，因为 OWA 是单页应用，点击下一个列表项通常会自动切换
        
        return clean_text.strip()
        
    except Exception as e:
        print(f"提取正文失败: {e}")
        return f"[正文提取失败: {str(e)[:100]}]"


async def main() -> None:
    load_dotenv()
    config_path = Path(__file__).with_name("config.json")
    config = load_config(config_path)
    email_url = config.get("email", {}).get("url")
    search_box = config.get("selectors", {}).get("search_box")
    print(f"邮箱 URL: {email_url}")
    print(f"search_box 选择器: {search_box}")
    playwright = None
    browser = None
    context = None
    try:
        playwright, browser, context, page = await get_browser_page(config)
        
        # 1. 用户手动输入关键词和总结指令
        search_keyword = input("请输入搜索关键词（留空则查看最新邮件）：").strip()
            
        user_instruction = input("请输入总结指令（例如：只告诉我前3封的内容、只总结有活动的邮件、用简单中文说一下）：").strip()
        if not user_instruction:
            user_instruction = "请用自然语气总结这10封邮件，重点关注活动、课程作业和重要事项。"

        try:
            emails = await search_emails(page, search_keyword, config=config)
        except RuntimeError as e:
            print(f"\n❌ {e}")
            return
        print(f"成功提取到 {len(emails)} 封邮件列表")
        
        print("正在提取10封邮件完整正文...")
        full_bodies = []
        for i, e in enumerate(emails):
            print(f"正在提取第 {i+1} 封邮件正文: {e['subject'][:30]}...")
            # 传入 locator 对象进行点击提取
            body = await extract_full_body(page, e["locator"])
            full_bodies.append(f"=== 邮件 {i+1}: {e['subject']} ===\n日期: {e['date']}\n\n{body}\n\n{'='*80}\n")
            # 稍作等待，模拟人类操作
            await page.wait_for_timeout(1000)

        if len(full_bodies) <= 4:
            aggregated_text = "\n".join(full_bodies)
            total_chars = len(aggregated_text)
        else:
            aggregated_text = "\n".join(full_bodies)
            total_chars = len(aggregated_text)

        CHARS_PER_CHUNK = 6000

        if total_chars <= CHARS_PER_CHUNK:
            summary_prompt = f"""
当前日期：{datetime.now().strftime('%Y-%m-%d')}

【用户指令（最高优先级，必须严格遵守）】：
{user_instruction}

请严格按照用户指令处理以下邮件。用户指令决定了你应该关注哪些邮件、输出什么内容、用什么风格。
【严禁编造】只能基于下方邮件原文作答，不得虚构任何内容。提取失败的邮件如实说明。
如果用户指令没有指定输出格式，则默认对涉及的每封邮件按以下格式输出：
- 邮件标题：[标题]
- 发件人：[发件人]
- 内容总结：[简练的总结]
- 重要程度：[1-5星，如 ★★★☆☆]

以下是全部 {len(full_bodies)} 封邮件的内容（供你根据用户指令选择性处理）：
{aggregated_text}
"""
            print(f"\n正在调用 LLM（单次调用，{total_chars} 字符）...")
            summary = call_llm(summary_prompt, config)
        else:
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

            print(f"\n智能分块: {total_chars} 字符 → {len(chunks)} 块")

            chunk_summaries = []
            for idx, chunk_text in enumerate(chunks):
                print(f"\n正在调用 LLM 分析第 {idx+1}/{len(chunks)} 批邮件...")
                chunk_prompt = f"""
当前日期：{datetime.now().strftime('%Y-%m-%d')}

【用户指令（最高优先级）】：
{user_instruction}

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
                chunk_summary = call_llm(chunk_prompt, config)
                chunk_summaries.append(f"【第 {idx+1} 批分析结果】\n{chunk_summary}")

            print("\n正在调用 LLM 生成最终总结...")
            aggregated_summaries = "\n\n".join(chunk_summaries)
            final_prompt = f"""
当前日期：{datetime.now().strftime('%Y-%m-%d')}

【用户指令（最高优先级，必须严格遵守）】：
{user_instruction}

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
            summary = call_llm(final_prompt, config)
        print("\n=== LLM 生成的邮件总结 ===\n")
        print(summary)
        print("\n总结完成！")

    finally:
        try:
            if context:
                await context.close()
            if browser:
                await browser.close()
            if playwright:
                await playwright.stop()
        except Exception:
            pass
    print("浏览器已安全关闭，Task 6 测试完成")


if __name__ == "__main__":
    asyncio.run(main())