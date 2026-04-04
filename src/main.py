# requirements.txt 依赖由 pip install -r requirements.txt 安装
from pathlib import Path
import asyncio
import json
import os
import re
from typing import Optional

import requests
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from calendar import month_abbr

_LIST_ITEM_SELECTOR = 'div[role="option"][data-convid], ._lvv_w[data-convid], [data-convid]'

_SCROLL_OWA_LIST_STEP_JS = """({ containerFraction, minDelta }) => {
    const cf = Number(containerFraction) || 0.88;
    const md = Number(minDelta) || 80;
    const nodes = document.querySelectorAll('[data-convid]');
    const n = nodes.length;
    if (!n) {
        window.scrollBy(0, Math.floor(window.innerHeight * Math.min(0.85, cf)));
        return { mode: "window", n: 0 };
    }
    const idx = Math.min(Math.floor(n / 2), n - 1);
    const opt = nodes[idx];
    let el = opt;
    while (el && el !== document.documentElement) {
        const st = window.getComputedStyle(el);
        const oy = st.overflowY;
        if ((oy === "auto" || oy === "scroll" || oy === "overlay") && el.scrollHeight > el.clientHeight + 8) {
            const delta = Math.max(md, Math.floor(el.clientHeight * cf));
            const next = Math.min(el.scrollTop + delta, el.scrollHeight - el.clientHeight);
            el.scrollTop = next;
            return { mode: "container", n: n };
        }
        el = el.parentElement;
    }
    window.scrollBy(0, Math.floor(window.innerHeight * Math.min(0.85, cf)));
    return { mode: "window-fallback", n: n };
}"""

_OWA_SCROLL_READING_PANE_BODY_JS = """() => {
    const visible = (el) =>
        el &&
        el.offsetParent !== null &&
        el.getClientRects &&
        el.getClientRects().length > 0;
    const READING_ROOT_SELS = [
        '[id*="ReadingPane"]',
        '[id*="readingPane"]',
        '[class*="ReadingPane"]',
        '[class*="readingPane"]',
        '[data-app-section="MessageReading"]',
        '[aria-label*="Reading Pane" i]',
        '[aria-label*="阅读窗格"]',
    ];
    const BODY_IN_PANE_SELS = [
        '[aria-label*="Message body" i]',
        '[aria-label*="邮件正文"]',
        'article[role="document"]',
        'div.AllowTextSelection',
        'div.gs div.ii',
        'div.rps_5055',
    ];
    const pickBodyEl = () => {
        for (const rs of READING_ROOT_SELS) {
            let roots;
            try {
                roots = document.querySelectorAll(rs);
            } catch (e) {
                continue;
            }
            for (const root of roots) {
                if (!visible(root)) continue;
                for (const bs of BODY_IN_PANE_SELS) {
                    try {
                        const nodes = root.querySelectorAll(bs);
                        for (const n of nodes) {
                            if (visible(n)) return n;
                        }
                    } catch (e) {}
                }
            }
        }
        const narrow = [
            '[aria-label*="Message body" i]',
            '[aria-label*="邮件正文"]',
        ];
        for (const bs of narrow) {
            try {
                const nodes = document.querySelectorAll(bs);
                for (const n of nodes) {
                    if (visible(n)) return n;
                }
            } catch (e) {}
        }
        return null;
    };
    const el = pickBodyEl();
    if (!el) return 0;
    let moved = 0;
    let w = el;
    for (let depth = 0; depth < 28 && w; depth++) {
        const st = window.getComputedStyle(w);
        const oy = st.overflowY;
        if (
            (oy === "auto" || oy === "scroll" || oy === "overlay") &&
            w.scrollHeight > w.clientHeight + 8
        ) {
            const before = w.scrollTop;
            w.scrollTop = w.scrollHeight;
            moved += Math.abs(w.scrollTop - before);
        }
        w = w.parentElement;
    }
    try {
        el.focus({ preventScroll: true });
    } catch (e) {}
    return moved;
}"""

_RESET_OWA_LIST_SCROLL_JS = """() => {
    const nodes = document.querySelectorAll('[data-convid]');
    if (!nodes.length) {
        window.scrollTo(0, 0);
        return;
    }
    const opt = nodes[0];
    let el = opt;
    while (el && el !== document.documentElement) {
        const st = window.getComputedStyle(el);
        const oy = st.overflowY;
        if ((oy === "auto" || oy === "scroll" || oy === "overlay") && el.scrollHeight > el.clientHeight + 8) {
            el.scrollTop = 0;
            return;
        }
        el = el.parentElement;
    }
    window.scrollTo(0, 0);
}"""


async def _scroll_owa_mail_list_step(
    frame,
    *,
    container_fraction: float = 0.88,
    min_delta: int = 100,
) -> dict:
    """在含邮件列表的 frame 内滚动可滚动父容器；OWA 虚拟列表不随主 window 滚动加载。"""
    try:
        return await frame.evaluate(
            _SCROLL_OWA_LIST_STEP_JS,
            {"containerFraction": container_fraction, "minDelta": min_delta},
        )
    except Exception as exc:
        return {"mode": "error", "err": str(exc)[:120]}


async def _reset_owa_mail_list_scroll(frame) -> None:
    """将邮件列表滚动容器回到顶部，避免长时间下滚后 DOM 第一项不是收件箱第一封。"""
    try:
        await frame.evaluate(_RESET_OWA_LIST_SCROLL_JS)
    except Exception:
        pass


def _line_looks_like_clock_time(line: str) -> bool:
    s = (line or "").strip()
    if len(s) > 32:
        return False
    if re.match(r"^\d{1,2}:\d{2}(\s*[APap][Mm])?$", s):
        return True
    if re.match(r"^(上午|下午|晚上|中午)\s*\d{1,2}:\d{2}", s):
        return True
    if re.match(r"^\d{1,2}:\d{2}\s*$", s):
        return True
    return False


def _extract_date_from_line(line: str) -> str:
    """返回 YYYY-MM-DD 或空；支持行内子串日期。"""
    line = (line or "").strip()
    m = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", line)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime(y, mo, d).strftime("%Y-%m-%d")
        except ValueError:
            pass
    m2 = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", line)
    if m2:
        mo, d, y = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
        try:
            return datetime(y, mo, d).strftime("%Y-%m-%d")
        except ValueError:
            pass
    m3 = re.search(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2}),?\s+(\d{4})",
        line,
        re.I,
    )
    if m3:
        mon_s = m3.group(1).title()[:3]
        try:
            mo = [x.lower() for x in month_abbr].index(mon_s.lower())
            d, y = int(m3.group(2)), int(m3.group(3))
            return datetime(y, mo, d).strftime("%Y-%m-%d")
        except (ValueError, IndexError):
            pass
    return ""


def _extract_date_from_line_safe(line: str, *, max_line_len: int = 44) -> str:
    """
    长行多为预览正文，其中的数字易被误识别为日期；优先短行，长行仅接受独立日期片段。
    """
    line = (line or "").strip()
    if not line:
        return ""
    if len(line) <= max_line_len:
        return _extract_date_from_line(line)
    m = re.search(
        r"(?:^|[\s,;，])(\d{4}[/-]\d{1,2}[/-]\d{1,2})(?:[\s,;，]|$)",
        line,
    )
    if m:
        return _extract_date_from_line(m.group(1))
    return ""


# OWA 列表日期：周一=0 … 周日=6（与 datetime.weekday() 一致）
_CN_WEEKDAY_TO_IDX = {
    "周一": 0,
    "周二": 1,
    "周三": 2,
    "周四": 3,
    "周五": 4,
    "周六": 5,
    "周日": 6,
    "星期一": 0,
    "星期二": 1,
    "星期三": 2,
    "星期四": 3,
    "星期五": 4,
    "星期六": 5,
    "星期日": 6,
    "星期天": 6,
}
_EN_WEEKDAY_TO_IDX = {
    "mon": 0,
    "monday": 0,
    "tue": 1,
    "tues": 1,
    "tuesday": 1,
    "wed": 2,
    "weds": 2,
    "wednesday": 2,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "thursday": 3,
    "fri": 4,
    "friday": 4,
    "sat": 5,
    "saturday": 5,
    "sun": 6,
    "sunday": 6,
}


def _parse_weekday_token(s: str) -> Optional[int]:
    t = (s or "").strip()
    if not t:
        return None
    if t in _CN_WEEKDAY_TO_IDX:
        return _CN_WEEKDAY_TO_IDX[t]
    tl = t.lower().rstrip(".")
    return _EN_WEEKDAY_TO_IDX.get(tl)


def _most_recent_weekday_date(now: datetime, weekday: int) -> datetime:
    """列表「周三」类：取不晚于当前时刻的最近一个该星期（最多回溯 6 天）。"""
    d = now.date()
    for i in range(7):
        cand = d - timedelta(days=i)
        if cand.weekday() == weekday:
            return datetime(cand.year, cand.month, cand.day)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _month_day_to_iso(month: int, day: int, now: datetime) -> str:
    """同一年 M/D；若落在今天之后则视为上一年（年末邮件在年初显示）。"""
    y = now.year
    try:
        candidate = datetime(y, month, day).date()
    except ValueError:
        return ""
    today = now.date()
    if candidate > today:
        try:
            candidate = datetime(y - 1, month, day).date()
        except ValueError:
            return ""
    return candidate.strftime("%Y-%m-%d")


def _parse_hm_clock(s: str) -> Optional[tuple[int, int]]:
    """从字符串末尾解析 HH:MM（可选 AM/PM）。"""
    t = (s or "").strip()
    if not t:
        return None
    m = re.search(r"(\d{1,2}):(\d{2})(?:\s*([APap][Mm]))?\s*$", t)
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    ap = m.group(3)
    if ap:
        ap_l = ap.lower()
        if ap_l == "pm" and h < 12:
            h += 12
        if ap_l == "am" and h == 12:
            h = 0
    if h > 23 or mi > 59:
        return None
    return (h, mi)


def _extract_owa_time_fragment(raw: str) -> str:
    """从长串中抽出最像 OWA 时间列的短片段（周/昨天/前天 + 时间等）。"""
    s = raw or ""
    if not s.strip():
        return ""
    patterns = [
        r"(周[一二三四五六日天]\s+\d{1,2}:\d{2})",
        r"((?:昨天|前天)\s*[,.，]?\s*\d{1,2}:\d{2})",
        r"((?:Yesterday|Today)\s*,?\s*\d{1,2}:\d{2}(?:\s*[APap][Mm])?)",
        r"((?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+\d{1,2}:\d{2}(?:\s*[APap][Mm])?)",
        r"((?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\s+\d{1,2}:\d{2})",
    ]
    for pat in patterns:
        m = re.search(pat, s, re.I)
        if m:
            return m.group(1).strip()
    return ""


def parse_owa_list_datetime(raw_text: str, now: Optional[datetime] = None) -> tuple[str, str]:
    """
    解析 OWA 列表/头部时间，返回 (展示用原文片段, 排序用字符串)。
    排序串为 YYYY-MM-DD 或 YYYY-MM-DD HH:MM（与 parse_email_date_for_filter 兼容）。
    """
    if now is None:
        now = datetime.now()
    s0 = (raw_text or "").strip()
    if not s0:
        return "", ""

    # 长串：只取 ISO 日期或内嵌时间片段，避免误用主题里的活动日期
    if len(s0) > 120:
        iso = _extract_date_from_line(s0) or _extract_date_from_line_safe(s0, max_line_len=300)
        if iso:
            return (iso, iso)
        frag = _extract_owa_time_fragment(s0)
        if frag:
            return parse_owa_list_datetime(frag, now)
        return "", ""

    s = s0
    full = _extract_date_from_line(s) or _extract_date_from_line_safe(
        s, max_line_len=min(len(s), 120)
    )
    if full:
        return (full, full)

    frag2 = _extract_owa_time_fragment(s)
    if frag2 and frag2 != s:
        return parse_owa_list_datetime(frag2, now)

    # --- 复合：周五 15:38 ---
    m_wd_cn = re.match(
        r"^(周[一二三四五六日天])\s+(\d{1,2}):(\d{2})\s*$",
        s.strip(),
    )
    if m_wd_cn:
        wk = m_wd_cn.group(1)
        h, mi = int(m_wd_cn.group(2)), int(m_wd_cn.group(3))
        wd_idx = _CN_WEEKDAY_TO_IDX.get(wk)
        if wd_idx is not None and h <= 23 and mi <= 59:
            ddt = _most_recent_weekday_date(now, wd_idx)
            dt = ddt.replace(hour=h, minute=mi, second=0, microsecond=0)
            sk = dt.strftime("%Y-%m-%d %H:%M")
            return (s.strip(), sk)

    # --- 复合：昨天 14:11 / 昨天, 14:11 ---
    m_y = re.match(r"^(昨天|前天)\s*[,.，]?\s*(\d{1,2}):(\d{2})\s*$", s.strip())
    if m_y:
        days_back = 1 if m_y.group(1) == "昨天" else 2
        h, mi = int(m_y.group(2)), int(m_y.group(3))
        if h <= 23 and mi <= 59:
            base = (now - timedelta(days=days_back)).replace(
                hour=h, minute=mi, second=0, microsecond=0
            )
            sk = base.strftime("%Y-%m-%d %H:%M")
            return (s.strip(), sk)

    # --- English Yesterday / Today + time ---
    s_low = s.lower().strip()
    m_en = re.match(
        r"^(yesterday|today)\s*,?\s*(\d{1,2}):(\d{2})(?:\s*([ap]m))?\s*$",
        s_low,
        re.I,
    )
    if m_en:
        h, mi = int(m_en.group(2)), int(m_en.group(3))
        ap = m_en.group(4)
        if ap:
            if ap.lower() == "pm" and h < 12:
                h += 12
            if ap.lower() == "am" and h == 12:
                h = 0
        days_back = 1 if m_en.group(1).lower() == "yesterday" else 0
        base = (now - timedelta(days=days_back)).replace(
            hour=h, minute=mi, second=0, microsecond=0
        )
        sk = base.strftime("%Y-%m-%d %H:%M")
        return (s.strip(), sk)

    # --- English weekday + time: Mon 3:45 PM ---
    m_wd_en = re.match(
        r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(\d{1,2}):(\d{2})(?:\s*([APap][Mm]))?\s*$",
        s.strip(),
        re.I,
    )
    if m_wd_en:
        tl = m_wd_en.group(1).lower()
        wd_map = {
            "mon": 0,
            "tue": 1,
            "wed": 2,
            "thu": 3,
            "fri": 4,
            "sat": 5,
            "sun": 6,
        }
        wd_idx = wd_map.get(tl)
        if wd_idx is not None:
            h, mi = int(m_wd_en.group(2)), int(m_wd_en.group(3))
            ap = m_wd_en.group(4)
            if ap:
                ap_l = ap.lower()
                if ap_l == "pm" and h < 12:
                    h += 12
                if ap_l == "am" and h == 12:
                    h = 0
            if h <= 23 and mi <= 59:
                ddt = _most_recent_weekday_date(now, wd_idx)
                dt = ddt.replace(hour=h, minute=mi, second=0, microsecond=0)
                sk = dt.strftime("%Y-%m-%d %H:%M")
                return (s.strip(), sk)

    # --- M/D HH:MM ---
    m_md_t = re.match(r"^(\d{1,2})[/-](\d{1,2})\s+(\d{1,2}):(\d{2})\s*$", s.strip())
    if m_md_t:
        mo, d, h, mi = (
            int(m_md_t.group(1)),
            int(m_md_t.group(2)),
            int(m_md_t.group(3)),
            int(m_md_t.group(4)),
        )
        ymd = _month_day_to_iso(mo, d, now)
        if ymd and h <= 23 and mi <= 59:
            y, mo2, da = map(int, ymd.split("-"))
            dt = datetime(y, mo2, da, h, mi, 0)
            return (s.strip(), dt.strftime("%Y-%m-%d %H:%M"))

    # --- 3月28日 15:38 ---
    m_cn_md_t = re.match(
        r"^(\d{1,2})月(\d{1,2})日\s+(\d{1,2}):(\d{2})\s*$",
        s.strip(),
    )
    if m_cn_md_t:
        mo, d = int(m_cn_md_t.group(1)), int(m_cn_md_t.group(2))
        h, mi = int(m_cn_md_t.group(3)), int(m_cn_md_t.group(4))
        ymd = _month_day_to_iso(mo, d, now)
        if ymd and h <= 23 and mi <= 59:
            y, mo2, da = map(int, ymd.split("-"))
            dt = datetime(y, mo2, da, h, mi, 0)
            return (s.strip(), dt.strftime("%Y-%m-%d %H:%M"))

    # 仅时间 -> 今天 + 时间
    if _line_looks_like_clock_time(s):
        hm = _parse_hm_clock(s)
        if hm:
            h, mi = hm
            dt = now.replace(hour=h, minute=mi, second=0, microsecond=0)
            return (s.strip(), dt.strftime("%Y-%m-%d %H:%M"))
        return (s.strip(), now.strftime("%Y-%m-%d"))

    if len(s) > 48:
        return "", ""

    if s in ("昨天",) or s_low == "yesterday":
        d = (now - timedelta(days=1)).date().strftime("%Y-%m-%d")
        return (s, d)
    if s in ("前天",):
        d = (now - timedelta(days=2)).date().strftime("%Y-%m-%d")
        return (s, d)

    wd = _parse_weekday_token(s)
    if wd is not None:
        d = _most_recent_weekday_date(now, wd).strftime("%Y-%m-%d")
        return (s, d)

    m = re.match(r"^(\d{1,2})[/-](\d{1,2})$", s)
    if m:
        mo, d = int(m.group(1)), int(m.group(2))
        iso = _month_day_to_iso(mo, d, now)
        if iso:
            return (s, iso)

    m3 = re.match(
        r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2}),?\s*$",
        s,
        re.I,
    )
    if m3:
        mon_s = m3.group(1).title()[:3]
        try:
            mo = [x.lower() for x in month_abbr].index(mon_s.lower())
            d = int(m3.group(2))
            iso = _month_day_to_iso(mo, d, now)
            if iso:
                return (s, iso)
        except (ValueError, IndexError):
            pass

    m4 = re.match(r"^(\d{1,2})月(\d{1,2})日?$", s)
    if m4:
        mo, d = int(m4.group(1)), int(m4.group(2))
        iso = _month_day_to_iso(mo, d, now)
        if iso:
            return (s, iso)

    return "", ""


def normalize_owa_list_date(raw_text: str, now: Optional[datetime] = None) -> str:
    """兼容旧调用：返回排序键（日期或日期+时间）。"""
    _, sk = parse_owa_list_datetime(raw_text, now)
    return sk


def pick_first_owa_datetime(candidates: list[str], now: datetime) -> tuple[str, str]:
    """按优先级尝试候选串，返回 (展示文本, 排序键)。"""
    seen: set[str] = set()
    for raw in candidates:
        if not raw or not str(raw).strip():
            continue
        key = raw.strip()[:200]
        if key in seen:
            continue
        seen.add(key)
        disp, sk = parse_owa_list_datetime(raw.strip(), now)
        if sk:
            return (disp or raw.strip()[:80], sk)
    return "", ""


def _line_looks_like_metadata_date_token(line: str) -> bool:
    """仅用于 inner_text 行：短且像列表日期列，而非主题/预览长句。"""
    s = (line or "").strip()
    if not s:
        return False
    if len(s) <= 64:
        if re.match(r"^周[一二三四五六日天]\s+\d{1,2}:\d{2}\s*$", s):
            return True
        if re.match(r"^(昨天|前天)\s*[,.，]?\s*\d{1,2}:\d{2}\s*$", s):
            return True
        if re.match(
            r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+\d{1,2}:\d{2}(\s*[APap][Mm])?\s*$",
            s,
            re.I,
        ):
            return True
    if len(s) > 48:
        return False
    if _line_looks_like_clock_time(s):
        return True
    if _extract_date_from_line(s) or _extract_date_from_line_safe(s, max_line_len=48):
        return True
    if re.match(r"^\d{1,2}[/-]\d{1,2}$", s):
        return True
    if re.match(
        r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s*$",
        s,
        re.I,
    ):
        return True
    if re.match(r"^(\d{1,2})月(\d{1,2})日?$", s):
        return True
    if s in ("昨天", "前天") or s.lower() == "yesterday":
        return True
    if _parse_weekday_token(s) is not None:
        return True
    return False


def _pick_datetime_from_inner_metadata_lines(
    lines: list,
    now: datetime,
    *,
    subject: str = "",
    sender: str = "",
) -> tuple[str, str]:
    """仅从看起来像日期列的短行取值，避免扫到主题里的 April 6, 2026。"""
    if not lines:
        return "", ""
    subj = (subject or "").strip()
    snd = (sender or "").strip()
    for line in reversed(lines):
        t = line.strip()
        if not t:
            continue
        if subj and t == subj:
            continue
        if snd and t == snd:
            continue
        if not _line_looks_like_metadata_date_token(t):
            continue
        disp, got = parse_owa_list_datetime(t, now)
        if got:
            return (disp or t, got)
    return "", ""


def _line_likely_owa_sender(line: str) -> bool:
    """OWA 列表常见：首行发件人，次行主题。"""
    s = (line or "").strip()
    if not s:
        return False
    if "@" in s:
        return True
    if re.search(r"\(\s*via\s+[^)]+\)", s, re.I):
        return True
    low = s.lower()
    if low.startswith(("re:", "fw:", "fwd:", "回复", "转发")) or "【" in s[:6]:
        return False
    if len(s) <= 32 and not re.search(r"[。！？!?]", s):
        return True
    return False


def _line_is_date_or_time_only(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return True
    if _line_looks_like_clock_time(s):
        return True
    if _extract_date_from_line_safe(s) and len(s) < 28:
        return True
    if _line_looks_like_metadata_date_token(s) and len(s) < 66:
        return True
    return False


def _infer_owa_list_subject(lines: list) -> str:
    if not lines:
        return ""
    if len(lines) == 1:
        return lines[0]
    a, b = lines[0], lines[1]
    lowa = a.lower()
    if lowa.startswith(("re:", "fw:", "fwd:", "回复", "转发")) or "【" in a[:8]:
        return a
    if len(a) > len(b) + 15 and len(a) >= 28:
        return a

    def _skip_meta(cand: str, idx: int) -> str:
        if idx + 1 < len(lines) and _line_is_date_or_time_only(cand):
            return lines[idx + 1]
        return cand

    if _line_likely_owa_sender(a):
        return _skip_meta(b, 1)
    pick = b if len(b) >= len(a) else a
    if pick == b:
        return _skip_meta(b, 1)
    return pick


def _infer_sender_and_preview(lines: list, subject: str) -> tuple[str, str]:
    """从列表行文本推断发件人（首行）与预览（去掉主题/日期后的剩余行）。"""
    if not lines:
        return "", ""
    subj_l = (subject or "").strip().lower()
    sender = ""
    if _line_likely_owa_sender(lines[0]):
        sender = lines[0].strip()
    elif "@" in lines[0] or re.search(r"\(\s*via\s+[^)]+\)", lines[0], re.I):
        sender = lines[0].strip()
    start_idx = 1 if sender else 0
    prev_parts: list[str] = []
    for ln in lines[start_idx:]:
        t = ln.strip()
        if not t:
            continue
        if subj_l:
            tl = t.lower()
            if tl == subj_l or (len(t) <= len(subj_l) + 8 and subj_l.startswith(tl)):
                continue
        if _line_is_date_or_time_only(t) and len(t) < 36:
            continue
        if re.match(r"^(周[一二三四五六日天]\s+\d{1,2}/\d{1,2})$", t):
            continue
        prev_parts.append(t)
        if sum(len(x) for x in prev_parts) > 520:
            break
    preview = " ".join(prev_parts)[:500]
    return sender, preview


# 与优先级计划 P2 对齐的规则分类（发件人 + 主题 + 列表预览，无 LLM）
EMAIL_CATEGORY_LABELS = (
    "活动",
    "招聘/实习",
    "课程/学习",
    "校友通知",
    "体育/场馆",
    "系统/安全",
    "其他",
)


def classify_email(sender: str, subject: str, preview: str) -> str:
    """
    基于关键词/发件人域名的零成本分类；首条命中规则即返回。
    """
    sub = (subject or "").strip()
    pre = (preview or "").strip()
    snd = (sender or "").strip()
    blob = f"{snd}\n{sub}\n{pre}".lower()

    def _has(*needles: str) -> bool:
        return any(n.lower() in blob for n in needles)

    # 系统 / 安全
    if re.search(r"spam-adm|quarantine|隔离区|unified identity|uim", blob, re.I):
        return "系统/安全"
    if _has("mfa", "验证码", "password reset", "login attempt", "abnormal_location"):
        return "系统/安全"
    if "pycharm team" in blob and "web actions" in blob:
        return "系统/安全"

    # 体育 / 场馆
    if re.search(
        r"sportscentre|sports centre|pec@|体育馆|健身|场地票|领票",
        blob,
        re.I,
    ):
        return "体育/场馆"

    # 招聘 / 实习
    if re.search(
        r"career|校招|招聘|recruitment|internship|实习|宣讲会|campus talk",
        blob,
        re.I,
    ):
        return "招聘/实习"

    # 课程 / 学习（LMS、课号）
    if re.search(r"\(via\s+lm\s+core\)|via lm core", blob, re.I):
        return "课程/学习"
    if re.search(
        r"\b(DTS|ENT|CET|CSE|ACM|DTS\d+|ENT\d+)\d*[A-Z]*[-_]?\d*",
        sub,
        re.I,
    ):
        return "课程/学习"
    if _has("forum", "announcements", "assignment", "lecture", "rescheduled"):
        return "课程/学习"

    # 活动
    if re.search(
        r"sa-office|student activity|【student activity】|art centre|workshop invitation|seminar",
        blob,
        re.I,
    ):
        return "活动"
    if "【" in sub and ("活动" in sub or "参访" in sub or "企业参访" in sub):
        return "活动"

    # 校友通知 / 校务通类
    if re.search(
        r"universitycommunications|liverpool|studyabroad|library|scc@|notice and events",
        blob,
        re.I,
    ):
        return "校友通知"

    return "其他"


def parse_email_date_for_filter(date_str: str) -> Optional[datetime]:
    """
    将列表解析出的日期字符串转为可比较的 datetime（当天 0 点），无法解析则 None。
    支持 YYYY-MM-DD、YYYY-MM-DD HH:MM；兼容仅含时间的旧字符串。
    """
    if not date_str or not str(date_str).strip():
        return None
    d = str(date_str).strip().replace("/", "-")
    m = re.match(
        r"^(\d{4})-(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{1,2})\s*$",
        d,
    )
    if m:
        try:
            return datetime(
                int(m.group(1)),
                int(m.group(2)),
                int(m.group(3)),
                int(m.group(4)),
                int(m.group(5)),
            )
        except (ValueError, TypeError):
            pass
    try:
        parts = d.split("-")
        if len(parts) == 3:
            y, mo, day = int(parts[0]), int(parts[1]), int(parts[2])
            return datetime(y, mo, day)
    except (ValueError, TypeError):
        pass
    if re.search(r"\d{1,2}:\d{2}", d):
        return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    return None


def sort_key_for_list_date(raw_date: str) -> datetime:
    """search_emails 排序：与 parse_email_date_for_filter 一致，无法解析则置为很早日期。"""
    pd = parse_email_date_for_filter(str(raw_date or ""))
    return pd if pd is not None else datetime(1900, 1, 1)


def merge_list_and_pane_datetime(
    list_sort: str,
    list_disp: str,
    pane_sort: str,
    pane_disp: str,
) -> tuple[str, str]:
    """列表解析优先；若列表无排序键则用阅读窗格头部。返回 (展示文本, 排序键)。"""
    ls = (list_sort or "").strip()
    ld = (list_disp or "").strip()
    ps = (pane_sort or "").strip()
    pd = (pane_disp or "").strip()
    if ls:
        return (ld or ls, ls)
    if ps:
        return (pd or ps, ps)
    return ("", "")


_ROW_DATE_DOM_JS = """(el) => {
  if (!el) return '';
  const norm = (y, mo, d) => {
    const mm = String(mo).padStart(2, '0');
    const dd = String(d).padStart(2, '0');
    return y + '-' + mm + '-' + dd;
  };
  const t = el.querySelector('time[datetime]');
  if (t) {
    const dt = (t.getAttribute('datetime') || '').trim();
    const m = dt.match(/^(\\d{4})-(\\d{2})-(\\d{2})/);
    if (m) return m[0];
    const m2 = dt.match(/(\\d{4})[\\/-](\\d{1,2})[\\/-](\\d{1,2})/);
    if (m2) return norm(m2[1], m2[2], m2[3]);
  }
  let best = '';
  const attrs = ['datetime', 'title', 'aria-label'];
  el.querySelectorAll('*').forEach((n) => {
    for (const a of attrs) {
      const v = n.getAttribute && n.getAttribute(a);
      if (!v || v.length < 8) continue;
      const m = v.match(/(\\d{4})[\\/-](\\d{1,2})[\\/-](\\d{1,2})/);
      if (m) best = norm(m[1], m[2], m[3]);
    }
  });
  return best;
}"""


async def _dom_date_from_list_row(item) -> str:
    """OWA 紧凑列表项里日期常在子节点 title/aria-label，不在 inner_text 行里。"""
    try:
        raw = await item.evaluate(_ROW_DATE_DOM_JS)
        s = (raw or "").strip()
        if len(s) >= 10 and s[4] == "-" and s[7] == "-":
            return s[:10]
    except Exception:
        pass
    return ""


async def _prepare_owa_mail_list_frame(page, keyword: str, config: dict = None):
    """
    导航到收件箱或搜索页，返回用于列表解析的 target_frame（与 search_emails 原逻辑一致）。
    """
    if keyword:
        base_url = (config or {}).get("email", {}).get("url", "").rstrip("/")
        if "#" in base_url:
            base_url = base_url.split("#")[0].rstrip("/")
        search_url = f"{base_url}/#path=/mail/search"
        await page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(1250)
        # OWA 偶发未挂载搜索框；与手动操作一致：进入搜索页后再刷新一次更稳定
        await page.reload(wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(1250)
        print("已跳转到搜索页面（已刷新一次）")
    else:
        base_url = (config or {}).get("email", {}).get("url", "").rstrip("/")
        if "#" in base_url:
            base_url = base_url.split("#")[0].rstrip("/")
        await page.goto(f"{base_url}/#path=/mail", wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(1250)
        print("已跳转到收件箱（无关键词，查看最新邮件）")

    candidate_frames = [page, *page.frames]

    async def pick_best_mail_frame():
        ranked = []
        for idx, frame in enumerate(candidate_frames):
            try:
                input_count = await frame.locator("input").count()
                mail_item_count = await frame.locator('[data-convid], div[role="option"][data-convid]').count()
                frame_url = (getattr(frame, "url", "") or "").lower()
                score = mail_item_count * 20 + min(input_count, 8)
                if "mail" in frame_url or "owa" in frame_url or "outlook" in frame_url:
                    score += 5
                ranked.append((score, idx, frame, input_count, mail_item_count, frame_url))
            except Exception:
                continue

        if not ranked:
            return page

        ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        best = ranked[0]
        print(
            f"✓ 使用 frame: {best[5][:100] or 'main page'} "
            f"(score={best[0]}, inputs={best[3]}, mails={best[4]})"
        )
        return best[2]

    target_frame = await pick_best_mail_frame()

    if keyword:
        config_sel = (config or {}).get("selectors", {}).get("search_box")
        search_box = None
        search_frame = target_frame

        selector_factories = []
        if config_sel:
            selector_factories.append(("配置选择器", lambda frame: frame.locator(config_sel).first))
        selector_factories.extend([
            ("role=searchbox", lambda frame: frame.get_by_role("searchbox").first),
            ("aria-label*=Search", lambda frame: frame.locator('input[aria-label*="Search" i]').first),
            ("aria-label*=搜索", lambda frame: frame.locator('input[aria-label*="搜索" i]').first),
            ("placeholder*=Search", lambda frame: frame.locator('input[placeholder*="Search" i]').first),
            ("placeholder*=搜索", lambda frame: frame.locator('input[placeholder*="搜索" i]').first),
            ("type=search", lambda frame: frame.locator('input[type="search"]').first),
            ("name=q", lambda frame: frame.locator('input[name="q"]').first),
        ])

        frames_to_try = [target_frame] + [f for f in candidate_frames if f is not target_frame]
        for frame in frames_to_try:
            for label, factory in selector_factories:
                try:
                    loc = factory(frame)
                    await loc.wait_for(state="visible", timeout=4000)
                    search_box = loc
                    search_frame = frame
                    print(f"✓ 成功定位搜索框（{label}）")
                    break
                except Exception:
                    continue
            if search_box:
                break

        if not search_box:
            print("\n⚠️  无法定位搜索框，可能是 Cookie 已过期，或邮箱页面结构发生变化。")
            await page.screenshot(path="debug_searchbox_final.png")
            raise RuntimeError(
                "无法定位搜索框。可能是 Cookie 已过期，或邮箱页面结构发生变化；已保存 debug_searchbox_final.png 供排查。"
            )

        await search_box.click()
        await search_box.fill(keyword)
        await search_box.press("Enter")
        await page.wait_for_timeout(950)
        target_frame = search_frame

    return target_frame


async def _parse_list_item_row(item, config: dict = None) -> Optional[dict]:
    """从列表项 locator 解析主题、日期、链接；失败或空主题返回 None。"""
    try:
        now = datetime.now()
        date_candidates: list[str] = []

        try:
            if await item.locator("time[datetime]").count() > 0:
                iso = await item.locator("time[datetime]").first.get_attribute("datetime")
                if iso and len(iso) >= 10:
                    date_candidates.append(iso[:10].replace("/", "-"))
        except Exception:
            pass

        dom_d = await _dom_date_from_list_row(item)
        if dom_d:
            date_candidates.append(dom_d)

        sel_date = (config or {}).get("selectors", {}).get("email_date")
        if sel_date:
            try:
                dt = await item.locator(sel_date).first.inner_text(timeout=600)
                if dt and dt.strip():
                    date_candidates.append(dt.strip())
            except Exception:
                pass

        row_aria = ""
        try:
            row_aria = (await item.get_attribute("aria-label")) or ""
        except Exception:
            row_aria = ""

        subject = ""
        sel_subj = (config or {}).get("selectors", {}).get("email_subject")
        if sel_subj:
            try:
                st = await item.locator(sel_subj).first.inner_text(timeout=800)
                if st and st.strip():
                    subject = st.strip()
            except Exception:
                pass
        if not subject and row_aria.strip():
            parts = [p.strip() for p in re.split(r"[,，;；|]", row_aria) if p.strip()]
            if len(parts) >= 2:
                subject = parts[1][:500]
            elif parts:
                subject = parts[0][:500]
        if not subject:
            try:
                tit = await item.locator("a").first.get_attribute("title", timeout=800)
                if tit and len(tit.strip()) > 2:
                    subject = tit.strip()[:500]
            except Exception:
                pass

        raw_text = await item.inner_text(timeout=2000)
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if not subject:
            subject = _infer_owa_list_subject(lines)

        for ln in lines:
            frag_ln = _extract_owa_time_fragment(ln)
            if frag_ln:
                date_candidates.append(frag_ln)

        if row_aria.strip():
            date_candidates.append(row_aria.strip())

        try:
            row_title = await item.get_attribute("title")
            if row_title and row_title.strip():
                date_candidates.append(row_title.strip())
        except Exception:
            pass

        date_display, date_str = pick_first_owa_datetime(date_candidates, now)

        if not date_str:
            snd_guess = ""
            if lines and _line_likely_owa_sender(lines[0]):
                snd_guess = lines[0].strip()
            date_display, date_str = _pick_datetime_from_inner_metadata_lines(
                lines,
                now,
                subject=subject.strip() if subject else "",
                sender=snd_guess,
            )

        href = ""
        try:
            href = await item.locator("a").first.get_attribute("href", timeout=2000)
            if href and not href.startswith("http"):
                from urllib.parse import urlparse

                mail_base = (config or {}).get("email", {}).get("url", "").split("#")[0].rstrip("/")
                parsed = urlparse(mail_base)
                mail_origin = f"{parsed.scheme}://{parsed.netloc}"
                href = mail_origin + href
        except Exception:
            pass

        if not subject.strip():
            return None
        sender, preview = _infer_sender_and_preview(lines, subject.strip())
        disp_out = (date_display or date_str or "").strip()
        return {
            "subject": subject.strip(),
            "date_str": date_str,
            "date_display": disp_out,
            "href": href,
            "raw_date": date_str,
            "sender": sender,
            "preview": preview,
        }
    except Exception:
        return None


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
    
    # 优先使用用户配置的 apiKey，若未配置或为空字符串，则 Fallback 到全局环境变量
    api_key = ai_config.get("api_key")
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")

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


# 人类语言 token：拉丁词（含撇号、连字符）、数字段、CJK 单字（统计正文规模用）
_HUMAN_TOKEN = re.compile(
    r"[0-9]+(?:[.,][0-9]+)*|"
    r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:'[A-Za-zÀ-ÖØ-öø-ÿ]+)?(?:-[A-Za-zÀ-ÖØ-öø-ÿ]+)*|"
    r"[\u4e00-\u9fff]",
    re.UNICODE,
)


def _human_token_spans(text: str) -> list:
    return [(m.start(), m.end()) for m in _HUMAN_TOKEN.finditer(text or "")]


def count_words_human(text: str) -> int:
    return len(_human_token_spans(text))


LLM_PARALLEL_BATCH_SIZE = 3


def format_human_email_fragment(
    subject: str,
    date: str,
    body: str,
    *,
    sender: str = "",
    date_display: str = "",
    part_index: int = 1,
    part_total: int = 1,
) -> str:
    """仅自然语言：主题、发件人、日期、正文。续段用一句人话衔接，无机器分隔符。"""
    subject = (subject or "").strip() or "无主题"
    # Prefer normalized sort key (YYYY-MM-DD / YYYY-MM-DD HH:MM) for LLM so merge step can echo ISO dates.
    disp = (date or date_display or "").strip() or "无日期"
    sender = (sender or "").strip()
    body = (body or "").strip()
    lines: list = []
    if part_total > 1 and part_index > 1:
        lines.append("以下内容紧接上一段，为同一封邮件的连续正文。")
        lines.append("")
    lines.append(f"主题：{subject}")
    if sender:
        lines.append(f"发件人：{sender}")
    lines.append(f"日期：{disp}")
    lines.append("")
    lines.append(body)
    return "\n".join(lines)


def total_extracted_body_words(items: list) -> int:
    return sum(count_words_human(str(x.get("body") or "")) for x in items)


def normalize_parallel_llm_result(result: object) -> str:
    if isinstance(result, Exception):
        return f"本封分析失败：{type(result).__name__}: {result}"
    return str(result)


def build_per_email_analysis_prompt(
    *,
    today: str,
    instruction: str,
    email_human_text: str,
    email_index: int,
    email_total: int,
) -> str:
    return f"""当前日期：{today}

【用户指令（最高优先级，必须严格遵守）】：
{instruction}

下面仅有第 {email_index}/{email_total} 封邮件的全部可读内容。请只根据这一封作答，不要提及其他邮件。
【严禁编造】只能基于下方原文作答。若正文缺失或提取失败须如实说明。
若用户指令未规定格式，默认按（每个字段必须单独占一行，只写该字段内容；多封邮件之间空一行）：
- 邮件标题：
- 发件人：（优先使用邮件元数据中已给出的「发件人」字段；若元数据未提供则从正文推断；均无则说明未提供）
- 发件日期：（优先使用邮件元数据中已给出的「日期」行；与元数据一致，勿编造；无则写「未提供」）
- 内容总结：
- 重要程度：（1-5 星）

邮件内容：
{email_human_text}
"""


def build_final_merge_prompt(
    *,
    today: str,
    instruction: str,
    per_email_sections: str,
    email_count: int,
) -> str:
    n = max(1, int(email_count))
    return f"""当前日期：{today}

【用户指令（最高优先级，必须严格遵守）】：
{instruction}

下面共有 {n} 封邮件的初步分析（按「第 1 封」…「第 {n} 封」排列）。请严格依据这些分析生成最终输出，满足用户指令。
【严禁编造】不得新增下方未出现的邮件或事实。

【输出结构与数量（务必遵守）】
除非用户指令里**明确**写了只要其中部分邮件（例如点明序号、或「仅活动类」等可执行的过滤条件），否则你必须输出恰好 {n} 个邮件块，与第 1～{n} 封**一一对应、顺序一致**：禁止把多封合并成一个块，禁止省略某一封（提取失败或无话可写时也要保留该块，在内容总结中如实说明）。

【格式硬性约束（逐字遵守，不得增减修改字段名）】
1. 第一行**必须直接**以「邮件标题：」开头（不要写任何引言、说明、序号前缀、编号或空行）。
2. 每个邮件块恰好 5 行，字段名与冒号完全固定（半角或全角冒号均可），示例：
邮件标题：XXX
发件人：XXX
发件日期：YYYY-MM-DD 或 YYYY-MM-DD HH:MM（须与下方初步分析中的日期一致，禁止编造；无则写「未提供」）
内容总结：XXX
重要程度：3星
3. 块与块之间仅用一个空行分隔，「重要程度」必须为半角 1～5 的单个数字紧跟「星」。
4. 输出中**不允许出现序号标记**（如「第1封」「1.」「#1」）和任何花哨格式（如 Markdown 加粗、标题、列表符号）。

若用户指令确实只要部分邮件，只能省略用户明确要求排除的序号，其余仍须按上述格式输出。

各封邮件的初步分析：
{per_email_sections}
"""


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
        wait_until="domcontentloaded",
        timeout=60000,
    )
    await page.wait_for_timeout(500)
    return playwright, browser, context, page


async def _append_visible_list_rows(
    target_frame,
    config: dict,
    *,
    mail_list: list,
    seen_cv: set,
    dedupe_convid: bool,
    pool_size: int,
    incremental_merge: bool,
) -> int:
    """扫描当前视口内列表行，按首次出现顺序追加到 mail_list；返回本轮新增条数。"""
    before = len(mail_list)
    if len(mail_list) >= pool_size:
        return 0
    all_items = await target_frame.locator(_LIST_ITEM_SELECTOR).all()
    for idx, item in enumerate(all_items):
        if len(mail_list) >= pool_size:
            break
        try:
            cvid = (await item.get_attribute("data-convid")) or ""
            if cvid and cvid in seen_cv and (incremental_merge or dedupe_convid):
                continue
            meta = await _parse_list_item_row(item, config)
            if not meta:
                continue
            mail_list.append(
                {
                    "subject": meta["subject"],
                    "date_str": meta["date_str"],
                    "date_display": meta.get("date_display") or meta["date_str"],
                    "href": meta["href"],
                    "raw_date": meta["raw_date"],
                    "sender": meta.get("sender") or "",
                    "preview": meta.get("preview") or "",
                    "locator": item,
                    "convid": cvid,
                }
            )
            if cvid:
                seen_cv.add(cvid)
            print(
                f"邮件项 {len(mail_list)} 提取成功: {meta['subject'][:70]} | 日期: {meta['date_str']}"
            )
        except Exception as e:
            print(f"邮件项(视口) {idx+1} 处理失败: {e}")
            continue
    return len(mail_list) - before


async def search_emails(
    page,
    keyword: str,
    config: dict = None,
    max_emails: int = 10,
    mode: str = "daily",
    *,
    sort_by_date: bool = True,
    dedupe_convid: bool = False,
    list_scroll_pause_ms: Optional[int] = None,
    list_scroll_step_fraction: Optional[float] = None,
    deep_list_initial_wait_ms: Optional[int] = None,
    deep_stagnation_pause_ms: Optional[int] = None,
    deep_stagnation_limit: Optional[int] = None,
) -> list:
    target_frame = await _prepare_owa_mail_list_frame(page, keyword, config)

    pool_size = max(max_emails, 50) if mode == "deep" else max_emails
    mail_list: list = []
    seen_cv: set = set()

    if mode == "deep":
        # 从列表顶开始，小步慢滚 + 去重合并，避免「猛滚到底」后 DOM 第一项变成很旧的邮件
        step_frac = (
            list_scroll_step_fraction
            if list_scroll_step_fraction is not None
            else 0.36
        )
        pause_ms = list_scroll_pause_ms if list_scroll_pause_ms is not None else 550
        max_steps = max(72, pool_size * 2)
        stagnation_limit = (
            deep_stagnation_limit if deep_stagnation_limit is not None else 15
        )
        initial_wait = (
            deep_list_initial_wait_ms
            if deep_list_initial_wait_ms is not None
            else 850
        )
        stagnation_wait = (
            deep_stagnation_pause_ms
            if deep_stagnation_pause_ms is not None
            else 1500
        )

        print(
            f"开始深度加载列表（自顶向下增量，步长≈{step_frac:.2f} 屏，间隔 {pause_ms}ms）..."
        )
        await _reset_owa_mail_list_scroll(target_frame)
        await page.wait_for_timeout(initial_wait)

        stagnation = 0
        for step in range(max_steps):
            if len(mail_list) >= pool_size:
                break
            added = await _append_visible_list_rows(
                target_frame,
                config,
                mail_list=mail_list,
                seen_cv=seen_cv,
                dedupe_convid=dedupe_convid,
                pool_size=pool_size,
                incremental_merge=True,
            )
            if added == 0:
                stagnation += 1
                if stagnation >= stagnation_limit:
                    print(f"连续 {stagnation_limit} 轮无新邮件，停止滚动（已 {len(mail_list)} 条）")
                    break
                # 停滞时做大步冲刺滚动，触发 OWA 异步加载
                await _scroll_owa_mail_list_step(
                    target_frame,
                    container_fraction=0.95,
                    min_delta=300,
                )
                await page.wait_for_timeout(stagnation_wait)
            else:
                stagnation = 0
                # 正常滚动
                await _scroll_owa_mail_list_step(
                    target_frame,
                    container_fraction=step_frac,
                    min_delta=48,
                )
                await page.wait_for_timeout(pause_ms)
            if len(mail_list) >= pool_size:
                break
            print(f"深度滚动 {step + 1}/{max_steps}，已累积 {len(mail_list)} 封")

        print(f"深度模式共解析到 {len(mail_list)} 个邮件列表项（去重后顺序）")
    else:
        print(f"开始滚动加载 (日常模式)...")
        scroll_times = 4
        for i in range(scroll_times):
            await _scroll_owa_mail_list_step(target_frame)
            await page.wait_for_timeout(280)
            print(f"滚动 {i+1}/{scroll_times} 次")
        await _reset_owa_mail_list_scroll(target_frame)
        await page.wait_for_timeout(420)
        await _append_visible_list_rows(
            target_frame,
            config,
            mail_list=mail_list,
            seen_cv=seen_cv,
            dedupe_convid=dedupe_convid,
            pool_size=pool_size,
            incremental_merge=False,
        )
        print(f"日常模式找到 {len(mail_list)} 个邮件列表项")

    # 使用 datetime 精确排序（从新到旧）；与 parse_email_date_for_filter 同一套解析
    if sort_by_date:
        mail_list.sort(key=lambda x: sort_key_for_list_date(x["raw_date"]), reverse=True)
    final_items = mail_list[:max_emails]

    emails = []
    for m in final_items:
        emails.append(
            {
                "subject": m["subject"],
                "date": m["date_str"],
                "date_display": m.get("date_display") or m["date_str"],
                "href": m["href"],
                "locator": m["locator"],
                "convid": (m.get("convid") or ""),
                "sender": m.get("sender") or "",
                "preview": m.get("preview") or "",
            }
        )
        print(
            f"最终邮件 {len(emails)}: {m['subject'][:75]} | 日期: {m.get('date_display') or m['date_str']}"
        )

    print(f"最终提取到最近前 {len(emails)} 封邮件")
    return emails


def _tail_after_last_long_breadcrumb_line(text: str) -> str:
    """取最后一行「含 » 且较长」之后的文本；用于区分列表预览与阅读窗格全文。"""
    lines = (text or "").splitlines()
    last_i = -1
    for i, ln in enumerate(lines):
        s = ln.strip()
        if "»" in s and len(s) > 85:
            last_i = i
    if last_i < 0:
        return (text or "").strip()
    return "\n".join(lines[last_i + 1 :]).strip()


def _looks_like_lms_list_row_preview(text: str) -> bool:
    """
    OWA 虚拟列表行里的 LM Core / 课程通知预览：面包屑 + 单行截断，不是阅读窗格全文。
    与「同结构但后面还有大段正文」的已打开邮件区分。
    """
    if not text:
        return False
    t = text.strip()
    n = len(t)
    if n > 8500:
        return False
    if t.count("\n\n") >= 4 and n > 4000:
        return False
    has_crumb = "»" in t and (
        "Forums" in t
        or "Forum" in t
        or "Announcements" in t
        or "Annoucements" in t
    )
    long_crumb_line = any(
        "»" in ln and len(ln.strip()) > 95 for ln in t.splitlines()
    )
    tail = _tail_after_last_long_breadcrumb_line(t)
    if has_crumb and long_crumb_line:
        if len(tail) < 380:
            return True
        return False
    if has_crumb and "(via LM Core)" in t and n < 4200 and len(tail) < 500:
        return True
    return False


def _reading_pane_activation_ok(text: str, expected_subject: str) -> bool:
    """点击列表项后，阅读区是否已像「单封邮件正文」而非列表预览/混排。"""
    if not (text or "").strip():
        return False
    # 列表预览里会重复出现主题，不能先于预览检测用「含主题」判断
    if _looks_like_mixed_mail_list(text):
        return False
    if _looks_like_lms_list_row_preview(text):
        return False
    if _body_contains_expected_subject(text, expected_subject):
        return True
    n = len(text.strip())
    if n >= 1400:
        return True
    if n >= 650:
        if re.search(r"(?m)^(Dear|Hi|Hello|各位|亲爱的)\b", text, re.I):
            return True
        if text.count("\n\n") >= 1:
            return True
        if n >= 950:
            return True
        return False
    if n >= 260 and re.search(r"(?m)^(Dear|Hi|Hello)\s+\w+", text, re.I):
        lines = [x.strip() for x in text.splitlines() if x.strip()]
        if len(lines) >= 5:
            return True
    return False


def _text_looks_abruptly_truncated(text: str) -> bool:
    """
    OWA 阅读窗格常做虚拟化：未滚到底时 inner_text 会在句中截断。
    用于触发「再滚动 + 重采」；规则偏保守以减少误判。
    """
    t = (text or "").rstrip()
    n = len(t)
    if n < 120 or n > 16000:
        return False
    if t[-1] in ".!?。！？…\"')】」":
        return False
    lines = [x.strip() for x in t.splitlines() if x.strip()]
    if not lines:
        return False
    tail = lines[-1]
    if len(tail) < 28:
        return False
    if tail[-1] in "。！？.!?…":
        return False
    if tail.endswith(":") or tail.endswith("："):
        return n < 5200
    tail_lower = tail.lower()
    if re.search(
        r"\b(the|a|an|to|and|or|of|for|in|is|are|be|with|into|at|on|by|from|as)\s*$",
        tail_lower,
    ):
        return True
    if re.search(r"[a-z]\s*$", tail) and len(tail) > 40:
        return True
    # 最后一行很长却以字母/数字结尾且无句末标点（如 “…Engineering 20”）
    if (
        len(tail) > 52
        and n < 8000
        and n > 160
        and tail[-1].isalnum()
        and not re.search(r"[。！？.!?…][\s\"'」\])]*$", tail)
    ):
        return True
    return False


async def _owa_scroll_reading_pane_for_lazy_body(page, *, rounds: int = 14) -> None:
    """在各 frame 内把邮件正文可滚动祖先滚到底，促使 OWA 挂载完整 DOM。"""
    stale = 0
    for r in range(max(1, rounds)):
        total_moved = 0.0
        for fr in list(page.frames):
            try:
                delta = await fr.evaluate(_OWA_SCROLL_READING_PANE_BODY_JS)
                if isinstance(delta, (int, float)) and delta > 0:
                    total_moved += float(delta)
            except Exception:
                continue
        pause = 75 if r < 3 else min(110 + r * 8, 220)
        await page.wait_for_timeout(pause)
        if total_moved < 0.5:
            stale += 1
        else:
            stale = 0
        if r >= 7 and stale >= 5:
            break


def _owa_body_candidate_score(text: str) -> float:
    """分越高越像单封阅读窗格正文（惩罚整页列表+导航）。"""
    if not text or len(text) < 40:
        return -1e9
    head = text[:2200]
    score = 0.0
    if _looks_like_lms_list_row_preview(text):
        score -= 260.0
    if "搜索邮件和人员" in head or "Search mail and people" in head:
        score -= 120.0
    if "收藏夹" in head[:900] and "收件箱" in head[:900]:
        score -= 80.0
    if "Inbox" in head[:600] and "Drafts" in head[:1200] and "Sent" in head[:1200]:
        score -= 70.0
    if re.search(r"总共\s+\d+\s+个项目.*已加载", text):
        score -= 220.0
    date_line_count = 0
    for ln in text.splitlines()[:500]:
        s = (ln or "").strip()
        if re.match(r"^(周[一二三四五六日天]\s+\d{1,2}/\d{1,2})$", s):
            date_line_count += 1
        elif re.match(r"^\d{4}[/-]\d{1,2}[/-]\d{1,2}$", s):
            date_line_count += 1
    if date_line_count >= 8:
        score -= min(150.0, date_line_count * 10.0)
    if head.count("\n") > 35 and len(head) > 1800:
        score -= 25.0
    lines = text.splitlines()
    if len(text) > 25000 and len(lines) > 180:
        short_lines = sum(1 for ln in lines[:300] if 0 < len(ln.strip()) < 120)
        if short_lines > 120:
            score -= 60.0
    return score


def _strip_owa_list_chrome_from_body(text: str) -> str:
    """
    OWA 有时把文件夹树+列表与阅读区拼进同一 inner_text。
    从「拟办事项」、隐私提示或英文信头起截断，保留当前打开邮件正文。
    """
    if not text or len(text) < 200:
        return text
    m_footer = re.search(r"(?m)^总共\s+\d+\s+个项目.*已加载.*$", text)
    if m_footer and m_footer.start() > 120:
        text = text[: m_footer.start()].rstrip()
    head = text[:2500]
    if "搜索邮件和人员" not in head and "Search mail and people" not in head:
        return text

    cut = -1
    for marker in (
        "\n拟办事项\n",
        "\n拟办事项",
        "\n为了保护你的隐私",
        "\nTo protect your privacy",
        "\nDate\tFrom\tSubject",
        "\n您的一次性验证码",
    ):
        i = text.rfind(marker)
        if i > cut:
            cut = i
    if cut > 200:
        tail = text[cut:].lstrip()
        if len(tail) > 150:
            return tail

    m_iter = list(re.finditer(r"(?m)^Dear\s+[A-Za-z]", text))
    if m_iter:
        last = m_iter[-1].start()
        if last > 300:
            return text[last:].lstrip()

    m2 = list(re.finditer(r"(?m)^Hi\s+[A-Za-z]", text))
    if m2:
        last = m2[-1].start()
        if last > 300:
            return text[last:].lstrip()

    return text


def _normalize_compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _body_contains_expected_subject(text: str, expected_subject: str) -> bool:
    subj = _normalize_compact_text(expected_subject)
    if len(subj) < 8:
        return False
    hay = _normalize_compact_text((text or "")[:2400])
    if not hay:
        return False
    if subj in hay:
        return True
    # 长主题在 OWA 中偶尔会被截断，允许用前缀做弱匹配
    return len(subj) >= 40 and subj[:40] in hay


def _looks_like_mixed_mail_list(text: str) -> bool:
    """识别「正文 + 多封列表项」混在一起的脏结果。"""
    if not text:
        return False
    if "Date From Subject Web Actions" in text:
        return True
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) < 6:
        return False

    date_like = 0
    sender_subject_date_triplets = 0
    via_count = 0
    for ln in lines[:220]:
        if _line_is_date_or_time_only(ln):
            date_like += 1
        if "(via LM Core)" in ln:
            via_count += 1

    for i in range(max(0, min(len(lines) - 2, 80))):
        a, b, c = lines[i], lines[i + 1], lines[i + 2]
        if (
            _line_likely_owa_sender(a)
            and not _line_is_date_or_time_only(a)
            and not _line_is_date_or_time_only(b)
            and _line_is_date_or_time_only(c)
        ):
            sender_subject_date_triplets += 1

    if sender_subject_date_triplets >= 2:
        return True
    if date_like >= 5 and via_count >= 2:
        return True
    return False


async def _row_is_selected(item_locator) -> bool:
    try:
        aria = ((await item_locator.get_attribute("aria-selected")) or "").strip().lower()
        if aria == "true":
            return True
    except Exception:
        pass
    try:
        cls = ((await item_locator.get_attribute("class")) or "").lower()
        if any(token in cls for token in ("selected", "is-selected", "isactive", "active")):
            return True
    except Exception:
        pass
    return False


async def _collect_owa_body_candidates(page, *, expected_subject: str = "") -> list[tuple[float, str]]:
    frames = list(page.frames)
    scoped_roots = [
        '[id*="ReadingPane"]',
        '[id*="readingPane"]',
        '[class*="ReadingPane"]',
        '[class*="readingPane"]',
        '[data-app-section="MessageReading"]',
        '[aria-label*="Reading Pane" i]',
        '[aria-label*="阅读窗格"]',
    ]
    body_selectors = [
        'div[aria-label*="Message body"]',
        'div[aria-label*="邮件正文"]',
        'div[aria-label*="正文"]',
        'div[aria-label*="Body"]',
        'article[role="document"]',
        'div.gs div.ii',
        'div.AllowTextSelection',
        'div[role="document"]',
        'div.rps_5055',
    ]
    fallback_body_selectors = [
        'article[role="document"]',
        'div[role="document"]',
        'div[aria-label*="Message body"]',
        'div[aria-label*="邮件正文"]',
        'div[aria-label*="正文"]',
        'div[aria-label*="Body"]',
        'div.AllowTextSelection',
        'div.rps_5055',
    ]

    candidates: list[tuple[float, str]] = []
    for f in frames:
        for root_sel in scoped_roots:
            try:
                roots = f.locator(root_sel)
                cnt = await roots.count()
            except Exception:
                continue
            for i in range(min(cnt, 3)):
                try:
                    root = roots.nth(i)
                    if not await root.is_visible(timeout=800):
                        continue
                    raw_root = await root.inner_text(timeout=2500)
                    if len(raw_root) >= 50:
                        score = _owa_body_candidate_score(raw_root) + 35.0
                        if _body_contains_expected_subject(raw_root, expected_subject):
                            score += 60.0
                        if _looks_like_mixed_mail_list(raw_root):
                            score -= 220.0
                        candidates.append((score, raw_root))
                except Exception:
                    continue

                for sel in body_selectors:
                    try:
                        loc = root.locator(sel)
                        sub_cnt = await loc.count()
                        for j in range(min(sub_cnt, 8)):
                            el = loc.nth(j)
                            if not await el.is_visible(timeout=600):
                                continue
                            raw = await el.inner_text(timeout=2500)
                            if len(raw) < 50:
                                continue
                            score = _owa_body_candidate_score(raw) + 55.0
                            if _body_contains_expected_subject(raw, expected_subject):
                                score += 75.0
                            if _looks_like_mixed_mail_list(raw):
                                score -= 240.0
                            candidates.append((score, raw))
                    except Exception:
                        continue

    for f in frames:
        for sel in fallback_body_selectors:
            try:
                loc = f.locator(sel)
                cnt = await loc.count()
                for i in range(min(cnt, 6)):
                    el = loc.nth(i)
                    if not await el.is_visible(timeout=600):
                        continue
                    raw = await el.inner_text(timeout=2500)
                    if len(raw) < 50:
                        continue
                    score = _owa_body_candidate_score(raw) + 10.0
                    if _body_contains_expected_subject(raw, expected_subject):
                        score += 50.0
                    if _looks_like_mixed_mail_list(raw):
                        score -= 220.0
                    candidates.append((score, raw))
            except Exception:
                continue

    if candidates:
        return candidates

    for f in frames:
        try:
            if await f.locator("body").count() == 0:
                continue
            raw = await f.locator("body").first.inner_text(timeout=2500)
            if len(raw) < 80 or len(raw) > 600000:
                continue
            score = _owa_body_candidate_score(raw) - 70.0
            if _body_contains_expected_subject(raw, expected_subject):
                score += 30.0
            if _looks_like_mixed_mail_list(raw):
                score -= 260.0
            candidates.append((score, raw))
        except Exception:
            continue
    return candidates


async def _best_owa_body_candidate(page, *, expected_subject: str = "") -> tuple[str, float]:
    candidates = await _collect_owa_body_candidates(page, expected_subject=expected_subject)
    if not candidates:
        return "", -1e9
    candidates.sort(key=lambda x: (x[0], -len(x[1])))
    return candidates[-1][1], candidates[-1][0]


async def _activate_mail_item(
    page,
    item_locator,
    *,
    expected_subject: str = "",
    fast: bool = False,
) -> None:
    previous_text, _ = await _best_owa_body_candidate(page)
    previous_norm = _normalize_compact_text(previous_text[:1600])

    click_targets = []
    try:
        anchor = item_locator.locator("a").first
        if await anchor.count() > 0:
            click_targets.append(anchor)
    except Exception:
        pass
    click_targets.append(item_locator)

    pre_click_ms = 70 if fast else 120
    base_settle_ms = 280 if fast else 520
    step_settle_ms = 140 if fast else 260
    enter_ms = 120 if fast else 240
    extra_settle = min(len(expected_subject or "") // 25, 420) if not fast else 0

    for attempt in range(5):
        await item_locator.scroll_into_view_if_needed()
        await page.wait_for_timeout(pre_click_ms)
        clicked = False
        use_dblclick = attempt >= 2
        for target in click_targets:
            try:
                if use_dblclick:
                    await target.dblclick(timeout=5000)
                else:
                    await target.click(timeout=5000)
                clicked = True
                break
            except Exception:
                continue
        if not clicked:
            break

        await page.wait_for_timeout(
            base_settle_ms + attempt * step_settle_ms + extra_settle
        )
        current_text, current_score = await _best_owa_body_candidate(
            page, expected_subject=expected_subject
        )
        if _reading_pane_activation_ok(current_text, expected_subject):
            return
        current_norm = _normalize_compact_text(current_text[:1600])
        # 未命中主题时仍可能已是长正文（主题只在标题栏）；高分且足够长则不再死磕
        if (
            current_text
            and not _looks_like_mixed_mail_list(current_text)
            and not _looks_like_lms_list_row_preview(current_text)
            and (current_norm != previous_norm or await _row_is_selected(item_locator))
            and current_score > 40
            and len((current_text or "").strip()) > 1100
        ):
            return
        try:
            await item_locator.press("Enter")
            await page.wait_for_timeout(enter_ms)
        except Exception:
            pass


_READING_PANE_HEADER_TIME_JS = """() => {
  const roots = document.querySelectorAll(
    '[id*="ReadingPane" i], [class*="ReadingPane" i], [data-app-section="MessageReading"], [aria-label*="Reading Pane" i], [aria-label*="阅读窗格"]'
  );
  for (const root of roots) {
    try {
      if (!root.offsetParent) continue;
    } catch (e) {
      continue;
    }
    const times = root.querySelectorAll("time[datetime]");
    for (const t of times) {
      const v = (t.getAttribute("datetime") || "").trim();
      if (v.length >= 10) return v;
    }
    const head = root.querySelector(
      '[class*="Header"], [class*="header"], [data-testid*="MessageHeader"]'
    );
    if (head) {
      const tx = (head.innerText || "").trim().slice(0, 2500);
      const lines = tx.split(/\\r?\\n/).map((l) => l.trim()).filter(Boolean);
      for (const ln of lines.slice(0, 18)) {
        if (
          /(昨天|前天|周[一二三四五六日天]|Today|Yesterday|Mon|Tue|Wed|Thu|Fri|Sat|Sun)\\b/.test(
            ln
          ) &&
          /\\d{1,2}:\\d{2}/.test(ln)
        )
          return ln;
      }
      if (lines.length) return lines.slice(0, 6).join(" | ");
    }
  }
  return "";
}"""


async def _read_reading_pane_header_raw(page) -> str:
    try:
        v = await page.evaluate(_READING_PANE_HEADER_TIME_JS)
        return (v or "").strip()
    except Exception:
        return ""


async def extract_full_body(
    page,
    item_locator,
    *,
    expected_subject: str = "",
    fast_activation: bool = False,
    extra_reading_pane_rounds: int = 0,
) -> tuple[str, str, str]:
    """
    返回 (正文, 阅读窗格头部时间展示文本, 阅读窗格时间排序键)。
    排序键可能为空；由调用方与列表解析结果合并。
    """
    pane_disp, pane_sort = "", ""
    try:
        await _activate_mail_item(
            page,
            item_locator,
            expected_subject=expected_subject,
            fast=fast_activation,
        )
        if not fast_activation:
            await page.wait_for_timeout(300)
        else:
            await page.wait_for_timeout(200)
        header_raw = await _read_reading_pane_header_raw(page)
        pane_disp, pane_sort = parse_owa_list_datetime(header_raw, datetime.now())
        # OWA 阅读窗格常虚拟渲染：先滚到底再取 inner_text，否则正文在句中被截断
        base_rp = 5 if fast_activation else 14
        extra_rp = max(0, int(extra_reading_pane_rounds or 0))
        await _owa_scroll_reading_pane_for_lazy_body(
            page, rounds=base_rp + extra_rp
        )
        body_text, _ = await _best_owa_body_candidate(page, expected_subject=expected_subject)
        if _looks_like_lms_list_row_preview(body_text) and not fast_activation:
            await page.wait_for_timeout(500)
            await _owa_scroll_reading_pane_for_lazy_body(page, rounds=12)
            body_text2, _ = await _best_owa_body_candidate(
                page, expected_subject=expected_subject
            )
            if len((body_text2 or "").strip()) > len((body_text or "").strip()) * 1.08:
                body_text = body_text2
            elif not _looks_like_lms_list_row_preview(body_text2):
                body_text = body_text2
        if not fast_activation and _text_looks_abruptly_truncated(body_text):
            await _owa_scroll_reading_pane_for_lazy_body(page, rounds=20)
            await page.wait_for_timeout(320)
            body_text2, _ = await _best_owa_body_candidate(
                page, expected_subject=expected_subject
            )
            if len((body_text2 or "").strip()) > len((body_text or "").strip()):
                body_text = body_text2
        if not body_text:
            try:
                body_text = await page.locator("body").first.inner_text(timeout=3000)
            except Exception:
                body_text = ""

        body_text = _strip_owa_list_chrome_from_body(body_text)

        from bs4 import BeautifulSoup

        soup = BeautifulSoup(body_text, "html.parser")
        for tag in soup(["script", "style", "header", "footer", "nav", "button"]):
            tag.decompose()
        clean_text = soup.get_text(separator="\n", strip=True)
        clean_text = _strip_owa_list_chrome_from_body(clean_text)

        return clean_text.strip(), pane_disp, pane_sort

    except Exception as e:
        print(f"提取正文失败: {e}")
        return f"[正文提取失败: {str(e)[:100]}]", "", ""


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
            user_instruction = "请用自然语气总结这些邮件，重点关注活动、课程作业和重要事项。"
        mode_input = input("请选择模式 (1: 日常模式[最高10], 2: 深度搜索[最高100])，默认 1：").strip()
        op_mode = "deep" if mode_input == "2" else "daily"
        
        limit_ceiling = 100 if op_mode == "deep" else 10
        email_count_raw = input(f"请输入处理邮件数量（1-{limit_ceiling}，默认 {limit_ceiling}）：").strip()
        try:
            email_count = int(email_count_raw) if email_count_raw else limit_ceiling
        except ValueError:
            email_count = limit_ceiling
        email_count = max(1, min(email_count, limit_ceiling))

        try:
            emails = await search_emails(page, search_keyword, config=config, max_emails=email_count, mode=op_mode)
        except RuntimeError as e:
            print(f"\n❌ {e}")
            return
        print(f"成功提取到 {len(emails)} 封邮件列表")
        
        print(f"正在提取 {len(emails)} 封邮件完整正文...")
        extracted_items = []
        for i, e in enumerate(emails):
            print(f"正在提取第 {i+1} 封邮件正文: {e['subject'][:30]}...")
            body, pane_disp, pane_sort = await extract_full_body(
                page,
                e["locator"],
                expected_subject=e.get("subject", ""),
                fast_activation=True,
            )
            merged_disp, merged_sort = merge_list_and_pane_datetime(
                str(e.get("date", "") or ""),
                str(e.get("date_display", "") or ""),
                pane_sort,
                pane_disp,
            )
            extracted_items.append(
                {
                    "index": i + 1,
                    "subject": e.get("subject", ""),
                    "date": merged_sort or e.get("date", ""),
                    "date_display": merged_disp or merged_sort or e.get("date", ""),
                    "sender": e.get("sender", ""),
                    "body": body,
                }
            )
            await page.wait_for_timeout(350)

        if not extracted_items:
            print("没有可分析的邮件。")
            return

        body_words = total_extracted_body_words(extracted_items)
        n = len(extracted_items)
        today = datetime.now().strftime("%Y-%m-%d")
        loop = asyncio.get_running_loop()
        total_batches = (n + LLM_PARALLEL_BATCH_SIZE - 1) // LLM_PARALLEL_BATCH_SIZE

        async def run_llm_task(prompt: str) -> str:
            return await loop.run_in_executor(None, call_llm, prompt, config)

        print(
            f"邮件正文总词数（仅 body）: {body_words}；"
            f"每批并发 {LLM_PARALLEL_BATCH_SIZE} 次，共 {total_batches} 批单封 LLM，再 1 次汇总。"
        )

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
                    instruction=user_instruction,
                    email_human_text=human,
                    email_index=i + 1,
                    email_total=n,
                )
            )

        normalized = []
        for batch_index, start in enumerate(range(0, n, LLM_PARALLEL_BATCH_SIZE), start=1):
            batch_prompts = prompts[start:start + LLM_PARALLEL_BATCH_SIZE]
            print(
                f"\n并行调用 LLM 分析第 {batch_index}/{total_batches} 批"
                f"（{len(batch_prompts)} 封邮件）…"
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
            instruction=user_instruction,
            per_email_sections=per_email_sections,
            email_count=n,
        )
        print("\n正在调用 LLM 生成最终汇总…")
        try:
            summary = await run_llm_task(final_prompt)
        except Exception as exc:
            summary = f"最终汇总失败：{exc}"

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