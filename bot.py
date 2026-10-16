import os
import re
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import html as html_lib
import asyncio
import calendar
import itertools
import nest_asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from PIL import Image
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)
from playwright.async_api import async_playwright
from google import genai
from google.genai import types

nest_asyncio.apply()

# =====================================================================================
# Settings
# =====================================================================================
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
GEMINI_KEY = os.environ["GEMINI_KEY"]

# Models are tried in this order, ONE attempt each
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_FALLBACKS = [
    m.strip()
    for m in os.environ.get("GEMINI_FALLBACKS", "gemini-3.5-flash-lite,gemini-3.1-pro").split(",")
    if m.strip()
]

# Login (Okta): accounts live in credentials.json (or CREDENTIALS_JSON env var on Render)
CREDENTIALS_FILE = os.environ.get("CREDENTIALS_FILE", "credentials.json")
if os.environ.get("CREDENTIALS_JSON") and not os.path.exists(CREDENTIALS_FILE):
    with open(CREDENTIALS_FILE, "w", encoding="utf-8") as _f:
        _f.write(os.environ["CREDENTIALS_JSON"])

STATE_FILE = "session_state.json"      # browser session saved after each successful run
USER_REPLY_TIMEOUT_SEC = 300           # how long to wait for a button press / reply

# Time zone used by the date picker in Telegram ("Today", "Now", the calendar).
CHART_TZ = os.environ.get("CHART_TZ", "Africa/Cairo")

# Report parameters: exact order and names used in the WhatsApp report
PARAM_ORDER = ["PIP", "PD", "Motor Amp", "Tm"]
PARAM_FULL = {
    "PIP": "Pump Intake Pressure / Press-Pump Intake",
    "PD": "Press-Pump Discharge",
    "Motor Amp": "Pwr-Motor Amps Ph B",
    "Tm": "Temp-Motor",
}

# Grid View > Facility Parameters: the rows to read (exact names as shown in the grid)
GRID_PARAMS = [
    "Status-Motor",
    "Status-Hz",
    "Shutdown-Current SD Reason",
    "Shutdown-Last Shutdown Reason",
    "Press-Pump Intake (Pi)",
    "Press-Pump Discharge",
    "Status-Pump Delta Pressure",
    "Temp-Motor",
    "Temp-Pump Intake",
    "Vib-Pump X axis",
    "Vib-Pump Y axis",
    "Pwr-Motor Amps Ph B",
]

# max seconds per model attempt (keeps total AI time well under 2 minutes)
GEMINI_MODEL_TIMEOUT_SEC = 60

# Browser size (a narrower tab gives a narrower chart)
VIEWPORT_W = 1400
VIEWPORT_H = 732

# Chart points: two clicks on the Highcharts plot area (start and end of the lines)
PLACE_POINTS = True
POINT_WAIT_SEC = 3                 # wait after each click
POINT_Y_FRAC = 0.70                # fallback vertical position (0 = top, 1 = bottom)
START_CLICK_FROM_LEFT_PX = 10      # first click: 10 px inside the plot's left edge
END_CLICK_FROM_RIGHT_PX = 4        # second click: 4 px inside the plot's right edge
CROP_RIGHT_PX = 35

# Custom date range: waits (seconds) so the page can keep up
CUSTOM_OPEN_ATTEMPTS = 5           # tries to open the date picker
CUSTOM_AFTER_OPEN_SEC = 5          # after clicking the date picker icon
CUSTOM_BEFORE_START_SEC = 3        # before entering the start date and time
CUSTOM_BEFORE_END_SEC = 3          # before entering the end date and time
CUSTOM_CHAR_DELAY_SEC = 0.2        # pause between typed characters
CUSTOM_AFTER_TYPE_SEC = 1.5        # after typing a value, before checking it
CUSTOM_AFTER_SET_SEC = 12          # after clicking "Set"

# Page selectors
TRENDS_TAB = "a[onclick*='showTrends']"
GRID_TAB = "a[onclick*='showTabularView']"
FACILITY_TAB = 'a[onclick*="getTableData"][onclick*="Facility Parameter"]'
CHART_SEL = 'div[data-dojo-type="ProductionChart"]'
POINT_TARGET_SELECTOR = 'rect.highcharts-background'
PD_TAG_SEL = 'div.tagContainer[name="Press-Pump Discharge"], div.tagContainer[channeldisplay="Press-Pump Discharge"]'

SITE_URL = "https://productionlink.bakerhughes.com/"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

ai_client = genai.Client(api_key=GEMINI_KEY)

CHOICES = {}          # button-question id -> Future (resolved by a button press)
TEXT_WAITERS = {}     # chat id -> (Future, compiled regex, delete the reply?)
_req_counter = itertools.count(1)


# =====================================================================================
# Session Control & Asking the user (buttons / text)
# =====================================================================================
NOOP = "__noop__"
END_SESSION = "__end_session__"


class SessionEndedException(Exception):
    """Raised whenever the user chooses to end/stop the current session."""
    pass


async def ask_grid(bot, chat_id, text, rows, msg=None, timeout=USER_REPLY_TIMEOUT_SEC, allow_end=True):
    """Show a keyboard made of `rows` [[(label, value), ...], ...] and wait for a press.
    Always appends a '🛑 End Session' button unless allow_end=False."""
    req = str(next(_req_counter))
    rows_to_render = list(rows)
    if allow_end:
        has_end = any(val == END_SESSION for r in rows_to_render for _, val in r)
        if not has_end:
            rows_to_render.append([("🛑 End Session", END_SESSION)])

    flat, kb_rows = [], []
    for row in rows_to_render:
        kb_row = []
        for label, value in row:
            kb_row.append(InlineKeyboardButton(label, callback_data=f"c|{req}|{len(flat)}"))
            flat.append(value)
        kb_rows.append(kb_row)
    kb = InlineKeyboardMarkup(kb_rows)

    if msg is None:
        msg = await bot.send_message(chat_id, text, reply_markup=kb)
    else:
        try:
            res = await msg.edit_text(text, reply_markup=kb)
            if hasattr(res, "edit_text"):
                msg = res
        except Exception:
            pass

    loop = asyncio.get_running_loop()
    try:
        while True:
            fut = loop.create_future()
            CHOICES[req] = fut
            try:
                idx = await asyncio.wait_for(fut, timeout)
            except asyncio.TimeoutError:
                raise Exception("Timed out waiting for your answer.")
            chosen = flat[idx]
            if chosen == END_SESSION:
                try:
                    await msg.edit_text(f"{text}\n\n🛑 Ending session...")
                except Exception:
                    pass
                raise SessionEndedException("Session ended by user.")
            if chosen != NOOP:
                return chosen, msg
    finally:
        CHOICES.pop(req, None)


async def ask_choice(bot, chat_id, text, options, allow_end=True):
    """One button per option [(label, value), ...]; the message then shows your choice."""
    (label, value), msg = await ask_grid(
        bot, chat_id, text, [[(lab, (lab, val))] for lab, val in options], allow_end=allow_end
    )
    try:
        await msg.edit_text(f"{text}\n\n➡️ {label}")
    except Exception:
        pass
    return value


async def _safe_answer(q, text=None):
    """Answer a callback query safely, ignoring Telegram's 'Query is too old' error."""
    try:
        await q.answer(text)
    except Exception:
        pass


async def on_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try:
        _, req, idx = q.data.split("|")
    except Exception:
        await _safe_answer(q)
        return

    if req.startswith("endtxt_"):
        chat_id = int(req.split("_", 1)[1])
        waiter = TEXT_WAITERS.get(chat_id)
        if waiter:
            fut, _, _ = waiter
            if not fut.done():
                fut.set_exception(SessionEndedException("Session ended by user."))
        await _safe_answer(q, "Ending session...")
        return

    fut = CHOICES.get(req)
    if fut and not fut.done():
        fut.set_result(int(idx))
        await _safe_answer(q)
    else:
        await _safe_answer(q, "This question has expired.")


async def ask_text(bot, chat_id, prompt, pattern, delete=False, timeout=USER_REPLY_TIMEOUT_SEC):
    """Send `prompt` (with an inline '🛑 End Session' button) and wait for the next message matching `pattern`."""
    fut = asyncio.get_running_loop().create_future()
    TEXT_WAITERS[chat_id] = (fut, re.compile(pattern), delete)
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("🛑 End Session", callback_data=f"c|endtxt_{chat_id}|0")]])
    prompt_msg = await bot.send_message(chat_id, prompt, reply_markup=kb)
    try:
        return await asyncio.wait_for(fut, timeout)
    except asyncio.TimeoutError:
        raise Exception("Timed out waiting for your reply.")
    finally:
        TEXT_WAITERS.pop(chat_id, None)
        try:
            await prompt_msg.edit_reply_markup(reply_markup=None)
        except Exception:
            pass


# =====================================================================================
# Pickers: view, period (presets / custom range), PD visibility, parameters to mark
# =====================================================================================
PRESETS = ["1 D", "3 D", "1 W", "2 W", "1 M", "2 M", "1 Y"]


def now_local():
    return datetime.now(ZoneInfo(CHART_TZ)).replace(second=0, microsecond=0, tzinfo=None)


def fmt_dt(dt):
    return dt.strftime("%d %b %Y %H:%M")


async def pick_date(bot, chat_id, title, msg):
    """Inline calendar. Returns (date, msg)."""
    today = now_local().date()
    year, month = today.year, today.month
    while True:
        rows = [
            [("◀", ("nav", -1)), (f"{calendar.month_abbr[month]} {year}", NOOP), ("▶", ("nav", 1))],
            [(d, NOOP) for d in ("Mo", "Tu", "We", "Th", "Fr", "Sa", "Su")],
        ]
        for week in calendar.monthcalendar(year, month):
            rows.append([(str(d) if d else " ", ("day", d) if d else NOOP) for d in week])
        rows.append([("Today", ("date", today)), ("Yesterday", ("date", today - timedelta(days=1)))])
        rows.append([("❌ Cancel", ("cancel", None))])

        (kind, val), msg = await ask_grid(bot, chat_id, f"📅 {title}\nPick a date:", rows, msg)
        if kind == "nav":
            month += val
            if month == 0:
                month, year = 12, year - 1
            elif month == 13:
                month, year = 1, year + 1
        elif kind == "day":
            return date(year, month, val), msg
        elif kind == "date":
            return val, msg
        else:
            raise Exception("Cancelled by you.")


async def pick_time(bot, chat_id, title, msg, allow_now=False):
    """Quick times, or hour then minute grids. Returns (hour, minute, msg)."""
    quick = [
        [("00:00", (0, 0)), ("06:00", (6, 0)), ("12:00", (12, 0))],
        [("18:00", (18, 0)), ("23:59", (23, 59)), ("🕐 Other time", "other")],
    ]
    if allow_now:
        quick.insert(0, [("⏱ Now", "now")])
    quick.append([("❌ Cancel", "cancel")])

    value, msg = await ask_grid(bot, chat_id, f"🕐 {title}\nPick a time:", quick, msg)
    if value == "cancel":
        raise Exception("Cancelled by you.")
    if value == "now":
        n = now_local()
        return n.hour, n.minute, msg
    if value != "other":
        return value[0], value[1], msg

    hour_rows = [[(f"{h:02d}", ("h", h)) for h in range(r, r + 6)] for r in range(0, 24, 6)]
    (_, hour), msg = await ask_grid(bot, chat_id, f"🕐 {title}\nPick the hour:", hour_rows, msg)

    minute_rows = [[(f"{m:02d}", ("m", m)) for m in range(r, r + 20, 5)] for r in (0, 20, 40)]
    minute_rows.append([("✍️ Exact minute", ("type", None))])
    (kind, minute), msg = await ask_grid(
        bot, chat_id, f"🕐 {title}\nHour {hour:02d}: pick the minute:", minute_rows, msg
    )
    if kind == "type":
        minute = int(await ask_text(bot, chat_id, "✍️ Send the exact minute (0-59):", r"^[0-5]?\d$", delete=True))
    return hour, minute, msg


async def pick_period(bot, chat_id, well):
    """Which period to show on the chart:
    {"type": "default"} | {"type": "preset", "label": "1 W"} | {"type": "custom", "start": dt, "end": dt}"""
    rows = [
        [("⏩ Keep the chart's current view", ("default", None))],
        [(p, ("preset", p)) for p in PRESETS[:4]],
        [(p, ("preset", p)) for p in PRESETS[4:]],
        [("📅 Custom range", ("custom", None))],
    ]
    (kind, val), msg = await ask_grid(bot, chat_id, f"🗓 Which period for {well}?", rows)

    if kind == "default":
        await msg.edit_text("🗓 Period: the chart's current view")
        return {"type": "default"}
    if kind == "preset":
        await msg.edit_text(f"🗓 Period: {val.replace(' ', '')} (preset)")
        return {"type": "preset", "label": val}

    while True:
        d1, msg = await pick_date(bot, chat_id, "START date", msg)
        h, m, msg = await pick_time(bot, chat_id, f"START time ({d1:%d %b %Y})", msg)
        start = datetime(d1.year, d1.month, d1.day, h, m)

        d2, msg = await pick_date(bot, chat_id, "END date", msg)
        h, m, msg = await pick_time(bot, chat_id, f"END time ({d2:%d %b %Y})", msg, allow_now=True)
        end = datetime(d2.year, d2.month, d2.day, h, m)

        if end <= start:
            await bot.send_message(chat_id, "⚠️ The end must be after the start. Let's pick again.")
            continue

        answer, msg = await ask_grid(
            bot,
            chat_id,
            f"🗓 Custom range for {well}:\nStart: {fmt_dt(start)}\nEnd:   {fmt_dt(end)}\n\nUse it?",
            [[("✅ Use this range", "ok")], [("🔁 Pick again", "again"), ("❌ Cancel", "cancel")]],
            msg,
        )
        if answer == "ok":
            break
        if answer == "cancel":
            raise Exception("Cancelled by you.")

    await msg.edit_text(f"🗓 Period: {fmt_dt(start)} → {fmt_dt(end)}")
    return {"type": "custom", "start": start, "end": end}


async def ask_remove_pd(bot, chat_id, well):
    """Ask the user whether to keep or remove PD (Press-Pump Discharge) from the chart."""
    return await ask_choice(
        bot,
        chat_id,
        f"📉 Do you want to keep or remove PD (Press-Pump Discharge) on the chart for {well}?",
        [
            ("✅ Keep PD on the chart", False),
            ("🚫 Remove PD from the chart", True),
        ],
    )


async def pick_view(bot, chat_id, well):
    """Trends chart (with analysis) or the Grid View > Facility Parameters table."""
    value = await ask_choice(
        bot,
        chat_id,
        f"What do you need for {well}?",
        [("📈 Trends chart + analysis", "trends"), ("🧾 Grid View (Facility Parameters)", "grid")],
    )
    return value


async def pick_single_param_to_mark(bot, chat_id, available):
    """Single-select of a grid parameter (only 1 parameter can be checked). Returns name or None."""
    rows = [[(f"⬜ {n}", ("pick", i))] for i, n in enumerate(available)]
    rows.append([("🛑 End Session", END_SESSION)])

    (kind, idx), msg = await ask_grid(
        bot,
        chat_id,
        "☑️ Select ONE parameter to mark (tick) in the grid:\n(All other parameters will be unchecked first)",
        rows,
        allow_end=False,
    )
    if kind == "skip" or idx is None:
        try:
            await msg.edit_text("⏭ Marking skipped")
        except Exception:
            pass
        return None

    chosen = available[idx]
    try:
        await msg.edit_text(f"☑️ Selected parameter: {chosen}")
    except Exception:
        pass
    return chosen


# =====================================================================================
# Login (Okta)
# =====================================================================================
def load_accounts():
    if not os.path.exists(CREDENTIALS_FILE):
        raise Exception(
            f"{CREDENTIALS_FILE} not found. Set CREDENTIALS_JSON in Render Environment Variables: "
            '{"accounts": [{"name": "...", "email": "...", "password": "..."}]}'
        )
    with open(CREDENTIALS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    accounts = data.get("accounts", []) if isinstance(data, dict) else data
    accounts = [a for a in accounts if a.get("email") and a.get("password")]
    if not accounts:
        raise Exception(f"{CREDENTIALS_FILE} has no account with both 'email' and 'password'.")
    return accounts


class LoginUI:
    """Everything the login flow needs to ask you in Telegram."""

    def __init__(self, bot, chat_id):
        self.bot = bot
        self.chat_id = chat_id

    async def choose_account(self):
        accounts = load_accounts()
        if len(accounts) == 1:
            return accounts[0]
        options = [(f"{a.get('name') or 'Account'} — {a['email']}", a) for a in accounts]
        return await ask_choice(self.bot, self.chat_id, "🔐 Login needed. Which email should I use?", options)

    async def confirm_mfa_email(self, shown_text):
        return await ask_choice(
            self.bot,
            self.chat_id,
            f"📧 Verification step:\n{shown_text}\n\nShould I send the code to this email?",
            [("✅ Yes, send the code", True), ("❌ No, use another email", False)],
        )

    async def ask_code(self):
        return await ask_text(
            self.bot,
            self.chat_id,
            "🔢 Send me the 6-digit code from your email now (or send 'cancel').",
            r"^\d{6}$",
            delete=True,
        )


HEADING = 'h2[data-se="o-form-head"]'
EMAIL_INPUT = "#idp-discovery-username"
EMAIL_NEXT = "#idp-discovery-submit"
REMEMBER_INPUT = 'input[name="remember"]'
REMEMBER_LABEL = 'label[data-se-for-name="remember"]'
PASS_USER = "#okta-signin-username"
PASS_INPUT = "#okta-signin-password"
PASS_SUBMIT = "#okta-signin-submit"
MFA_TEXT = ".mfa-send-email-content"
MFA_SEND = 'input[type="submit"][value="Send me the code"]'
CODE_INPUT = 'input[name="answer"]'
DEVICE_INPUT = 'input[name="rememberDevice"]'
DEVICE_LABEL = 'label[data-se-for-name="rememberDevice"]'
CODE_VERIFY = 'input[type="submit"][value="Verify"]'
BACK_LINK = 'a[data-se="signout-link"]'
ERROR_BOX = '[data-se="o-form-error-container"]'


async def _visible(page, selector):
    try:
        return await page.locator(selector).first.is_visible()
    except Exception:
        return False


async def login_state(page):
    """Which Okta step is on screen right now (None if no known step)."""
    if await _visible(page, PASS_INPUT):
        return "password"
    if await _visible(page, EMAIL_INPUT):
        return "email"
    if await _visible(page, CODE_INPUT):
        return "mfa_code"
    if await _visible(page, MFA_SEND) or await _visible(page, MFA_TEXT):
        return "mfa_send"
    return None


async def logged_in(page):
    """True once the ProductionLink search icon exists in any frame."""
    for fr in page.frames:
        try:
            if await fr.query_selector("#assetSearch"):
                return True
        except Exception:
            pass
    return False


async def error_text(page):
    try:
        t = await page.locator(ERROR_BOX).first.inner_text(timeout=1000)
        return re.sub(r"\s+", " ", t or "").strip()
    except Exception:
        return ""


async def ensure_checked(page, input_sel, label_sel):
    """Make sure a (custom) checkbox is ticked: click its label, fall back to a JS click."""
    box = page.locator(input_sel).first
    try:
        if await page.locator(input_sel).count() == 0 or await box.is_checked():
            return
    except Exception:
        return
    try:
        await page.locator(label_sel).first.click(timeout=3000)
    except Exception:
        try:
            await box.evaluate("e => e.click()")
        except Exception:
            pass


async def click_back(page):
    try:
        await page.locator(BACK_LINK).first.click(timeout=5000)
    except Exception:
        pass
    await asyncio.sleep(2)


async def okta_login(page, ui, notify):
    account = None
    attempts = {"password": 0, "code": 0}

    for _ in range(150):
        if await logged_in(page):
            await notify("✅ Login completed")
            return

        state = await login_state(page)
        if state is None:
            await asyncio.sleep(1.5)
            continue

        if account is None:
            account = await ui.choose_account()
            await notify(f"👤 Using account: {account['email']}")
        email = account["email"]

        if state == "email":
            await notify("✉️ Entering the email...")
            await page.fill(EMAIL_INPUT, email)
            await ensure_checked(page, REMEMBER_INPUT, REMEMBER_LABEL)
            await page.locator(EMAIL_NEXT).first.click()
            await asyncio.sleep(3)

        elif state == "password":
            attempts["password"] += 1
            if attempts["password"] > 3:
                raise Exception("The password step came back several times; stopped to avoid locking the account.")

            if await page.locator(PASS_USER).count():
                user_loc = page.locator(PASS_USER).first
                if (await user_loc.input_value()).strip().lower() != email.lower():
                    if await user_loc.is_enabled():
                        await user_loc.fill(email)
                    else:
                        await notify("↩️ The page shows another email, going back to sign in...")
                        await click_back(page)
                        continue

            await notify("🔑 Entering the password...")
            await page.fill(PASS_INPUT, account["password"])
            await ensure_checked(page, REMEMBER_INPUT, REMEMBER_LABEL)
            await page.locator(PASS_SUBMIT).first.click()
            await asyncio.sleep(3)
            err = await error_text(page)
            if err:
                raise Exception(f"Login failed: {err}")

        elif state == "mfa_send":
            try:
                shown = re.sub(r"\s+", " ", await page.locator(MFA_TEXT).first.inner_text(timeout=3000)).strip()
            except Exception:
                shown = "(could not read the masked email from the page)"
            if await ui.confirm_mfa_email(shown):
                await notify("📨 Requesting the verification code...")
                await page.locator(MFA_SEND).first.click()
                await asyncio.sleep(3)
            else:
                await notify("↩️ Going back to choose another email...")
                await click_back(page)
                account = await ui.choose_account()
                await notify(f"👤 Using account: {account['email']}")

        elif state == "mfa_code":
            attempts["code"] += 1
            if attempts["code"] > 3:
                raise Exception("The verification code was rejected 3 times; stopped.")
            await page.fill(CODE_INPUT, await ui.ask_code())
            await ensure_checked(page, DEVICE_INPUT, DEVICE_LABEL)
            await page.locator(CODE_VERIFY).first.click()
            await asyncio.sleep(3)
            err = await error_text(page)
            if err:
                await notify(f"⚠️ The code was not accepted: {err}")

    try:
        heading = (await page.locator(HEADING).first.inner_text(timeout=2000)).strip()
    except Exception:
        heading = ""
    raise Exception(f"Login did not complete in time (page heading: '{heading}').")


async def ensure_logged_in(page, ui, notify):
    """Either already logged in, or on the Okta page (then run the login flow)."""
    await notify("🔐 Checking login status...")
    for _ in range(40):
        if await logged_in(page):
            await notify("✅ Already logged in")
            return
        if await login_state(page) is not None or await _visible(page, HEADING):
            await notify("🔐 Login page detected, signing in...")
            await okta_login(page, ui, notify)
            return
        await asyncio.sleep(2)


# =====================================================================================
# Page helpers
# =====================================================================================
async def find_in_frames(page, selector, timeout=60, need_size=False):
    """Poll every frame (including nested ones) until `selector` exists. Returns (frame, element)."""
    for _ in range(int(timeout)):
        for fr in page.frames:
            try:
                el = await fr.query_selector(selector)
                if el and (
                    not need_size
                    or await el.evaluate(
                        "e => { const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0; }"
                    )
                ):
                    return fr, el
            except Exception:
                pass
        await asyncio.sleep(1)
    return None, None


def frames_report(page):
    lines = [f"page.url = {page.url}"]
    for i, fr in enumerate(page.frames):
        lines.append(f"[{i}] name='{fr.name}' url={fr.url[:120]}")
    return "\n".join(lines)


async def click(el):
    """Real click, with a JS click as fallback."""
    try:
        await el.click(timeout=3000)
    except Exception:
        await el.evaluate("e => e.click()")


async def js_click(el):
    await el.evaluate("e => e.click()")


async def click_well_result(page, search_term):
    """Click an element whose whole text equals the well name (case-insensitive)."""
    pattern = re.compile(rf"^\s*{re.escape(search_term)}\s*$", re.I)
    for fr in page.frames:
        try:
            loc = fr.get_by_text(pattern)
            for i in range(await loc.count()):
                if await loc.nth(i).is_visible():
                    await click(loc.nth(i))
                    await asyncio.sleep(1)
                    return f"frame '{fr.name}' match #{i}"
        except Exception:
            continue
    return None


async def dump_candidates(page, needle):
    """List elements whose text contains the well name (for error messages)."""
    out = []
    for fr in page.frames:
        try:
            rows = await fr.evaluate(
                """(needle) => {
                    const res = [];
                    const n = needle.toLowerCase();
                    for (const el of document.querySelectorAll('body *')) {
                        const t = (el.innerText || '').trim();
                        if (t && t.length < 80 && t.toLowerCase().includes(n) && el.children.length < 3) {
                            const r = el.getBoundingClientRect();
                            res.push(`<${el.tagName.toLowerCase()} id='${el.id}'> '${t}' visible=${r.width>0&&r.height>0}`);
                            if (res.length >= 8) break;
                        }
                    }
                    return res;
                }""",
                needle,
            )
            out += [f"[frame '{fr.name}'] {r}" for r in rows]
        except Exception:
            pass
    return "\n".join(out) if out else "(none found)"


ACTIVE_JS = """el => {
    const re = /(^|[\\s_-])(active|selected|current|on)([\\s_-]|$)/i;
    let n = el;
    for (let i = 0; i < 3 && n; i++, n = n.parentElement) {
        if (re.test(n.className || '')) return true;
        if (n.getAttribute && n.getAttribute('aria-selected') === 'true') return true;
    }
    return false;
}"""


async def ensure_tab_open(page, selector, name, notify, timeout=40, required=True):
    """Make sure the tab/link matching `selector` is the active one; click it if it isn't."""
    _, tab = await find_in_frames(page, selector, timeout=timeout)
    if not tab:
        if required:
            raise Exception(f"The '{name}' tab was not found.\n" + frames_report(page))
        await notify(f"ℹ️ {name} link not found; continuing (it may already be open)")
        return
    if await tab.evaluate(ACTIVE_JS):
        await notify(f"✅ {name} is already open")
        return
    await notify(f"📑 Opening {name}...")
    await click(tab)
    await asyncio.sleep(4)
    await notify(f"✅ {name} opened")


async def open_well(well_name, page, notify, login_ui):
    """Open the site, log in if needed, search the well and select it."""
    await notify("🌐 Opening ProductionLink...")
    await page.goto(SITE_URL, wait_until="domcontentloaded")
    await ensure_logged_in(page, login_ui, notify)

    frame, search_icon = await find_in_frames(page, "#assetSearch", timeout=90)
    if not search_icon:
        raise Exception("Search icon (#assetSearch) not found in any frame.\n" + frames_report(page))
    await notify("🔍 Search icon found")

    clear_icon = await frame.query_selector("span.deleteiconAsset span")
    if clear_icon:
        try:
            await js_click(clear_icon)
            await asyncio.sleep(1)
        except Exception:
            pass

    await js_click(search_icon)
    await asyncio.sleep(1)
    search_input = await frame.query_selector("#assetSearchText")
    if not search_input:
        _, search_input = await find_in_frames(page, "#assetSearchText", timeout=15)
    if not search_input:
        raise Exception("Search input (#assetSearchText) not found.\n" + frames_report(page))

    upper_name = well_name.upper().strip()
    prefix = "HPUMP_" if ("SHAHD" in upper_name or "ZAHRAA" in upper_name) else "ESP_"
    search_term = f"{prefix}{upper_name}"

    await notify(f"⌨️ Typing well name: {search_term}")
    await search_input.evaluate("el => { el.value = ''; el.focus(); }")
    await page.keyboard.type(search_term, delay=80)
    await asyncio.sleep(3)

    clicked = await click_well_result(page, search_term)
    if not clicked:
        await page.keyboard.press("Enter")
        await asyncio.sleep(3)
        clicked = await click_well_result(page, search_term)
    await notify("🖱️ Well selected from the search results" if clicked else "↩️ Pressed Enter to search")
    return search_term, upper_name, clicked


# =====================================================================================
# Trends: chart well title, period estimation (in hours / hours and half), PD toggle
# =====================================================================================
async def get_chart_well_name(page, fallback=""):
    """Read the well name from <div id='Gauge_...chartHeader_Title' class='chart-header-title'>FAJR-48</div>."""
    for fr in page.frames:
        try:
            el = await fr.query_selector('div.chart-header-title, div[id$="chartHeader_Title"]')
            if el:
                t = re.sub(r"\s+", " ", (await el.inner_text()) or "").strip()
                if t:
                    return re.sub(r"^(?:ESP_|HPUMP_)", "", t, flags=re.I).strip()
        except Exception:
            pass
    return re.sub(r"^(?:ESP_|HPUMP_)", "", (fallback or "").strip(), flags=re.I)


async def get_range_text(page):
    """The 'Oct 12, 2026 00:23 - Oct 13, 2026 19:00' text shown above the chart."""
    for fr in page.frames:
        try:
            el = await fr.query_selector("div.data-range")
            if el:
                t = re.sub(r"\s+", " ", await el.inner_text()).strip()
                if t:
                    return t
        except Exception:
            pass
    return ""


def _parse_any_dt(s: str):
    """Parse date/time strings like 'Oct 14, 2026 18:00' or '10/14/2026 18:00:00'."""
    s = re.sub(r"\s+", " ", (s or "").strip())
    for fmt in (
        "%b %d, %Y %H:%M:%S",
        "%b %d, %Y %H:%M",
        "%B %d, %Y %H:%M:%S",
        "%B %d, %Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%d %b %Y %H:%M",
    ):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _format_hours_half(total_hours: float) -> str:
    """Round `total_hours` to the nearest 0.5 hour and format as 'Last X Hours' or 'Last X.5 Hours'."""
    half_steps = max(1, round(total_hours * 2))
    val = half_steps / 2.0

    if val.is_integer():
        whole = int(val)
        unit = "Hour" if whole == 1 else "Hours"
        return f"Last {whole} {unit}"
    else:
        return f"Last {val:.1f} Hours"


def compute_period_label(period: dict, range_text: str = "") -> str:
    """Estimate the chart period in hours (rounded to 0.5h) and return 'Last X Hours' / 'Last X.5 Hours'."""
    if range_text:
        clean_rng = re.sub(r"\(.*?\)", "", range_text).strip()
        parts = [p.strip() for p in re.split(r"\s+-\s+|\s+to\s+", clean_rng) if p.strip()]
        if len(parts) == 2:
            dt1, dt2 = _parse_any_dt(parts[0]), _parse_any_dt(parts[1])
            if dt1 and dt2 and dt2 > dt1:
                hours = (dt2 - dt1).total_seconds() / 3600.0
                return _format_hours_half(hours)

    if period and period.get("type") == "custom" and period.get("start") and period.get("end"):
        hours = (period["end"] - period["start"]).total_seconds() / 3600.0
        if hours > 0:
            return _format_hours_half(hours)

    preset_hours = {
        "1D": 24,
        "3D": 72,
        "1W": 168,
        "2W": 336,
        "1M": 720,
        "2M": 1440,
        "1Y": 8760,
    }
    if period and period.get("type") == "preset":
        key = re.sub(r"\s+", "", period.get("label", "")).upper()
        if key in preset_hours:
            return _format_hours_half(preset_hours[key])

    return "Last 24 Hours"


async def set_pd_visibility_on_chart(page, remove_pd: bool, notify):
    """Ensure Press-Pump Discharge (PD) on the chart is hidden (if remove_pd=True) or visible (if remove_pd=False)."""
    _, tag_el = await find_in_frames(page, PD_TAG_SEL, timeout=12, need_size=True)
    if not tag_el:
        if remove_pd:
            await notify("⚠️ Press-Pump Discharge tag container not found on chart.")
        return

    async def is_pd_currently_hidden():
        return await tag_el.evaluate("""(el) => {
            const valSpan = el.querySelector('.tagContainerValue');
            const lineEl = el.querySelector('svg line');
            const valColor = valSpan ? (valSpan.style.color || getComputedStyle(valSpan).color || '') : '';
            const strokeColor = lineEl ? (lineEl.style.stroke || lineEl.getAttribute('stroke') || '') : '';
            const combined = (valColor + ' ' + strokeColor).replace(/\\s+/g, '').toLowerCase();
            return combined.includes('204,204,204') || combined.includes('#ccc') || combined.includes('#cccccc');
        }""")

    currently_hidden = await is_pd_currently_hidden()

    if remove_pd and not currently_hidden:
        await notify("📉 Clicking 'Press-Pump Discharge' tag to remove PD from the chart...")
        await click(tag_el)
        await asyncio.sleep(2.5)
        if await is_pd_currently_hidden():
            await notify("✅ PD (Press-Pump Discharge) removed from the chart.")
        else:
            await js_click(tag_el)
            await asyncio.sleep(2.0)
            await notify("✅ PD (Press-Pump Discharge) toggled off.")
    elif remove_pd and currently_hidden:
        await notify("✅ PD (Press-Pump Discharge) is already removed from the chart.")
    elif not remove_pd and currently_hidden:
        await notify("📈 Clicking 'Press-Pump Discharge' tag to restore PD on the chart...")
        await click(tag_el)
        await asyncio.sleep(2.5)
        await notify("✅ PD (Press-Pump Discharge) restored on the chart.")


async def find_preset_button(page, label, timeout=20):
    """The span.defaultFilter whose text is the label ('1 D', '3 D', ...; the page uses &nbsp;)."""
    want = re.sub(r"\s+", "", label)
    for _ in range(timeout):
        for fr in page.frames:
            try:
                for el in await fr.query_selector_all("span.defaultFilter"):
                    if re.sub(r"\s+", "", await el.inner_text()) == want:
                        return el
            except Exception:
                pass
        await asyncio.sleep(1)
    return None


async def open_date_picker(page, notify):
    """Click the date picker icon until the start-date box shows up. Returns that box."""
    for attempt in range(1, CUSTOM_OPEN_ATTEMPTS + 1):
        _, icon = await find_in_frames(page, "div.custom-duration-cal.trendsPlotDatePicker", timeout=4, need_size=True)
        if icon:
            await notify("📅 Opening the custom date picker...")
            await click(icon)
            await asyncio.sleep(CUSTOM_AFTER_OPEN_SEC)
            _, left = await find_in_frames(page, 'input[id$="_DateRangePickerLeftInput"]', timeout=4, need_size=True)
            if left:
                return left
        await notify(f"🔁 Date picker not open yet (attempt {attempt}/{CUSTOM_OPEN_ATTEMPTS})")
    raise Exception("failed to open the date picker")


async def type_date_cleanly(page, el, digits, label, notify):
    """Clear the box (JS click, empty value, Ctrl+A, Delete) and type the digits one by one."""
    for attempt in range(1, 4):
        await el.evaluate("e => { e.click(); e.focus(); }")
        await asyncio.sleep(1)
        await el.evaluate("e => { e.value = ''; }")
        await page.keyboard.press("Control+A")
        await page.keyboard.press("Delete")
        await asyncio.sleep(1)
        for ch in digits:
            await page.keyboard.type(ch)
            await asyncio.sleep(CUSTOM_CHAR_DELAY_SEC)
        await asyncio.sleep(CUSTOM_AFTER_TYPE_SEC)
        shown = await el.input_value()

        shown_digits = re.sub(r"\D", "", shown)
        if shown_digits.startswith(digits):
            await notify(f"✅ {label} entered: {shown}")
            return
        await notify(f"⚠️ {label} shows '{shown}' (expected to start with {digits}), retry {attempt}/3")
    await notify(f"⚠️ {label} may be wrong; continuing anyway")


async def set_chart_dates(page, start, end, notify):
    """Custom range. The boxes take MMDDYYYYHHMM (seconds removed)."""
    start_s = start.strftime("%m%d%Y%H%M")
    end_s = end.strftime("%m%d%Y%H%M")

    left = await open_date_picker(page, notify)

    await asyncio.sleep(CUSTOM_BEFORE_START_SEC)
    await type_date_cleanly(page, left, start_s, "Start", notify)

    await asyncio.sleep(CUSTOM_BEFORE_END_SEC)
    _, right = await find_in_frames(page, 'input[id$="_DateRangePickerRightInput"]', timeout=10, need_size=True)
    if not right:
        raise Exception("the end date box was not found")
    await type_date_cleanly(page, right, end_s, "End", notify)

    _, set_btn = await find_in_frames(page, 'input[id$="_DateRangePickerSet"][value="Set"]', timeout=10, need_size=True)
    if not set_btn:
        raise Exception("the Set button was not found")
    await click(set_btn)
    await notify(f"⏳ Set clicked, waiting {CUSTOM_AFTER_SET_SEC}s for the chart...")
    await asyncio.sleep(CUSTOM_AFTER_SET_SEC)


async def apply_period(page, period, notify):
    """Apply the period chosen in Telegram (preset button or custom date range)."""
    if period["type"] == "preset":
        btn = await find_preset_button(page, period["label"])
        if btn:
            await notify(f"📅 Selecting preset {period['label'].replace(' ', '')}...")
            await click(btn)
            await asyncio.sleep(8)
        else:
            await notify(f"⚠️ Preset '{period['label']}' not found; keeping the current view")

    elif period["type"] == "custom":
        try:
            await set_chart_dates(page, period["start"], period["end"], notify)
        except Exception as e:
            await notify(f"⚠️ Could not set the custom range ({str(e)[:150]}). Cancelling the picker...")
            _, cancel = await find_in_frames(page, 'input[id$="_DateRangePickerCancel"]', timeout=3)
            if cancel:
                try:
                    await click(cancel)
                except Exception:
                    pass
            await asyncio.sleep(2)
            await notify("↩️ Continuing with the chart's current range")

    rng = await get_range_text(page)
    if rng:
        await notify(f"📅 Chart range: {rng}")
    return rng


# =====================================================================================
# Trends: points, crop, capture
# =====================================================================================
async def place_points(page, chart_area, notify):
    """Ensure 2 freeze points are placed at the true start (+10px) and true right end of _chartArea."""

    chart_frame, plot_el = await find_in_frames(
        page, 'div[id^="Gauge_"][id$="_chartArea"]', timeout=15, need_size=True
    )
    if not plot_el:
        chart_frame, plot_el = await find_in_frames(
            page, 'div.highcharts-container', timeout=10, need_size=True
        )
    if not plot_el:
        raise Exception("Could not find Highcharts _chartArea in any frame.")

    await plot_el.scroll_into_view_if_needed()
    await asyncio.sleep(0.5)

    box = await plot_el.bounding_box()
    if not box:
        raise Exception("Could not read _chartArea bounding box.")

    curve_targets = await plot_el.evaluate("""(el, {leftMargin, rightMargin}) => {
        const w = el.clientWidth || el.getBoundingClientRect().width;
        const h = el.clientHeight || el.getBoundingClientRect().height;
        const plotOffsetLeft = parseFloat(el.style.left) || 311;

        const paths = [...el.querySelectorAll('path.highcharts-tracker')];
        const leftCandidates = [];
        const rightCandidates = [];

        const wantLeftX = leftMargin;
        const wantRightX = w - rightMargin;

        for (const p of paths) {
            const d = p.getAttribute('d') || '';
            const pts = [...d.matchAll(/[ML]\\s*(-?[\\d.]+)\\s+([\\d.]+)/g)].map(m => ({
                x: parseFloat(m[1]),
                y: parseFloat(m[2])
            })).filter(pt => pt.x >= 0 && pt.x <= w && pt.y >= 0 && pt.y <= h);

            if (pts.length > 2) {
                const leftPt = pts.reduce((a, b) => Math.abs(b.x - wantLeftX) < Math.abs(a.x - wantLeftX) ? b : a);
                const realPts = pts.filter(pt => pt.x <= w - 2);
                const pool = realPts.length ? realPts : pts;
                const rightPt = pool.reduce((a, b) => Math.abs(b.x - wantRightX) < Math.abs(a.x - wantRightX) ? b : a);

                leftCandidates.push(leftPt);
                rightCandidates.push(rightPt);
            }
        }

        const sortByActive = (a, b) => (a.y < h - 4 ? -1 : 1) - (b.y < h - 4 ? -1 : 1);
        leftCandidates.sort(sortByActive);
        rightCandidates.sort(sortByActive);

        return {
            w: w,
            h: h,
            plotOffsetLeft: plotOffsetLeft,
            left: leftCandidates,
            right: rightCandidates
        };
    }""", {"leftMargin": START_CLICK_FROM_LEFT_PX, "rightMargin": END_CLICK_FROM_RIGHT_PX})

    plot_w = curve_targets.get("w") or box["width"]
    plot_h = curve_targets.get("h") or box["height"]
    plot_left_css = curve_targets.get("plotOffsetLeft") or 311.0

    left_max_css = plot_left_css + (plot_w * 0.25)
    right_min_css = plot_left_css + (plot_w * 0.75)

    async def get_freeze_boxes():
        return await chart_frame.evaluate("""() => {
            const els = [...document.querySelectorAll('#freezeDataContainer > div[id$="_freezeContainer"].draggable')];
            return els.map(e => ({
                id: e.id,
                left: parseFloat(e.style.left) || 0,
                time: (e.innerText || '').trim().split('\\n')[0] || ''
            }));
        }""")

    async def remove_misplaced_boxes():
        await chart_frame.evaluate("""({leftMax, rightMin}) => {
            const els = [...document.querySelectorAll('#freezeDataContainer > div[id$="_freezeContainer"].draggable')];
            for (const e of els) {
                const l = parseFloat(e.style.left) || 0;
                if (l > leftMax && l < rightMin) {
                    const closeBtn = e.querySelector('img[id$="_close"]');
                    if (closeBtn) closeBtn.click();
                }
            }
        }""", {"leftMax": left_max_css, "rightMin": right_min_css})

    await chart_frame.evaluate("""() => {
        document.querySelectorAll('#freezeDataContainer img[id$="_close"]').forEach(b => b.click());
        const fc = document.getElementById('freezeDataContainer');
        if (fc) fc.style.pointerEvents = 'none';
    }""")
    await asyncio.sleep(0.4)

    async def hover_and_click(x_rel, y_rel, from_right=False):
        abs_x = box["x"] + x_rel
        abs_y = box["y"] + y_rel

        approach_x = abs_x - 25 if from_right else abs_x + 25
        await page.mouse.move(approach_x, abs_y)
        await asyncio.sleep(0.15)
        await page.mouse.move(abs_x, abs_y, steps=6)
        await asyncio.sleep(0.35)
        await page.mouse.down()
        await asyncio.sleep(0.1)
        await page.mouse.up()
        await asyncio.sleep(POINT_WAIT_SEC)

    await notify(
        f"📍 Placing 2 points inside _chartArea (width {plot_w:.0f}x{plot_h:.0f}px)..."
    )

    left_pts = curve_targets.get("left") or [{"x": float(START_CLICK_FROM_LEFT_PX), "y": plot_h * POINT_Y_FRAC}]
    right_pts = curve_targets.get("right") or [{"x": plot_w - float(END_CLICK_FROM_RIGHT_PX), "y": plot_h * POINT_Y_FRAC}]

    for attempt in range(4):
        await remove_misplaced_boxes()
        boxes = await get_freeze_boxes()
        has_left = any(b["left"] <= left_max_css for b in boxes)
        has_right = any(b["left"] >= right_min_css for b in boxes)

        if has_left and has_right:
            break

        if not has_left:
            pt = left_pts[attempt % len(left_pts)]
            target_x = max(float(START_CLICK_FROM_LEFT_PX), pt["x"]) + (attempt * 2.0)
            await hover_and_click(target_x, pt["y"], from_right=False)

        await remove_misplaced_boxes()
        boxes = await get_freeze_boxes()
        has_right = any(b["left"] >= right_min_css for b in boxes)

        if not has_right:
            pt = right_pts[attempt % len(right_pts)]
            target_x = min(plot_w - float(END_CLICK_FROM_RIGHT_PX), pt["x"]) - (attempt * 2.0)
            await hover_and_click(target_x, pt["y"], from_right=True)

        await remove_misplaced_boxes()
        boxes = await get_freeze_boxes()
        if any(b["left"] <= left_max_css for b in boxes) and any(b["left"] >= right_min_css for b in boxes):
            break

        await notify(
            f"⚠️ Point check (attempt {attempt + 1}/4): {len(boxes)} valid point(s). Adjusting by 2px and retrying..."
        )

    await chart_frame.evaluate("""() => {
        const fc = document.getElementById('freezeDataContainer');
        if (fc) fc.style.pointerEvents = 'auto';
    }""")

    final_boxes = await get_freeze_boxes()
    if len(final_boxes) >= 2:
        info = ", ".join(f"{b['time']} (left={b['left']:.1f}px)" for b in final_boxes)
        await notify(f"✅ 2 freeze points verified: {info}")
    else:
        await notify(f"⚠️ Only {len(final_boxes)}/2 freeze points could be placed.")


def crop_right(path, px):
    """Remove `px` pixels from the right side of the image."""
    if px <= 0:
        return
    img = Image.open(path)
    w, h = img.size
    if w > px + 10:
        img.crop((0, 0, w - px, h)).save(path)


async def render_and_screenshot_chart(
    page, notify, out_path, period, search_term="", extra_wait=0, reload_tab=False, remove_pd=False
):
    """Apply `period` to the Trends chart, wait, toggle PD if requested, place the 2 points,
    and save a screenshot. Returns (chart_well_name, period_label)."""
    if reload_tab:
        await notify(f"🔃 Reloading Trends tab (with +{extra_wait}s extra wait)...")
        _, tab = await find_in_frames(page, TRENDS_TAB, timeout=15)
        if tab:
            await click(tab)
            await asyncio.sleep(6 + extra_wait)

    _, chart_area = await find_in_frames(page, CHART_SEL, timeout=75, need_size=True)
    if not chart_area:
        raise Exception(f"Chart not found ({search_term}).\n" + frames_report(page))

    rng_text = await apply_period(page, period, notify)
    if period["type"] != "default":
        if extra_wait > 0:
            await notify(f"⏳ Waiting an extra {extra_wait}s for custom/preset range...")
            await asyncio.sleep(extra_wait)
        _, new_area = await find_in_frames(page, CHART_SEL, timeout=30, need_size=True)
        chart_area = new_area or chart_area

    curve_wait = 6 + extra_wait
    await notify(f"📈 Waiting {curve_wait}s for the curves to finish drawing...")
    await asyncio.sleep(curve_wait)

    chart_well_name = await get_chart_well_name(page, fallback=search_term)
    period_label = compute_period_label(period, rng_text)
    await notify(f"🏷 Chart header well: {chart_well_name} | Estimated period: {period_label}")

    await set_pd_visibility_on_chart(page, remove_pd=remove_pd, notify=notify)

    if PLACE_POINTS:
        try:
            await place_points(page, chart_area, notify)
        except Exception as e:
            await notify(f"⚠️ Could not place the points: {str(e)[:200]} (continuing without them)")

    await chart_area.screenshot(path=out_path)
    crop_right(out_path, CROP_RIGHT_PX)
    await notify("📸 Chart screenshot captured")
    return chart_well_name, period_label


async def capture_chart_advanced(well_name, page, notify, out_path, login_ui, period, remove_pd=False):
    """Open the well for the first time, open the Trends tab, and capture the chart.
    Returns (chart_well_name, period_label)."""
    search_term, upper_name, clicked = await open_well(well_name, page, notify, login_ui)

    await ensure_tab_open(page, TRENDS_TAB, "Trends tab", notify, timeout=20, required=False)

    opened = None
    for _ in range(2):
        opened, _ = await find_in_frames(page, "iframe[id='measurementtblFrame']", timeout=25)
        if opened:
            break
        await notify("🔁 Chart frame not ready, clicking the Trends tab again...")
        _, tab = await find_in_frames(page, TRENDS_TAB, timeout=5)
        if tab:
            await click(tab)
            await asyncio.sleep(3)
    if not opened:
        raise Exception(
            f"Well '{search_term}' was typed but never opened (clicked result: {clicked}).\n"
            f"Elements containing '{upper_name}':\n{await dump_candidates(page, upper_name)}\n\n{frames_report(page)}"
        )

    await notify("⏳ Loading well data and chart...")
    await asyncio.sleep(14)
    return await render_and_screenshot_chart(
        page, notify, out_path, period, search_term=upper_name, extra_wait=0, reload_tab=False, remove_pd=remove_pd
    )


# =====================================================================================
# Grid View > Facility Parameters
# =====================================================================================
GRID_ROW_SEL = 'div[role="row"][id^="row"][id$="facilitytblcontainer"]'
GRID_JS = """() => {
  const out = [];
  document.querySelectorAll('div[role="row"][id^="row"][id$="facilitytblcontainer"]').forEach(row => {
    const lab = row.querySelector('div[columnindex="3"] label');
    if (!lab) return;
    const cell = i => {
      const c = row.querySelector('div[columnindex="' + i + '"]');
      return c ? (c.innerText || '').replace(/\\s+/g, ' ').trim() : '';
    };
    out.push({
      name: (lab.getAttribute('actualtagname') || lab.innerText || '').trim(),
      value: cell(4),
      unit: cell(5),
      time: cell(6),
    });
  });
  return out;
}"""


def norm_name(s):
    return re.sub(r"\s+", " ", s or "").strip().casefold()


async def grid_frame(page):
    """The frame that currently holds the grid rows."""
    for fr in page.frames:
        try:
            if await fr.query_selector(GRID_ROW_SEL):
                return fr
        except Exception:
            pass
    return None


async def scan_grid(page, visit, max_steps=80):
    """Scroll through the Facility Parameters grid from the top, calling `await visit(frame)`."""
    fr = None
    for _ in range(40):
        try:
            fr = await grid_frame(page)
            if fr and await fr.evaluate(GRID_JS):
                break
        except Exception:
            pass
        await asyncio.sleep(1)
    else:
        raise Exception("The Facility Parameters grid did not load.\n" + frames_report(page))

    content = await fr.query_selector("#contentfacilitytblcontainer")
    box = await content.bounding_box() if content else None
    if box:
        await page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        for _ in range(8):
            await page.mouse.wheel(0, -3000)
            await asyncio.sleep(0.2)
        await asyncio.sleep(0.8)

    seen, stale = set(), 0
    for _ in range(max_steps):
        result = await visit(fr)
        if result:
            return result
        names = {norm_name(r["name"]) for r in await fr.evaluate(GRID_JS)}
        stale = stale + 1 if names <= seen else 0
        seen |= names
        if not box or stale >= 3:
            break
        await page.mouse.wheel(0, 300)
        await asyncio.sleep(0.8)
    return None


async def collect_grid_params(page, notify):
    """Read Measurement / Value / Unit / Date Time of the GRID_PARAMS rows."""
    wanted = {norm_name(n): n for n in GRID_PARAMS}
    found = {}

    async def visit(fr):
        for r in await fr.evaluate(GRID_JS):
            key = norm_name(r["name"])
            if key in wanted and key not in found:
                found[key] = r
        return len(found) == len(wanted)

    await notify("🧾 Reading the Facility Parameters grid...")
    await scan_grid(page, visit)

    results = []
    for name in GRID_PARAMS:
        r = found.get(norm_name(name)) or {}
        results.append(
            {"param": name, "found": bool(r), "value": r.get("value", ""), "unit": r.get("unit", ""), "time": r.get("time", "")}
        )
    await notify(f"🧾 Found {len(found)}/{len(results)} parameters")
    return results


async def _cb_checked(cb):
    try:
        return await cb.evaluate(
            "e => /checkbox-check-checked/i.test(e.innerHTML) || e.getAttribute('aria-checked') === 'true'"
        )
    except Exception:
        return False


async def capture_grid_view(well_name, page, notify, login_ui):
    """Open the well, go to Grid View > Facility Parameters and read the wanted rows."""
    await open_well(well_name, page, notify, login_ui)
    await ensure_tab_open(page, GRID_TAB, "Grid View", notify)
    await ensure_tab_open(page, FACILITY_TAB, "Facility Parameters", notify)
    await asyncio.sleep(3)
    return await collect_grid_params(page, notify)


def format_grid_message(well, results):
    lines = [f"<b>🧾 Facility Parameters — {html_lib.escape(well)}</b>", "Measurement → Value (Unit) | Date Time", ""]
    for i, r in enumerate(results, 1):
        name = html_lib.escape(r["param"])
        if r["found"]:
            val = f"{r['value']} {r['unit'] if r['unit'] not in ('', '-') else ''}".strip()
            lines.append(f"{i}. <b>{name}</b>\n    {html_lib.escape(val)}  |  {html_lib.escape(r['time'])}")
        else:
            lines.append(f"{i}. <b>{name}</b>\n    ⚠️ not found in the grid")
    return "\n".join(lines)


# =====================================================================================
# Gemini analysis and report (including follow-up prompt editing)
# =====================================================================================
EXCLUDED_EXTRAS_RE = re.compile(
    r"vib.*y|vy\b|y\s*axis|temp.*intake|intake.*temp|ti\b",
    re.I,
)


def _fast_gen_config(model_name: str, json_mode: bool = False):
    """Build a fast GenerateContentConfig with low/zero thinking overhead."""
    kwargs = {"temperature": 0.1}
    if json_mode:
        kwargs["response_mime_type"] = "application/json"

    try:
        if "pro" in model_name.lower():
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=128)
        else:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    except Exception:
        pass
    return types.GenerateContentConfig(**kwargs)


def format_report_numbers(param_key: str, text: str) -> str:
    """Enforce strict decimal rounding rules on any numbers inside a report line."""
    if not text:
        return text

    text = re.sub(
        r"\b(\d+(?:,\d{3})*(?:\.\d+)?)(\s*Hz\b)",
        lambda m: f"{round(float(m.group(1).replace(',', ''))):.0f}{m.group(2)}",
        text,
        flags=re.I,
    )

    key_upper = (param_key or "").upper()
    is_whole_only = key_upper in ("PIP", "PD")

    def _replace_num(m):
        full = m.group(0)
        after = text[m.end() : m.end() + 5]
        if re.match(r"^\s*Hz\b", after, re.I):
            return full

        val = float(full.replace(",", ""))
        if is_whole_only:
            return f"{round(val):.0f}"
        else:
            rounded = round(val, 1)
            return f"{rounded:.0f}" if rounded.is_integer() else f"{rounded:.1f}"

    return re.sub(r"\b\d+(?:,\d{3})*\.\d+\b", _replace_num, text)


def build_prompt(well: str, period_label: str = "Last 24 Hours", remove_pd: bool = False) -> str:
    pd_instruction = (
        '- "PD" was intentionally hidden/removed by the user; set "PD" status to "not_visible" with empty text.'
        if remove_pd
        else '- "PD"        = "Press-Pump Discharge"'
    )
    return f"""You are an ESP (Electric Submersible Pump) surveillance engineer.
Analyze the attached ProductionLink chart image for well {well} and describe what happened
during the period shown on the chart ({period_label}).

Return ONLY one JSON object (no markdown, no code fence) with exactly this structure:

{{
  "parameters": {{
    "PIP":       {{"status": "ok", "text": "Decreased from 2164 Psi to 1105 Psi", "note": ""}},
    "PD":        {{"status": "ok", "text": "...", "note": ""}},
    "Motor Amp": {{"status": "ok", "text": "...", "note": ""}},
    "Tm":        {{"status": "ok", "text": "Increased from 271.5 to 296.2 F", "note": ""}}
  }},
  "extras": [
    {{"name": "Vibration (Vx)", "text": "Varied between 0.2 & 0.5 G"}}
  ]
}}

CURVE MAPPING (use the legend/labels on the chart):
- "PIP"       = "Pump Intake Pressure" or "Press-Pump Intake"
{pd_instruction}
- "Motor Amp" = "Pwr-Motor Amps Ph B"
- "Tm"        = "Temp-Motor"

STRICT EXCLUSIONS (DO NOT analyze or include these under any circumstances):
- NEVER include "Vibration (Vy)" or "Vib-Pump Y axis".
- NEVER include "Temp-Pump Intake" or "Pump Intake Temperature".

NUMBER ROUNDING & PRECISION RULES (MANDATORY):
- "PIP" and "PD": NO fractions/decimals at all (round to whole integers, e.g. 2864 Psi, 1105 Psi).
- "Frequency (Hz)": NO fractions/decimals at all (round to whole integers, e.g. 45 Hz, 50 Hz — never 45.01 Hz).
- "Motor Amp", "Tm", and "Vibration (Vx)": AT MOST ONE decimal place (1 fraction digit, e.g. 40.2 A, 324.5 F, 0.1 G — round 0.12 G to 0.1 G).

STATUS RULES:
- "ok": you can read the curve and describe it. Fill "text".
- "not_visible": the curve for this parameter is not on the chart (or its legend tag is grayed out). Leave "text" empty and explain in "note".
- "unreadable": the curve exists but the values cannot be read reliably. Leave "text" empty and explain in "note".
- "zero": the curve is visible (active colored legend, not grayed out) and stays at 0 (flat on the zero line) for the WHOLE period.
  Put a short description in "text", for example "Constant at 0 Psi".
- Never guess or invent numbers just to fill a parameter.

STYLE OF "text" (no parameter name, no asterisks, no trailing period, units always included: Psi, A, F, G, Hz):
- Trend: "Decreased from X Psi to Y Psi" / "Increased from X to Y F"
- Range: "Varied between X & Y G" / "Fluctuated between X & Y"
- Spike: add "& spiked up to Z" or "& dropped to Z" when relevant
- Motor Amp must also describe the frequency, for example:
  "Started at 52.7 A, then changed between 64.1 & 65.1 A while Frequency started at 43 Hz and increased to 50 Hz"
  or, if frequency did not change: "... while Frequency was constant at 45 Hz".
  If no frequency curve is visible, describe only the amps.
- If the well was shut down or restarted during the period, mention it briefly inside the relevant texts.

"extras": other parameters clearly visible on the chart (for example Vibration (Vx)), EXCEPT "Vibration (Vy)" / "Vib-Pump Y axis" and "Temp-Pump Intake" which must NEVER be included.
Use the same style. Leave the list empty if there are none. Do not put the four main parameters there.

Write in English."""


def parse_json_loose(text: str) -> dict:
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", (text or "").strip()).strip()
    try:
        return json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def clean_text(s) -> str:
    return str(s or "").replace("*", "").strip().rstrip(".").strip()


async def _call_gemini_fast(model: str, contents: list, json_mode: bool = False, timeout: float = GEMINI_MODEL_TIMEOUT_SEC):
    """Call a Gemini model with a strict timeout so it never hangs."""
    cfg = _fast_gen_config(model, json_mode=json_mode)
    try:
        return await asyncio.wait_for(
            ai_client.aio.models.generate_content(model=model, contents=contents, config=cfg),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        raise Exception(f"timed out after {timeout:.0f}s")
    except Exception as e:
        if "thinking" in str(e).lower() or "invalid" in str(e).lower():
            fallback_cfg = types.GenerateContentConfig(
                temperature=0.1,
                **({"response_mime_type": "application/json"} if json_mode else {}),
            )
            return await asyncio.wait_for(
                ai_client.aio.models.generate_content(model=model, contents=contents, config=fallback_cfg),
                timeout=timeout,
            )
        raise


async def analyze_with_ai(image_path, well, period_label="Last 24 Hours", remove_pd=False, notify=None):
    """Ask the models in order and return the parsed JSON."""
    with open(image_path, "rb") as f:
        contents = [
            types.Part.from_bytes(data=f.read(), mime_type="image/png"),
            build_prompt(well, period_label=period_label, remove_pd=remove_pd),
        ]

    models = [GEMINI_MODEL] + [m for m in GEMINI_FALLBACKS if m != GEMINI_MODEL]
    errors = []
    for i, model in enumerate(models):
        try:
            if notify:
                await notify(f"⚡ Analyzing the chart with {model} (max {GEMINI_MODEL_TIMEOUT_SEC}s)...")
            resp = await _call_gemini_fast(model, contents, json_mode=True, timeout=GEMINI_MODEL_TIMEOUT_SEC)
            if not resp.text:
                raise Exception("empty response")
            data = parse_json_loose(resp.text)
            if not isinstance(data, dict) or "parameters" not in data:
                raise Exception("unexpected JSON structure")
            return data
        except Exception as e:
            short = re.sub(r"\s+", " ", str(e))[:160]
            errors.append(f"{model}: {short}")
            if notify:
                nxt = f" Trying {models[i + 1]}..." if i + 1 < len(models) else ""
                await notify(f"⚠️ {model} failed ({short}).{nxt}")
    raise Exception("All Gemini models failed:\n" + "\n".join(errors))


async def refine_report_with_ai(image_path, well, period_label, current_report, user_instruction, notify=None):
    """Refine the existing WhatsApp report using the chart image and the user's follow-up prompt."""
    prompt = f"""You are an ESP (Electric Submersible Pump) surveillance engineer.
Here is the current well performance report for well {well}:

{current_report}

The user wants to edit/update this observation with the following instruction:
"{user_instruction}"

STRICT RULES:
1. Never include "Vibration (Vy)" / "Vib-Pump Y axis" or "Temp-Pump Intake" unless explicitly requested.
2. Rounding rules:
   - PIP, PD, and Frequency (Hz): whole numbers only (NO decimals, e.g. 2864 Psi, 45 Hz).
   - Motor Amp, Tm, and Vibration (Vx): maximum 1 decimal place (e.g. 40.2 A, 324.5 F, 0.1 G).

Using the attached chart image and the user's instruction, return ONLY the updated report in the exact same WhatsApp format (starting with *{well} | Well Performance Update ({period_label})* and using bullet points `- *Parameter*: ...`). Do not wrap it in markdown code fences or add extra commentary."""

    with open(image_path, "rb") as f:
        contents = [types.Part.from_bytes(data=f.read(), mime_type="image/png"), prompt]

    models = [GEMINI_MODEL] + [m for m in GEMINI_FALLBACKS if m != GEMINI_MODEL]
    errors = []
    for i, model in enumerate(models):
        try:
            if notify:
                await notify(f"⚡ Updating observation with {model} (max {GEMINI_MODEL_TIMEOUT_SEC}s)...")
            resp = await _call_gemini_fast(model, contents, json_mode=False, timeout=GEMINI_MODEL_TIMEOUT_SEC)
            if not resp.text:
                raise Exception("empty response")
            out_lines = []
            for line in resp.text.strip().splitlines():
                m = re.match(r"^(-\s*\*([^*]+)\*:\s*)(.+)$", line)
                if m:
                    out_lines.append(f"{m.group(1)}{format_report_numbers(m.group(2), m.group(3))}")
                else:
                    out_lines.append(line)
            return "\n".join(out_lines)
        except Exception as e:
            short = re.sub(r"\s+", " ", str(e))[:160]
            errors.append(f"{model}: {short}")
            if notify:
                nxt = f" Trying {models[i + 1]}..." if i + 1 < len(models) else ""
                await notify(f"⚠️ {model} failed ({short}).{nxt}")
    raise Exception("All Gemini models failed:\n" + "\n".join(errors))


def param_info(data, key):
    p = (data.get("parameters") or {}).get(key) or {}
    return str(p.get("status", "")).lower(), clean_text(p.get("text")), clean_text(p.get("note"))


ZERO_TEXT_RE = re.compile(
    r"\b(?:at|of|was|is|remained|stayed)\s+0(?:\.0+)?\s*(?:psi|a|f|hz|g)\b"
    r"|\bfrom\s+0(?:\.0+)?\s*\w*\s+to\s+0(?:\.0+)?\b",
    re.I,
)


def param_state(data, key):
    """'ok', 'zero' (flat at 0 for the whole period) or 'missing' (not visible / unreadable)."""
    status, text, _ = param_info(data, key)
    if status == "zero":
        return "zero"
    if status == "ok" and text:
        return "zero" if len(text) < 40 and ZERO_TEXT_RE.search(text) else "ok"
    return "missing"


def build_report(well, period_label, data, decisions):
    lines = [f"*{well} | Well Performance Update ({period_label})*", ""]
    for key in PARAM_ORDER:
        if decisions.get(key, (None, None))[0] == "remove":
            continue
        _, text, _ = param_info(data, key)
        text = format_report_numbers(key, text)
        state = param_state(data, key)
        if state == "ok":
            lines.append(f"- *{key}*: {text}")
            continue
        action, value = decisions.get(key, ("keep" if state == "zero" else "na", None))
        if action == "remove":
            continue
        if action == "manual" and value:
            lines.append(f"- *{key}*: {format_report_numbers(key, value)}")
        elif state == "zero":
            lines.append(f"- *{key}*: {text or 'Constant at 0'}")
        else:
            lines.append(f"- *{key}*: Not available")
    for extra in data.get("extras") or []:
        name, text = clean_text(extra.get("name")), clean_text(extra.get("text"))
        if name and text and not EXCLUDED_EXTRAS_RE.search(name):
            lines.append(f"- *{name}*: {format_report_numbers(name, text)}")
    return "\n".join(lines)


async def resolve_decisions(
    bot,
    chat_id,
    well,
    period_label,
    data,
    image_path,
    page=None,
    period=None,
    remove_pd=False,
    update_msg=None,
    notify=None,
):
    """Ask what to do with each parameter that is missing or 0 all along.
    If `remove_pd=True`, PD is automatically excluded without asking."""
    decisions = {"PD": ("remove", None)} if remove_pd else {}
    extra_wait = 0

    while True:
        pending = [k for k in PARAM_ORDER if param_state(data, k) != "ok" and k not in decisions]
        if not pending:
            return data, decisions, well, period_label

        key = pending[0]
        status, ptext, note = param_info(data, key)
        details = f"\nDetails: {note}" if note else ""
        if param_state(data, key) == "zero":
            text = (
                f"⚠️ {key} ({PARAM_FULL[key]}) is 0 along the whole period.\n"
                f"It would appear in the report as:\n- *{key}*: {ptext or 'Constant at 0'}{details}\n\nShould I keep it?"
            )
            first = [("✅ Keep it in the report", "keep"), ("🚫 Remove it from the report", "rm")]
        else:
            reason = "is not on the chart" if status == "not_visible" else "could not be read reliably"
            text = f"⚠️ {key} ({PARAM_FULL[key]}) {reason}.{details}\n\nWhat should I do?"
            first = [("🚫 Remove it from the report", "rm"), ("📝 Show it as 'Not available'", "na")]

        extra_options = [
            ("✍️ I will type it myself", "man"),
            ("🔄 Re-analyze the chart", "re"),
        ]
        if page is not None and period is not None:
            extra_options.append(("🔃 Reload chart & resend screenshot (+5s wait)", "reload"))

        action = await ask_choice(bot, chat_id, text, first + extra_options)

        if action == "keep":
            decisions[key] = ("keep", None)
        elif action == "rm":
            decisions[key] = ("remove", None)
        elif action == "na":
            decisions[key] = ("na", None)
        elif action == "man":
            typed = clean_text(
                await ask_text(
                    bot,
                    chat_id,
                    f"✍️ Send the text for {key}, for example:\nDecreased from 2164 Psi to 1105 Psi\n\n(or send 'skip' to remove it)",
                    r"(?s)^.+$",
                )
            )
            decisions[key] = ("remove", None) if typed.lower() in ("", "skip", "remove") else ("manual", typed)
        elif action == "reload":
            extra_wait += 5
            try:
                well, period_label = await render_and_screenshot_chart(
                    page,
                    notify,
                    image_path,
                    period,
                    well,
                    extra_wait=extra_wait,
                    reload_tab=True,
                    remove_pd=remove_pd,
                )
                with open(image_path, "rb") as photo:
                    if update_msg:
                        await update_msg.reply_photo(
                            photo=photo, caption=f"📊 Reloaded chart (+{extra_wait}s wait) for well: {well}"
                        )
                    else:
                        await bot.send_photo(
                            chat_id, photo=photo, caption=f"📊 Reloaded chart (+{extra_wait}s wait) for well: {well}"
                        )
                data = await analyze_with_ai(
                    image_path, well, period_label=period_label, remove_pd=remove_pd, notify=notify
                )
                decisions = {"PD": ("remove", None)} if remove_pd else {}
            except SessionEndedException:
                raise
            except Exception as e:
                await bot.send_message(chat_id, f"❌ Reload & re-analysis failed:\n{str(e)[:1500]}")
        else:  # re-analyze
            try:
                data = await analyze_with_ai(
                    image_path, well, period_label=period_label, remove_pd=remove_pd, notify=notify
                )
                decisions = {"PD": ("remove", None)} if remove_pd else {}
            except SessionEndedException:
                raise
            except Exception as e:
                await bot.send_message(chat_id, f"❌ Re-analysis failed:\n{str(e)[:1500]}")
                decisions[key] = ("remove", None) if param_state(data, key) == "zero" else ("na", None)


# =====================================================================================
# Grid View actions: single-select mark, Demand Scan, Edit Alarm Settings
# =====================================================================================
LOADING_DIALOG_SEL = (
    'div[aria-describedby="prodml_loading_dialog"], '
    '#prodml_loading_dialog, '
    '#appplication_wait_dialog_notification'
)


async def is_loading_dialog_visible(page):
    """Return True if the 'Fetching data. Please wait...' loading dialog is currently visible in any frame."""
    for fr in page.frames:
        try:
            for el in await fr.query_selector_all(LOADING_DIALOG_SEL):
                if await el.is_visible():
                    return True
        except Exception:
            pass
    return False


async def wait_loading_dialog_gone(page, notify, appear_timeout=6, disappear_timeout=120):
    """Wait for #prodml_loading_dialog ('Fetching data. Please wait...') to appear and then disappear."""
    appeared = False
    for _ in range(int(appear_timeout * 2)):
        if await is_loading_dialog_visible(page):
            appeared = True
            await notify("⏳ Loading dialog appeared ('Fetching data. Please wait...'), waiting for it to finish...")
            break
        await asyncio.sleep(0.5)

    for _ in range(int(disappear_timeout)):
        if not await is_loading_dialog_visible(page):
            if appeared:
                await notify("✅ Loading dialog finished.")
            return
        await asyncio.sleep(1)

    await notify("⚠️ Loading dialog wait timed out; continuing...")


async def reset_and_mark_single_param(page, target_name, notify):
    """Click Facility Parameters tab to refresh/uncheck all, ensure no other row is checked,
    and tick only `target_name`."""
    await notify("🔄 Clicking 'Facility Parameters' to refresh and uncheck all parameters...")
    _, fac_tab = await find_in_frames(page, FACILITY_TAB, timeout=20)
    if fac_tab:
        await click(fac_tab)
        await asyncio.sleep(4)
        await wait_loading_dialog_gone(page, notify, appear_timeout=3, disappear_timeout=45)

    want_key = norm_name(target_name)
    result_status = "not found"

    async def visit(fr):
        nonlocal result_status
        for row in await fr.query_selector_all(GRID_ROW_SEL):
            lab = await row.query_selector('div[columnindex="3"] label')
            if not lab:
                continue
            key = norm_name((await lab.get_attribute("actualtagname")) or (await lab.inner_text()))
            cb = await row.query_selector('div[columnindex="1"] .jqx-checkbox')
            if not cb:
                if key == want_key:
                    result_status = "no checkbox"
                continue

            is_checked = await _cb_checked(cb)
            if key == want_key:
                if not is_checked:
                    await click(cb)
                    await asyncio.sleep(0.8)
                result_status = "marked" if await _cb_checked(cb) else "clicked"
                await notify(f"☑️ {target_name}: {result_status}")
            elif is_checked:
                await click(cb)
                await asyncio.sleep(0.5)

        return result_status in ("marked", "clicked", "no checkbox")

    await scan_grid(page, visit)
    return result_status


async def run_demand_scan(bot, chat_id, well, page, notify, debug_path):
    """Click Demand Scan -> click Ok if confirmation appears, or skip if loading bar appears directly ->
    wait for loading bar to disappear -> send screenshot."""
    await notify("📡 Clicking Demand Scan...")
    _, btn = await find_in_frames(page, 'a[title="Demand Scan"], a[onclick*="demandScan"]', timeout=15)
    if not btn:
        raise Exception("Demand Scan button (onclick='demandScan()') was not found.")
    await click(btn)
    await asyncio.sleep(1.5)

    await notify("🆗 Checking for Demand Scan Confirmation or Loading bar...")
    ok_clicked = False
    loading_seen = False

    for _ in range(24):
        if await is_loading_dialog_visible(page):
            loading_seen = True
            await notify("⏳ Loading bar detected directly (confirmation dialog skipped).")
            break

        for fr in page.frames:
            try:
                dlg = await fr.query_selector("#demandScanConfirmationDlg")
                if dlg and await dlg.is_visible():
                    ok_btn = await fr.query_selector(
                        'div[aria-describedby="demandScanConfirmationDlg"] .ui-dialog-buttonset button:has-text("Ok")'
                    )
                    if not ok_btn:
                        for b in await fr.query_selector_all(".ui-dialog-buttonset button"):
                            if (await b.inner_text()).strip().lower() == "ok":
                                ok_btn = b
                                break
                    if ok_btn:
                        await click(ok_btn)
                        ok_clicked = True
                        break
            except Exception:
                pass
        if ok_clicked:
            await notify("✅ Confirmed 'Ok'. Waiting for data scan to complete...")
            break
        await asyncio.sleep(0.5)

    if not ok_clicked and not loading_seen:
        await notify("ℹ️ No 'Ok' dialog appeared; waiting to ensure any scan loading finishes...")

    await wait_loading_dialog_gone(page, notify, appear_timeout=8, disappear_timeout=180)
    await asyncio.sleep(3)

    shot = debug_path.replace("debug_error", "demand_scan")
    try:
        fr = await grid_frame(page)
        content = await fr.query_selector("#contentfacilitytblcontainer") if fr else None
        if content:
            await content.screenshot(path=shot)
        else:
            await page.screenshot(path=shot)
        with open(shot, "rb") as ph:
            await bot.send_photo(chat_id, ph, caption=f"📡 Demand Scan completed — {well}")
    finally:
        if os.path.exists(shot):
            os.remove(shot)


async def run_edit_alarm_settings(bot, chat_id, well, param_name, page, notify, debug_path):
    """Click Edit Alarm Settings -> screenshot popup -> ask which Threshold Settings to update ->
    type new values -> send preview screenshot for approval -> Save or Close."""
    await notify(f"🚨 Opening Edit Alarm Settings for {param_name}...")
    _, edit_btn = await find_in_frames(
        page, '#exceptionSettingAnchor, a[title="Edit Alarm Settings"], a[onclick*="setExceptionSettingsParameters"]', timeout=15
    )
    if not edit_btn:
        raise Exception("Edit Alarm Settings button (#exceptionSettingAnchor) was not found.")
    await click(edit_btn)
    await asyncio.sleep(4)

    _, dlg_el = await find_in_frames(
        page, 'div[aria-describedby="prodExceptionSettingsDiv"], #prodExceptionSettingsDiv', timeout=20, need_size=True
    )
    if not dlg_el:
        raise Exception("Alarm Settings popup (#prodExceptionSettingsDiv) did not appear.")

    alarm_frame = None
    for _ in range(20):
        for fr in page.frames:
            if "alarmSettings.do" in (fr.url or "") or fr.name == "prodExceptionSettingsFrame":
                alarm_frame = fr
                break
            try:
                if await fr.query_selector("#setAlarmSettingsButton, #closeAlarmSettingsButton"):
                    alarm_frame = fr
                    break
            except Exception:
                pass
        if alarm_frame:
            break
        await asyncio.sleep(1)

    if not alarm_frame:
        raise Exception("Could not locate the inner Alarm Settings iframe (#prodExceptionSettingsFrame).")

    await asyncio.sleep(2)

    async def send_popup_screenshot(caption):
        shot = debug_path.replace("debug_error", "alarm_settings")
        try:
            await dlg_el.screenshot(path=shot)
            with open(shot, "rb") as ph:
                await bot.send_photo(chat_id, ph, caption=caption)
        finally:
            if os.path.exists(shot):
                os.remove(shot)

    await send_popup_screenshot(f"🚨 Alarm Settings — {param_name} ({well})")

    fields = await alarm_frame.evaluate("""() => {
        const inputs = [...document.querySelectorAll('input[type="text"], input[type="number"], input:not([type])')];
        const out = [];
        inputs.forEach((inp, idx) => {
            const r = inp.getBoundingClientRect();
            if (r.width <= 0 || r.height <= 0 || inp.disabled || inp.readOnly) return;
            inp.setAttribute('data-bot-input-idx', String(idx));

            let label = '';
            const tr = inp.closest('tr');
            if (tr) {
                const cells = [...tr.querySelectorAll('td, th')];
                const myCell = inp.closest('td, th');
                const myIdx = cells.indexOf(myCell);
                if (myIdx > 0) {
                    label = (cells[myIdx - 1].innerText || '').replace(/\\s+/g, ' ').trim();
                }
                if (!label) {
                    label = cells.map(c => (c.innerText || '').replace(/\\s+/g, ' ').trim()).filter(Boolean)[0] || '';
                }
            }
            if (!label) {
                label = inp.name || inp.id || inp.title || `Threshold #${out.length + 1}`;
            }
            out.push({
                idx: idx,
                label: label.replace(/:$/, '').trim(),
                id: inp.id || inp.name || '',
                value: inp.value || ''
            });
        });
        return out;
    }""")

    if not fields:
        await bot.send_message(chat_id, "⚠️ No editable threshold inputs were found inside Alarm Settings.")
        return

    while True:
        rows = [
            [(f"✏️ {f['label']} (current: {f['value'] or 'empty'})", ("edit", i))]
            for i, f in enumerate(fields)
        ]
        rows.append([("✅ Done editing — Preview & Save", ("done", None)), ("❌ Cancel & Close", ("cancel", None))])

        (action, idx), msg = await ask_grid(
            bot,
            chat_id,
            f"🚨 Which Threshold Setting do you want to update for {param_name}?\n"
            "Tap a threshold to enter its new value, then press 'Done editing':",
            rows,
        )

        if action == "cancel":
            await msg.edit_text("❌ Closing Alarm Settings without saving...")
            close_btn = await alarm_frame.query_selector(
                'input#closeAlarmSettingsButton, input[onclick*="closeExceptionSettingsDiv"]'
            )
            if close_btn:
                await click(close_btn)
            return

        if action == "done":
            await msg.edit_text("📸 Capturing updated Alarm Settings for your approval...")
            break

        f = fields[idx]
        await msg.edit_text(f"✏️ Updating: {f['label']} (current: {f['value'] or 'empty'})")
        new_val = (
            await ask_text(
                bot,
                chat_id,
                f"✍️ Send the new value for '{f['label']}' (or send 'clear' to leave it empty, or 'cancel'):",
                r"^.*$",
            )
        ).strip()

        if new_val.lower() == "clear":
            new_val = ""

        inp_el = await alarm_frame.query_selector(f'input[data-bot-input-idx="{f["idx"]}"]')
        if not inp_el:
            raise Exception(f"Input for '{f['label']}' was lost.")

        await inp_el.evaluate("e => { e.click(); e.focus(); e.value = ''; }")
        await page.keyboard.press("Control+A")
        await page.keyboard.press("Delete")
        await asyncio.sleep(0.3)
        if new_val:
            await page.keyboard.type(new_val, delay=60)
        await inp_el.evaluate(
            "e => { e.dispatchEvent(new Event('input', {bubbles:true})); e.dispatchEvent(new Event('change', {bubbles:true})); }"
        )
        await asyncio.sleep(0.5)

        f["value"] = await inp_el.input_value()
        await notify(f"✏️ Set '{f['label']}' → '{f['value']}'")

    await send_popup_screenshot(f"🔍 Preview of updated Alarm Settings — {param_name}\nPlease approve to Save:")

    approve = await ask_choice(
        bot,
        chat_id,
        "Do you approve saving these Alarm Settings?",
        [
            ("💾 Yes, Save changes", True),
            ("❌ No, Close without saving", False),
        ],
    )

    if approve:
        await notify("💾 Clicking Save (#setAlarmSettingsButton)...")
        save_btn = await alarm_frame.query_selector(
            'input#setAlarmSettingsButton, input[onclick*="saveOrUpdateAlarmSettings"]'
        )
        if not save_btn:
            raise Exception("Save button (#setAlarmSettingsButton) was not found.")
        await click(save_btn)
        await asyncio.sleep(4)
        await wait_loading_dialog_gone(page, notify, appear_timeout=4, disappear_timeout=60)
        await bot.send_message(chat_id, f"✅ Alarm Settings saved for {param_name}.")
    else:
        await notify("❎ Clicking Close (#closeAlarmSettingsButton)...")
        close_btn = await alarm_frame.query_selector(
            'input#closeAlarmSettingsButton, input[onclick*="closeExceptionSettingsDiv"]'
        )
        if not close_btn:
            raise Exception("Close button (#closeAlarmSettingsButton) was not found.")
        await click(close_btn)
        await asyncio.sleep(2)
        await bot.send_message(chat_id, f"❎ Alarm Settings closed without saving for {param_name}.")


# =====================================================================================
# Telegram: browser session and request handling
# =====================================================================================
@asynccontextmanager
async def browser_page(debug_path):
    """A Chromium page (re-using the saved session). Saves the session on success and an
    error screenshot on failure."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
                f"--window-size={VIEWPORT_W},{VIEWPORT_H}",
            ],
        )
        ctx_kwargs = dict(
            viewport={"width": VIEWPORT_W, "height": VIEWPORT_H},
            device_scale_factor=2,
            user_agent=USER_AGENT,
        )
        if os.path.exists(STATE_FILE):
            ctx_kwargs["storage_state"] = STATE_FILE
        ctx = await browser.new_context(**ctx_kwargs)
        page = await ctx.new_page()
        try:
            yield page
            try:
                await ctx.storage_state(path=STATE_FILE)
            except Exception:
                pass
        except SessionEndedException:
            try:
                await ctx.storage_state(path=STATE_FILE)
            except Exception:
                pass
            raise
        except Exception:
            try:
                await page.screenshot(path=debug_path)
            except Exception:
                pass
            raise
        finally:
            await browser.close()


async def run_grid_flow(bot, chat_id, well, notify, debug_path):
    """Open Facility Parameters, send values, and loop allowing the user to select a parameter,
    perform Demand Scan or Edit Alarm Settings, view updated values, and repeat until End Session."""
    async with browser_page(debug_path) as page:
        results = await capture_grid_view(well, page, notify, LoginUI(bot, chat_id))
        await bot.send_message(chat_id, format_grid_message(well, results), parse_mode="HTML")

        while True:
            available = [r["param"] for r in results if r["found"]]
            if not available:
                await notify("✅ Done")
                return

            chosen_param = await pick_single_param_to_mark(bot, chat_id, available)
            if not chosen_param:
                await notify("✅ Done")
                return

            status = await reset_and_mark_single_param(page, chosen_param, notify)
            await bot.send_message(chat_id, f"☑️ Parameter marked: {chosen_param} ({status})")

            try:
                shot = debug_path.replace("debug_error", "grid")
                content = await (await grid_frame(page)).query_selector("#contentfacilitytblcontainer")
                await content.screenshot(path=shot)
                with open(shot, "rb") as ph:
                    await bot.send_photo(chat_id, ph, caption=f"🧾 Grid with '{chosen_param}' marked")
                os.remove(shot)
            except Exception:
                pass

            action = await ask_choice(
                bot,
                chat_id,
                f"What would you like to do with '{chosen_param}'?",
                [
                    ("📡 Demand Scan", "demand_scan"),
                    ("🚨 Edit Alarm Settings", "alarm_settings"),
                    ("🔁 Pick another parameter", "another"),
                ],
            )

            if action == "another":
                continue
            elif action == "demand_scan":
                await run_demand_scan(bot, chat_id, well, page, notify, debug_path)
            elif action == "alarm_settings":
                await run_edit_alarm_settings(bot, chat_id, well, chosen_param, page, notify, debug_path)
                _, fac_tab = await find_in_frames(page, FACILITY_TAB, timeout=15)
                if fac_tab:
                    await click(fac_tab)
                    await asyncio.sleep(3)
                    await wait_loading_dialog_gone(page, notify, appear_timeout=3, disappear_timeout=45)

            results = await collect_grid_params(page, notify)
            await bot.send_message(chat_id, format_grid_message(well, results), parse_mode="HTML")


async def run_trends_flow(bot, chat_id, well, update_msg, notify, path, debug_path):
    """Open Trends chart, read the true well name from the chart header, optionally remove PD,
    estimate the period in hours (or hours and half), and analyze with Gemini."""
    period = await pick_period(bot, chat_id, well)
    remove_pd = await ask_remove_pd(bot, chat_id, well)
    extra_wait = 0

    async with browser_page(debug_path) as page:
        chart_well, period_label = await capture_chart_advanced(
            well, page, notify, path, LoginUI(bot, chat_id), period, remove_pd=remove_pd
        )

        while True:
            with open(path, "rb") as photo:
                await update_msg.reply_photo(
                    photo=photo, caption=f"📊 Chart for well: {chart_well} ({period_label})"
                )

            data = await analyze_with_ai(
                path, chart_well, period_label=period_label, remove_pd=remove_pd, notify=notify
            )
            data, decisions, chart_well, period_label = await resolve_decisions(
                bot,
                chat_id,
                chart_well,
                period_label,
                data,
                path,
                page=page,
                period=period,
                remove_pd=remove_pd,
                update_msg=update_msg,
                notify=notify,
            )
            report_text = build_report(chart_well, period_label, data, decisions)
            await bot.send_message(chat_id, report_text)
            await notify("✅ Analysis ready")

            # Post-report options loop
            while True:
                pd_toggle_label = (
                    "📈 Restore PD on chart" if remove_pd else "🚫 Remove PD from chart"
                )
                next_action = await ask_choice(
                    bot,
                    chat_id,
                    f"What would you like to do next for {chart_well}?",
                    [
                        ("🔃 Reload chart & resend screenshot (+5s wait)", "reload_chart"),
                        ("🗓 Edit chart period", "edit_period"),
                        (pd_toggle_label, "toggle_pd"),
                        ("💬 Edit observation with Gemini", "edit_gemini"),
                    ],
                )

                if next_action == "edit_gemini":
                    followup = await ask_text(
                        bot,
                        chat_id,
                        "✍️ Send your follow-up instructions for Gemini to update the observation\n"
                        "(e.g. 'Mention that the well shut down at 14:00' or 'Update PIP to 1500 Psi'):",
                        r"(?s)^.+$",
                    )
                    report_text = await refine_report_with_ai(
                        path, chart_well, period_label, report_text, followup, notify
                    )
                    await bot.send_message(chat_id, report_text)
                    await notify("✅ Observation updated")
                    continue

                if next_action == "toggle_pd":
                    remove_pd = not remove_pd
                    chart_well, period_label = await render_and_screenshot_chart(
                        page,
                        notify,
                        path,
                        period,
                        chart_well,
                        extra_wait=extra_wait,
                        reload_tab=False,
                        remove_pd=remove_pd,
                    )
                    break

                if next_action == "reload_chart":
                    extra_wait += 5
                    chart_well, period_label = await render_and_screenshot_chart(
                        page,
                        notify,
                        path,
                        period,
                        chart_well,
                        extra_wait=extra_wait,
                        reload_tab=True,
                        remove_pd=remove_pd,
                    )
                    break

                if next_action == "edit_period":
                    extra_wait = 0
                    period = await pick_period(bot, chat_id, chart_well)
                    remove_pd = await ask_remove_pd(bot, chat_id, chart_well)
                    chart_well, period_label = await render_and_screenshot_chart(
                        page,
                        notify,
                        path,
                        period,
                        chart_well,
                        extra_wait=0,
                        reload_tab=False,
                        remove_pd=remove_pd,
                    )
                    break


async def handle_request(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    bot = context.bot

    waiter = TEXT_WAITERS.get(chat_id)
    if waiter:
        fut, rx, delete = waiter
        txt = (update.message.text or "").strip()
        if txt.lower() in ("stop", "end", "end session"):
            if not fut.done():
                fut.set_exception(SessionEndedException("Session ended by user."))
        elif txt.lower() == "cancel":
            if not fut.done():
                fut.set_exception(Exception("Cancelled by you."))
        elif rx.match(txt):
            if not fut.done():
                fut.set_result(txt)
            if delete:
                try:
                    await update.message.delete()
                except Exception:
                    pass
        else:
            await update.message.reply_text("That is not the format I am waiting for (or send 'cancel' / 'stop').")
        return

    well = update.message.text.strip()
    steps = [f"⏳ Processing well: {well}"]
    status = await update.message.reply_text("\n".join(steps))

    async def notify(text):
        steps.append(text)
        try:
            await status.edit_text("\n".join(steps)[-3500:])
        except Exception:
            pass

    token = str(next(_req_counter))
    path = f"{re.sub(r'[^A-Za-z0-9_-]', '_', well)}_{token}.png"
    debug_path = f"debug_error_{token}.png"

    try:
        view_choice = await pick_view(bot, chat_id, well)
        if view_choice == "grid":
            await run_grid_flow(bot, chat_id, well, notify, debug_path)
        else:
            await run_trends_flow(bot, chat_id, well, update.message, notify, path, debug_path)

        await notify("✅ Done")

    except SessionEndedException:
        await notify("🛑 Session stopped by user.")

    except Exception as e:
        await notify(f"❌ Error:\n{str(e)[:2500]}")
        if os.path.exists(debug_path):
            with open(debug_path, "rb") as debug_img:
                await update.message.reply_photo(photo=debug_img, caption="📸 Screenshot at the moment of the error")

    finally:
        for f in (path, debug_path):
            if os.path.exists(f):
                os.remove(f)
        try:
            await bot.send_message(chat_id, f"🛑 Session ended for well: {well}.")
        except Exception:
            pass


# =====================================================================================
# Lightweight Health-Check Server for Render (keeps the Free Web Service alive)
# =====================================================================================
class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"ESP Well Monitor Bot is alive and running!")

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        pass  # silence HTTP ping logs


def start_health_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), _HealthHandler)
    print(f"🌐 Health-check server listening on port {port}")
    server.serve_forever()


if __name__ == "__main__":
    # Start the HTTP health server in a background thread so Render sees an open port
    threading.Thread(target=start_health_server, daemon=True).start()

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).concurrent_updates(True).build()
    app.add_handler(CallbackQueryHandler(on_choice, pattern=r"^c\|"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_request))
    print("✅ Bot is online and running...")
    app.run_polling(drop_pending_updates=True)