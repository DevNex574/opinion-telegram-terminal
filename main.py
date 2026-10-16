"""
====================================================================
 OPINION Markets Bot — unofficial Telegram analytics bot
 for the OPINION prediction market (https://opinion.trade)

 Built for the Opinion Builders Program.
 Stack: Python 3.11+ | aiogram 3.x | httpx | python-dotenv
====================================================================
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
from datetime import datetime, timezone
from typing import Any

import httpx
from aiogram import Bot, Dispatcher, F, Router, html
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    ErrorEvent,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
)
from dotenv import load_dotenv

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("opinion-bot")

# ----------------------------------------------------------------------
# Configuration (secrets come ONLY from the environment / .env file)
# ----------------------------------------------------------------------
load_dotenv()

BOT_TOKEN: str = os.getenv("BOT_TOKEN", "").strip()
OPINION_API_KEY: str = os.getenv("OPINION_API_KEY", "").strip()
OPINION_API_BASE_URL: str = os.getenv("OPINION_API_BASE_URL", "https://openapi.opinion.trade").strip().rstrip("/")
OPINION_WEB_URL: str = os.getenv("OPINION_WEB_URL", "https://opinion.trade").strip().rstrip("/")
REF_CODE: str = os.getenv("REF_CODE", "").strip()
REQUEST_TIMEOUT: float = float(os.getenv("REQUEST_TIMEOUT", "15"))

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is not set. Create a .env file (see .env.example).")

# Behaviour tuning
HOT_MARKETS_COUNT = 5
ANALYTICS_MARKETS_SCAN = 10
ANALYTICS_RESULTS = 5
SEARCH_MAX_PAGES = 5
SEARCH_RESULTS = 5
MAX_RETRIES = 4

CLOSED_STATUSES = {"resolved", "cancelled", "canceled", "paused"}

# Menu buttons
BTN_HOT = "🔥 Hot Markets"
BTN_ANALYTICS = "📈 Smart Analytics"
BTN_SEARCH = "🔍 Market Search"
BTN_ABOUT = "ℹ️ About OPINION"


# ----------------------------------------------------------------------
# Opinion Open API client (async, fault-tolerant)
# Docs: https://docs.opinion.trade/developer-guide/opinion-open-api
# ----------------------------------------------------------------------
class OpinionAPIError(RuntimeError):
    """Raised when the Opinion Open API cannot fulfil a request."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _backoff(attempt: int) -> float:
    """Exponential backoff with jitter, capped at 20 s."""
    return min(2 ** attempt + random.random(), 20.0)


class OpinionClient:
    """Thin async client for the Opinion OpenAPI REST endpoints."""

    def __init__(self, base_url: str, api_key: str | None, web_url: str, ref_code: str) -> None:
        headers = {"Accept": "application/json", "User-Agent": "opinion-markets-bot/1.0"}
        if api_key:
            headers["apikey"] = api_key
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=5.0),
            follow_redirects=True,
        )
        self._web = web_url
        self._ref = ref_code

    async def aclose(self) -> None:
        await self._http.aclose()

    # -- low level ----------------------------------------------------
    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = await self._http.request(method, path, **kwargs)
            except httpx.TimeoutException as exc:
                last_exc = exc
                wait = _backoff(attempt)
                log.warning("Timeout %s %s (attempt %d/%d), retry in %.1fs", method, path, attempt, MAX_RETRIES, wait)
                await asyncio.sleep(wait)
                continue
            except httpx.HTTPError as exc:
                last_exc = exc
                wait = _backoff(attempt)
                log.warning("Network error %s %s: %s (attempt %d/%d)", method, path, exc, attempt, MAX_RETRIES)
                await asyncio.sleep(wait)
                continue

            if resp.status_code == 429:  # rate limited — respect Retry-After
                retry_after = float(resp.headers.get("Retry-After", "2") or 2)
                log.warning("Rate limited (429) on %s %s, sleeping %.1fs", method, path, retry_after)
                await asyncio.sleep(retry_after)
                continue
            if resp.status_code >= 500:  # server side — backoff and retry
                wait = _backoff(attempt)
                log.warning("API %d on %s %s, retry in %.1fs", resp.status_code, method, path, wait)
                await asyncio.sleep(wait)
                continue
            if resp.status_code == 401:
                raise OpinionAPIError("Invalid OPINION_API_KEY (401 Unauthorized).", status=401)
            if resp.status_code >= 400:
                raise OpinionAPIError(f"API HTTP {resp.status_code}: {resp.text[:200]}", status=resp.status_code)

            try:
                payload = resp.json()
            except ValueError as exc:
                raise OpinionAPIError(f"Invalid JSON in API response: {exc}") from exc
            if isinstance(payload, dict) and payload.get("code", 0) != 0:
                raise OpinionAPIError(f"API error {payload.get('code')}: {payload.get('msg')}")
            return payload if isinstance(payload, dict) else {"result": payload}

        raise OpinionAPIError(f"Request failed after {MAX_RETRIES} attempts: {last_exc}")

    # -- high level ---------------------------------------------------
    async def get_markets(
        self,
        *,
        page: int = 1,
        limit: int = 20,
        sort_by: int | None = None,
        market_type: int = 2,  # 0=Binary, 1=Categorical, 2=All
    ) -> list[dict[str, Any]]:
        """GET /openapi/market — paginated market list.

        sortBy: 1=new, 2=ending soon, 3=volume desc, 5=volume24h desc, 7=volume7d desc
        """
        params: dict[str, Any] = {"page": page, "limit": limit, "marketType": market_type}
        if sort_by is not None:
            params["sortBy"] = sort_by
        payload = await self._request("GET", "/openapi/market", params=params)
        result = payload.get("result") or {}
        return list(result.get("list") or [])

    async def get_orderbook(self, token_id: str) -> dict[str, Any]:
        """GET /openapi/token/orderbook?token_id=... — bids/asks depth."""
        payload = await self._request("GET", "/openapi/token/orderbook", params={"token_id": token_id})
        return payload.get("result") or {}

    async def get_latest_price(self, token_id: str) -> dict[str, Any]:
        """GET /openapi/token/latest-price?token_id=... — last trade price."""
        payload = await self._request("GET", "/openapi/token/latest-price", params={"token_id": token_id})
        return payload.get("result") or {}

    def market_url(self, market_id: Any) -> str:
        url = f"{self._web}/market/{market_id}"
        if self._ref:
            url += f"?ref={self._ref}"
        return url


# ----------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------
def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt_money(value: float) -> str:
    if value >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    if value >= 1_000:
        return f"${value / 1_000:,.1f}K"
    return f"${value:,.2f}"


def fmt_pct(price: float | None) -> str:
    return f"{price * 100:.0f}%" if price is not None else "—"


def fmt_cutoff(ts: Any) -> str:
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return "—"


def is_active_market(m: dict[str, Any]) -> bool:
    return str(m.get("statusEnum") or "").strip().lower() not in CLOSED_STATUSES


def api_error_text(exc: OpinionAPIError) -> str:
    return (
        "⚠️ <b>Could not reach the Opinion API.</b>\n\n"
        f"<code>{html.quote(str(exc))}</code>\n\n"
        "Please try again in a moment."
    )


def extract_price(result: dict[str, Any]) -> float | None:
    """Tolerant extraction of a price from the /token/latest-price response."""
    if not isinstance(result, dict):
        return None
    for key in ("price", "latestPrice", "lastPrice"):
        if key in result:
            p = _f(result[key], -1.0)
            return p if 0 <= p <= 1 else None
    data = result.get("data")
    if isinstance(data, dict):
        return extract_price(data)
    return None


# ----------------------------------------------------------------------
# Keyboards
# ----------------------------------------------------------------------
def main_menu_kb() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=BTN_HOT), KeyboardButton(text=BTN_ANALYTICS)],
            [KeyboardButton(text=BTN_SEARCH), KeyboardButton(text=BTN_ABOUT)],
        ],
        resize_keyboard=True,
    )


def market_bet_kb(client: OpinionClient, m: dict[str, Any]) -> InlineKeyboardMarkup:
    mid = m.get("marketId", "")
    url = client.market_url(mid)
    yes_label = str(m.get("yesLabel") or "YES")
    no_label = str(m.get("noLabel") or "NO")
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text=f"⚡ Bet {yes_label}", url=url),
                InlineKeyboardButton(text=f"🛡 Bet {no_label}", url=url),
            ],
            [InlineKeyboardButton(text="🔗 Open market on OPINION", url=url)],
        ]
    )


# ----------------------------------------------------------------------
# Market card renderer
# ----------------------------------------------------------------------
def market_card(
    m: dict[str, Any],
    *,
    rank: int | None = None,
    yes_price: float | None = None,
) -> str:
    title = html.quote(str(m.get("marketTitle") or "Untitled market"))
    yes_label = html.quote(str(m.get("yesLabel") or "YES"))
    no_label = html.quote(str(m.get("noLabel") or "NO"))
    vol24 = _f(m.get("volume24h"))
    vol_total = _f(m.get("volume"))
    cutoff = fmt_cutoff(m.get("cutoffAt"))

    header = f"🔥 <b>Top-{rank} by 24h volume</b>\n\n" if rank else ""
    no_price = (1 - yes_price) if yes_price is not None else None

    lines = [
        f"{header}❓ <b>{title}</b>\n",
        f"{yes_label}: <b>{fmt_pct(yes_price)}</b>   •   {no_label}: <b>{fmt_pct(no_price)}</b>",
        f"💰 24h volume: <b>{fmt_money(vol24)}</b>   •   Total: {fmt_money(vol_total)}",
        f"📅 Ends: {cutoff}",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Router & handlers
# ----------------------------------------------------------------------
router = Router()


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    text = (
        "👋 Welcome to <b>OPINION Markets Bot</b>!\n\n"
        "I stream the hottest prediction markets from "
        f"{html.link('OPINION', 'https://opinion.trade')} straight into your "
        "Telegram: live outcome probabilities, trading volumes, volume spikes "
        "and order book anomalies.\n\n"
        "Pick a section below 👇"
    )
    await message.answer(text, reply_markup=main_menu_kb(), disable_web_page_preview=True)


# ------------------------- 🔥 Hot markets -------------------------
@router.message(F.text == BTN_HOT)
async def hot_markets(message: Message, client: OpinionClient) -> None:
    wait = await message.answer("⏳ Loading the hottest markets…")
    try:
        # sortBy=5 → volume24h desc (per Opinion OpenAPI docs)
        markets = await client.get_markets(limit=20, sort_by=5, market_type=2)
    except OpinionAPIError as exc:
        await wait.edit_text(api_error_text(exc))
        return

    active = [m for m in markets if is_active_market(m)][:HOT_MARKETS_COUNT]
    if not active:
        await wait.edit_text("😕 No active markets right now. Check back soon!")
        return

    # Enrich with live YES prices (concurrent, semaphore + polite delay
    # to stay within the public 5 req/s rate limit).
    prices: dict[Any, float] = {}
    sem = asyncio.Semaphore(3)

    async def _price(m: dict[str, Any]) -> None:
        tid = str(m.get("yesTokenId") or "")
        if not tid:
            return
        async with sem:
            try:
                res = await client.get_latest_price(tid)
            except OpinionAPIError as exc:
                log.info("latest-price failed for %s: %s", tid, exc)
                return
            await asyncio.sleep(0.25)
        p = extract_price(res)
        if p is not None:
            prices[m.get("marketId")] = p

    await asyncio.gather(*[_price(m) for m in active])

    await wait.delete()
    for rank, m in enumerate(active, start=1):
        await message.answer(
            market_card(m, rank=rank, yes_price=prices.get(m.get("marketId"))),
            reply_markup=market_bet_kb(client, m),
            disable_web_page_preview=True,
        )


# ------------------- 📈 Smart analytics & signals -------------------
def _book_stats(book: dict[str, Any] | None) -> tuple[float | None, float | None, float]:
    """Return (mid_price, spread, depth_imbalance).

    mid_price — midpoint of the best bid/ask of the YES token book
    spread    — best_ask - best_bid (in price points)
    imbalance — (bid_depth - ask_depth) / (bid_depth + ask_depth), [-1, 1]
    """
    if not book:
        return None, None, 0.0
    bids = [(_f(b.get("price")), _f(b.get("size"))) for b in book.get("bids") or []]
    asks = [(_f(a.get("price")), _f(a.get("size"))) for a in book.get("asks") or []]
    if not bids or not asks:
        return None, None, 0.0
    best_bid = max(p for p, _ in bids)
    best_ask = min(p for p, _ in asks)
    bid_depth = sum(s for _, s in bids)
    ask_depth = sum(s for _, s in asks)
    spread = max(best_ask - best_bid, 0.0)
    imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth) if (bid_depth + ask_depth) else 0.0
    return (best_bid + best_ask) / 2, spread, imbalance


@router.message(F.text == BTN_ANALYTICS)
async def smart_analytics(message: Message, client: OpinionClient) -> None:
    wait = await message.answer("📊 Scanning order books and volume flows…")
    try:
        markets = await client.get_markets(limit=20, sort_by=5, market_type=2)
    except OpinionAPIError as exc:
        await wait.edit_text(api_error_text(exc))
        return

    active = [m for m in markets if is_active_market(m)][:ANALYTICS_MARKETS_SCAN]
    if not active:
        await wait.edit_text("😕 No active markets to analyse.")
        return

    sem = asyncio.Semaphore(4)

    async def _analyze(m: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        tid = str(m.get("yesTokenId") or "")
        book: dict[str, Any] | None = None
        if tid:
            async with sem:
                try:
                    book = await client.get_orderbook(tid)
                except OpinionAPIError as exc:
                    log.info("orderbook failed for %s: %s", tid, exc)
                await asyncio.sleep(0.25)
        return m, book

    results = await asyncio.gather(*[_analyze(m) for m in active])

    signals: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for m, book in results:
        vol24 = _f(m.get("volume24h"))
        vol7d = _f(m.get("volume7d"))
        avg_daily = vol7d / 7 if vol7d > 0 else 0.0
        spike = vol24 / avg_daily if avg_daily > 0 else 0.0  # 1.0 ≈ average day
        mid, spread, imbalance = _book_stats(book)
        # Score: volume activity (spike) + buy/sell pressure from the book
        score = spike + abs(imbalance) * 3.0
        signals.append((score, m, {"mid": mid, "spread": spread, "imbalance": imbalance, "spike": spike}))

    signals.sort(key=lambda s: s[0], reverse=True)
    await wait.delete()
    await message.answer(
        f"📈 <b>Smart Analytics:</b> top-{min(ANALYTICS_RESULTS, len(signals))} signals "
        "by volume spikes and order book imbalance:"
    )
    for score, m, s in signals[:ANALYTICS_RESULTS]:
        title = html.quote(str(m.get("marketTitle") or "Untitled market"))
        imb = s["imbalance"]
        side = "buyers 🟢" if imb > 0.05 else "sellers 🔴" if imb < -0.05 else "neutral ⚪"
        strength = "🟢 strong" if score >= 4 else "🟡 moderate" if score >= 2 else "⚪ weak"
        spread_str = f"{s['spread']:.3f}" if s["spread"] is not None else "—"
        lines = [
            f"❓ <b>{title}</b>\n",
            f"💵 YES mid price: <b>{fmt_pct(s['mid'])}</b>   •   Spread: {spread_str}",
            f"⚖️ Book imbalance: <b>{imb:+.0%}</b> towards {side}",
            f"🚀 24h activity vs average: <b>×{s['spike']:.1f}</b>",
            f"🎯 Signal: {strength}",
        ]
        await message.answer(
            "\n".join(lines),
            reply_markup=market_bet_kb(client, m),
            disable_web_page_preview=True,
        )


# ------------------------- 🔍 Market search -------------------------
class SearchStates(StatesGroup):
    waiting_query = State()


@router.message(F.text == BTN_SEARCH)
async def search_start(message: Message, state: FSMContext) -> None:
    await state.set_state(SearchStates.waiting_query)
    await message.answer(
        "🔍 <b>Market search</b>\n\nType a keyword — e.g. "
        "<i>Bitcoin</i>, <i>Election</i> or <i>Crypto</i>:\n\n"
        "Cancel: /cancel"
    )


@router.message(Command("cancel"))
async def search_cancel(message: Message, state: FSMContext) -> None:
    if await state.get_state() is not None:
        await state.clear()
        await message.answer("❌ Search cancelled.", reply_markup=main_menu_kb())


@router.message(SearchStates.waiting_query, F.text)
async def search_run(message: Message, state: FSMContext, client: OpinionClient) -> None:
    query = (message.text or "").strip()
    if len(query) < 2:
        await message.answer("⚠️ Query is too short — at least 2 characters. Or /cancel.")
        return
    await state.clear()

    wait = await message.answer(f"🔍 Searching markets for “{html.quote(query)}”…")
    needle = query.lower()
    found: list[dict[str, Any]] = []
    try:
        for page in range(1, SEARCH_MAX_PAGES + 1):
            batch = await client.get_markets(page=page, limit=20, sort_by=3, market_type=2)
            if not batch:
                break
            for m in batch:
                if needle in str(m.get("marketTitle") or "").lower() and is_active_market(m):
                    found.append(m)
            await asyncio.sleep(0.25)  # stay within the public rate limit
    except OpinionAPIError as exc:
        await wait.edit_text(api_error_text(exc))
        return

    await wait.delete()
    if not found:
        await message.answer(
            f"😕 Nothing found for “{html.quote(query)}”.\nTry a different keyword."
        )
        return

    found.sort(key=lambda m: _f(m.get("volume24h")), reverse=True)
    await message.answer(
        f"✅ Found <b>{len(found)}</b> market(s). Showing top-{min(SEARCH_RESULTS, len(found))} by 24h volume:"
    )
    for m in found[:SEARCH_RESULTS]:
        await message.answer(
            market_card(m, yes_price=None),
            reply_markup=market_bet_kb(client, m),
            disable_web_page_preview=True,
        )


# ------------------------- ℹ️ About -------------------------
@router.message(F.text == BTN_ABOUT)
async def about_platform(message: Message) -> None:
    text = (
        "ℹ️ <b>About OPINION</b>\n\n"
        "OPINION is a decentralized prediction market on BNB Chain: "
        "trade the outcomes of real-world events, where contract prices "
        "reflect the market-implied probability of each outcome.\n\n"
        f"• {html.link('Platform website', 'https://opinion.trade')}\n"
        f"• {html.link('Opinion Open API (docs)', 'https://docs.opinion.trade/developer-guide/opinion-open-api')}\n"
        f"• {html.link('Opinion Builders Program', 'https://docs.opinion.trade/developer-guide/opinion-builders-program')}\n"
        f"• {html.link('Referral program', 'https://docs.opinion.trade/incentive-plans/referral-program')}\n\n"
        "⚠️ Trading prediction contracts involves risk. Never invest more "
        "than you can afford to lose."
    )
    await message.answer(text, disable_web_page_preview=True)


# ------------------------- Global error handler -------------------------
@router.error()
async def global_error_handler(event: ErrorEvent) -> None:
    log.exception("Unhandled exception: %s", event.exception)
    msg = event.update.message if event.update else None
    if isinstance(msg, Message):
        await msg.answer(
            "⚠️ Internal bot error. Please try again or press /start.",
            reply_markup=main_menu_kb(),
        )


# ----------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------
async def main() -> None:
    client = OpinionClient(
        base_url=OPINION_API_BASE_URL,
        api_key=OPINION_API_KEY or None,
        web_url=OPINION_WEB_URL,
        ref_code=REF_CODE,
    )
    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)
    dp["client"] = client  # dependency injection for handlers

    log.info("Starting OPINION Markets Bot… (api=%s, ref=%s)", OPINION_API_BASE_URL, REF_CODE or "—")
    try:
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot)
    finally:
        await client.aclose()
        await bot.session.close()
        log.info("Bot stopped.")


if __name__ == "__main__":
    asyncio.run(main())
