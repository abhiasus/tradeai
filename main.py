from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import requests
import json
import base64
import numpy as np
from datetime import datetime, timedelta
import ta
import uvicorn

app = FastAPI(title="TradeAI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"  # change to llama3 if you want

# ─── Indian stocks (NSE) ───────────────────────────────────────────
INDIAN_STOCKS = {
    "Technology": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
    "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS"],
    "Pharma": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS"],
    "Energy": ["RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS"],
    "FMCG": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS"],
    "Auto": ["MARUTI.NS", "TATAMOTORS.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS"],
}

# ─── US Stocks ─────────────────────────────────────────────────────
US_STOCKS = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "NVDA", "META"],
    "Banking": ["JPM", "BAC", "GS", "MS", "WFC"],
    "Pharma": ["JNJ", "PFE", "MRK", "ABBV"],
    "Energy": ["XOM", "CVX", "COP", "SLB"],
    "FMCG": ["PG", "KO", "PEP", "WMT"],
    "Auto": ["TSLA", "F", "GM", "TM"],
}


def ask_ollama(prompt: str) -> str:
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 500}
        }, timeout=60)
        return response.json().get("response", "")
    except Exception as e:
        return f"Ollama error: {str(e)}"


def get_stock_data(symbol: str):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="3mo")
        if hist.empty:
            return None
        info = ticker.info
        return {"history": hist, "info": info}
    except:
        return None


def calculate_signals(hist):
    close = hist["Close"]
    volume = hist["Volume"]

    # RSI
    rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]

    # MACD
    macd_ind = ta.trend.MACD(close)
    macd = macd_ind.macd().iloc[-1]
    macd_signal = macd_ind.macd_signal().iloc[-1]

    # Moving Averages
    ma20 = close.rolling(20).mean().iloc[-1]
    ma50 = close.rolling(50).mean().iloc[-1]
    current_price = close.iloc[-1]
    prev_price = close.iloc[-5]

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close)
    bb_upper = bb.bollinger_hband().iloc[-1]
    bb_lower = bb.bollinger_lband().iloc[-1]

    # Score system
    score = 0
    reasons = []

    # RSI scoring
    if rsi < 30:
        score += 3
        reasons.append(f"RSI={rsi:.1f} (oversold → bullish)")
    elif rsi < 45:
        score += 1
        reasons.append(f"RSI={rsi:.1f} (slightly oversold)")
    elif rsi > 70:
        score -= 3
        reasons.append(f"RSI={rsi:.1f} (overbought → bearish)")
    elif rsi > 60:
        score -= 1
        reasons.append(f"RSI={rsi:.1f} (slightly overbought)")
    else:
        reasons.append(f"RSI={rsi:.1f} (neutral)")

    # MACD scoring
    if macd > macd_signal:
        score += 2
        reasons.append("MACD bullish crossover")
    else:
        score -= 2
        reasons.append("MACD bearish crossover")

    # MA scoring
    if current_price > ma20 > ma50:
        score += 2
        reasons.append("Price above MA20 & MA50 (uptrend)")
    elif current_price < ma20 < ma50:
        score -= 2
        reasons.append("Price below MA20 & MA50 (downtrend)")

    # Price momentum
    change_pct = ((current_price - prev_price) / prev_price) * 100
    if change_pct > 3:
        score += 1
        reasons.append(f"Strong momentum +{change_pct:.1f}% (5d)")
    elif change_pct < -3:
        score -= 1
        reasons.append(f"Weak momentum {change_pct:.1f}% (5d)")

    # Bollinger
    if current_price < bb_lower:
        score += 1
        reasons.append("Near lower Bollinger band (potential bounce)")
    elif current_price > bb_upper:
        score -= 1
        reasons.append("Near upper Bollinger band (potential reversal)")

    # Final signal
    if score >= 3:
        signal = "BUY"
        confidence = min(95, 60 + score * 5)
    elif score <= -3:
        signal = "SELL"
        confidence = min(95, 60 + abs(score) * 5)
    else:
        signal = "HOLD"
        confidence = 55

    return {
        "signal": signal,
        "confidence": int(confidence),
        "score": score,
        "rsi": round(rsi, 1),
        "macd": round(macd, 3),
        "ma20": round(ma20, 2),
        "ma50": round(ma50, 2),
        "current_price": round(current_price, 2),
        "change_5d": round(change_pct, 2),
        "reasons": reasons
    }


def allocate_budget(budget: float, stocks: list, strategy: str):
    buy_stocks = [s for s in stocks if s["signal"] == "BUY"]
    if not buy_stocks:
        buy_stocks = sorted(stocks, key=lambda x: x["confidence"], reverse=True)[:2]

    if strategy == "allin":
        best = sorted(buy_stocks, key=lambda x: x["confidence"], reverse=True)[0]
        best["allocation"] = budget
        best["allocation_pct"] = 100
        return [best]
    else:
        # Split by confidence weight
        total_conf = sum(s["confidence"] for s in buy_stocks[:4])
        result = []
        for s in buy_stocks[:4]:
            pct = round((s["confidence"] / total_conf) * 100)
            s["allocation"] = round((pct / 100) * budget, 2)
            s["allocation_pct"] = pct
            result.append(s)
        return result


# ─── ROUTES ────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("templates/index.html") as f:
        return f.read()


@app.post("/api/budget")
async def budget_advice(
    budget: float = Form(...),
    market: str = Form(...),
    strategy: str = Form(...),
    risk: str = Form(...)
):
    results = []

    # Pick stock universe
    if market == "india":
        universe = INDIAN_STOCKS
        currency = "₹"
    elif market == "us":
        universe = US_STOCKS
        currency = "$"
    else:
        universe = {**INDIAN_STOCKS, **US_STOCKS}
        currency = "₹/$"

    # Risk filter
    risk_sectors = {
        "low": ["Banking", "FMCG"],
        "medium": ["Technology", "Banking", "Pharma", "FMCG"],
        "high": ["Technology", "Auto", "Energy"]
    }
    sectors = risk_sectors.get(risk, list(universe.keys()))
    
    # Analyze stocks
    analyzed = []
    for sector in sectors:
        if sector not in universe:
            continue
        for sym in universe[sector][:2]:  # top 2 per sector
            data = get_stock_data(sym)
            if not data:
                continue
            signals = calculate_signals(data["history"])
            name = sym.replace(".NS", "").replace(".BO", "")
            exchange = "NSE" if ".NS" in sym else "BSE" if ".BO" in sym else "NASDAQ/NYSE"
            analyzed.append({
                "symbol": sym,
                "name": name,
                "exchange": exchange,
                "sector": sector,
                "price": signals["current_price"],
                "signal": signals["signal"],
                "confidence": signals["confidence"],
                "rsi": signals["rsi"],
                "change_5d": signals["change_5d"],
                "reasons": signals["reasons"][:2],
            })

    if not analyzed:
        return JSONResponse({"error": "Could not fetch stock data"}, status_code=500)

    # Allocate budget
    allocated = allocate_budget(budget, analyzed, strategy)

    # Ask Ollama for summary
    stock_summary = ", ".join([f"{s['name']} ({s['signal']})" for s in allocated])
    prompt = f"""You are a stock market analyst. Based on technical analysis, these stocks were selected: {stock_summary}.
Budget: {currency}{budget}, Strategy: {strategy}, Risk: {risk}.
Give a 2-3 sentence investment summary and key advice. Be concise and practical."""

    ai_summary = ask_ollama(prompt)

    return {
        "stocks": allocated,
        "ai_summary": ai_summary,
        "currency": currency,
        "total_analyzed": len(analyzed)
    }


@app.post("/api/chart")
async def analyze_chart(
    file: UploadFile = File(...),
    stock_name: str = Form("Unknown"),
    timeframe: str = Form("1 Month")
):
    img_bytes = await file.read()
    img_b64 = base64.b64encode(img_bytes).decode()

    prompt = f"""You are an expert technical analyst. A trader uploaded a stock chart for {stock_name} on {timeframe} timeframe.

Analyze the chart and provide:
1. Overall signal: BUY, SELL, or HOLD
2. Confidence percentage (50-90%)
3. Trend: BULLISH, BEARISH, or SIDEWAYS
4. Key patterns visible (e.g., Double Top, Head & Shoulders, Breakout, Support bounce)
5. Analysis (2-3 sentences)
6. Entry suggestion
7. Stop loss suggestion
8. Target suggestion

Respond in this exact JSON format only, no extra text:
{{
  "signal": "BUY",
  "confidence": 72,
  "trend": "BULLISH",
  "patterns": ["Breakout", "Higher Highs"],
  "analysis": "The chart shows a clear uptrend with higher highs and higher lows. Price has broken above key resistance. Volume is confirming the move.",
  "entry": "Buy near current price or on pullback to support",
  "stop_loss": "Below recent swing low",
  "target": "Next resistance level, approx 8-10% above entry"
}}"""

    response = ask_ollama(prompt)

    try:
        clean = response.strip()
        start = clean.find("{")
        end = clean.rfind("}") + 1
        data = json.loads(clean[start:end])
    except:
        data = {
            "signal": "HOLD",
            "confidence": 60,
            "trend": "SIDEWAYS",
            "patterns": ["Unable to parse chart details"],
            "analysis": response[:200] if response else "Could not analyze chart. Try a clearer image.",
            "entry": "Wait for clearer signal",
            "stop_loss": "Use 2-3% below entry",
            "target": "Use 1:2 risk-reward ratio"
        }

    return data


@app.post("/api/screener")
async def screener(
    sector: str = Form(...),
    market: str = Form(...),
    horizon: str = Form(...),
    signal_filter: str = Form(...)
):
    if market == "india":
        universe = INDIAN_STOCKS
    elif market == "us":
        universe = US_STOCKS
    else:
        universe = {**INDIAN_STOCKS, **US_STOCKS}

    sectors_to_scan = [sector] if sector != "any" else list(universe.keys())
    results = []

    for sec in sectors_to_scan:
        if sec not in universe:
            continue
        for sym in universe[sec]:
            data = get_stock_data(sym)
            if not data:
                continue
            signals = calculate_signals(data["history"])
            if signal_filter != "all" and signals["signal"].lower() != signal_filter:
                continue
            name = sym.replace(".NS", "").replace(".BO", "")
            exchange = "NSE" if ".NS" in sym else "NYSE/NASDAQ"
            currency = "₹" if ".NS" in sym or ".BO" in sym else "$"
            results.append({
                "name": name,
                "symbol": sym,
                "exchange": exchange,
                "sector": sec,
                "signal": signals["signal"],
                "confidence": signals["confidence"],
                "price": f"{currency}{signals['current_price']}",
                "rsi": signals["rsi"],
                "change_5d": signals["change_5d"],
                "reason": signals["reasons"][0] if signals["reasons"] else ""
            })

    results.sort(key=lambda x: x["confidence"], reverse=True)

    prompt = f"""Stock screener found {len(results)} stocks in {sector} sector for {horizon} horizon.
Top picks: {', '.join([r['name'] + ' (' + r['signal'] + ')' for r in results[:5]])}.
Give 1-2 sentence market context for these results."""

    market_note = ask_ollama(prompt)

    return {
        "results": results[:10],
        "total": len(results),
        "market_note": market_note
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
