# 📈 TradeAI – Smart Stock Signal App

AI-powered trading signal app covering NSE, BSE, NASDAQ, NYSE.

## Features
- 💰 Budget Advisor – Enter budget, get stock picks (All-In or Split)
- 📊 Chart Analysis – Upload chart image → AI gives BUY/SELL/HOLD
- 🔍 Stock Screener – Filter by sector, market, horizon

## How to Run

### Option 1: Just open the file (Easiest)
Double click `index.html` → opens in browser → works directly!

### Option 2: Local server (Recommended)
```bash
# If you have Python installed
python -m http.server 8080

# Then open browser → http://localhost:8080
```

## Tech Stack
- Pure HTML + CSS + JavaScript (no frameworks needed)
- Claude AI API (claude-sonnet-4) for analysis
- No backend required

## Notes
- Internet connection required (calls Claude API)
- Chart analysis works best with clear candlestick chart screenshots
- All analysis is for educational purposes only, not financial advice
