# ​ FinSage — Automated Stock Forecasting with Open‑Source LLMs

A full-stack, open‑source system for daily stock price prediction using:

- LSTM-based time series forecasting (PyTorch)
- Sentiment analysis on financial news using the Mistral model via **Ollama**
- Orchestration via **n8n** for data ingestion, model training, and automation
- **Flask** backend + **Streamlit** dashboard
- PostgreSQL for data storage
- Containerized orchestration via **Docker Compose**

---

## ​ Features

- Daily scraping of financial news and OHLCV data
- Sentiment scoring of headlines via local LLM
- Model retraining and next-day price prediction
- Interactive dashboard to view price history, sentiment analysis, and predictions

---

## ​ Project Structure

```
FinSage/
│
├── backend/
│   ├── app.py
│   ├── data_fetcher.py
│   ├── sentiment.py
│   ├── model.py
│   └── train.py
│
├── dashboard/
│   ├── app.py
│   └── requirements.txt
│
├── docker-compose.yml
├── init.sql
└── n8n_workflow.json
```

---

## ​ Prerequisites

- Docker & Docker Compose installed  
- Internet access for initial LLM download  
- Ports available: `5432`, `5000`, `5678`, `8501`

---

## ​ Installation & Usage

1. **Clone the repo:**
   ```bash
   git clone https://github.com/bysani2003/FinSage.git
   cd FinSage
   ```

2. **Start all services:**
   ```bash
   docker compose up --build
   ```

3. **First-time setup:**
   - Wait several minutes during the initial build as the Mistral model downloads
   - Watch logs for backend, n8n, and dashboard services

4. **Access services:**
   - n8n Workflow Editor: `http://localhost:5678`
   - Streamlit Dashboard: `http://localhost:8501`

5. **Import the workflow:**
   In n8n → "Workflows" → "Import from file" → select `n8n_workflow.json` → activate → click “Execute Workflow” once.

6. **Use the Dashboard:**
   - Enter a ticker symbol (e.g., AAPL)
   - View past close prices, recent news & sentiment scores, and the predicted next closing price

---

## ​ Architecture Overview

```text
[ n8n scheduler ]
         │
┌────────▼────────┐
│  /scrape-news   │ ← pulls articles, sentiment
├─────────────────┤
│  /fetch-stock   │ ← fetch daily OHLC data
├─────────────────┤
│  /train-model   │ ← trains LSTM, saves model
└─────────────────┘
         │
[ PostgreSQL Database ]
         │
[ Streamlit dashboard displays results and predictions ]
```

---

## ​ Technology Stack

| Component            | Technology             |
|---------------------|------------------------|
| Backend API         | Flask, Python          |
| News Fetching       | feedparser, newspaper3k |
| Sentiment Analysis  | Mistral LLM via Ollama |
| Time-Series Model   | PyTorch LSTM           |
| Workflow Orchestration | n8n                |
| Dashboard UI        | Streamlit + Plotly     |
| Database            | PostgreSQL             |
| Containerization    | Docker & Docker Compose |

---

## ​ Sample Workflow (AAPL)

1. Daily trigger in n8n → calls backend endpoints
2. `/scrape-news` → fetches and processes headlines
3. `/fetch-stock-data` → stores price data
4. `/train-model` → trains or updates LSTM + sentiment features
5. Dashboard reads from DB and saved model → shows charts & prediction

---

## ​ Optional Extensions

- Multiple tickers support in workflows
- Email/Telegram notifications on model update
- Sentiment from Reddit or Twitter
- Earnings report summarizer using Ollama
- Transformer-based forecasting models (e.g. TFT)

---

## ​ Contribution & License

- All components are **fully open source** (no paid APIs)
- Feel free to fork, adapt, or reuse this project
- Add your own visual enhancements, new model types, or automation paths

---

## ​ About

FinSage was built by Sujith Bysani Santhosh (*bysanisujith2003*). It integrates AI, automation, and time-series analysis to deliver transparent and extendable financial insights.

---

### ​ Connect

- GitHub: [bysani2003](https://github.com/bysani2003)  
- LinkedIn: [bysani2003](https://www.linkedin.com/in/sujith-bysani-santhosh-51b55020a/)
- Email: bysanisujith2003@gmail.com

