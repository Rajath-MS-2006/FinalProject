# 📊 Real-Time Industry Insight & Strategic Intelligence System  
**Final Project – Infosys Springboard Internship 6.0 (B3)**

---

## 🔍 Project Overview

The **Real-Time Industry Insight & Strategic Intelligence System** is a full-stack intelligent analytics platform designed to collect real-time industry news, analyze trends using NLP and AI, forecast future patterns, and notify stakeholders through automated alerts.

This system helps organizations make **data-driven strategic decisions** by transforming unstructured news data into actionable intelligence.

---

## 🚀 Features

- Real-time news extraction using **NewsAPI**
- Text preprocessing and sentiment analysis using **NLTK**
- AI-powered contextual insight generation using **Gemini API**
- Time-series forecasting using **Facebook Prophet**
- Automated **Slack alerts** based on predicted trends
- Interactive web dashboard for visualization and control
- RESTful backend services built using **Flask**

---

## 🛠 Tech Stack

| Layer | Technologies |
|-----|-------------|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, Flask |
| NLP | NLTK |
| AI Integration | Gemini API |
| Forecasting | Prophet |
| Alerts | Slack Webhooks |
| Data Source | NewsAPI |

---

## 🧠 System Workflow

1. Fetches real-time news from **NewsAPI**
2. Cleans and processes text using **NLP**
3. Generates AI-based contextual insights via **Gemini**
4. Forecasts upcoming trends using **Prophet**
5. Sends automated alerts to **Slack**
6. Displays results on the web dashboard

---

## 📂 Project Structure

```bash
├── app.py
├── config.py
├── requirements.txt
├── static/
│   ├── css/
│   └── js/
├── templates/
│   └── index.html
├── modules/
│   ├── news_fetcher.py
│   ├── nlp_processor.py
│   ├── ai_analyzer.py
│   ├── forecast_engine.py
│   └── slack_alerts.py
└── README.md
---

##⚙️ Installation & Setup
1. Clone the repository
git clone https://github.com/Rajath-MS-2006/FinalProject.git
cd FinalProject

2. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# OR
source venv/bin/activate  # macOS / Linux

3. Install dependencies
pip install -r requirements.txt

4. Configure API Keys

Create a .env file:

NEWS_API_KEY=your_newsapi_key
GEMINI_API_KEY=your_gemini_key
SLACK_WEBHOOK_URL=your_slack_webhook

▶ Run the Application
python app.py


Open in browser:
http://localhost:5000

👨‍💻 Team Contribution

Served as a Technical Team Member

Designed system architecture

Implemented NLP, AI integration, forecasting & full-stack modules

Delivered final system during team presentation

🎓 Internship

Infosys Springboard Internship 6.0 (B3)
Project: Real-Time Industry Insight & Strategic Intelligence System

⭐ Future Enhancements

Live industry dashboards

Database integration (PostgreSQL/MongoDB)

User authentication

More ML forecasting models

Multi-platform alert integrations
