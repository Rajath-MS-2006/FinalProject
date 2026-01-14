# 📊 Real-Time Industry Insight & Strategic Intelligence System  
**Final Project – Infosys Springboard Internship 6.0 (B3)**

---

## 🔍 Project Overview

The **Real-Time Industry Insight & Strategic Intelligence System** is an intelligent analytics platform that collects real-time industry news, analyzes trends using NLP and AI, forecasts future sentiment patterns, visualizes insights, and sends automated alerts.

It converts raw news data into **actionable strategic intelligence** for data-driven decision making.

---

## 🚀 Features

- Real-time news extraction & preprocessing  
- Sentiment analysis using **NLTK**  
- AI-based contextual insight generation  
- Trend forecasting using **Prophet**  
- Automated **Slack alerts**  
- Interactive **Flask web dashboard**  
- Data visualization (word clouds, timelines, sentiment plots)

---

## 🛠 Tech Stack

| Layer | Technologies |
|-----|-------------|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, Flask |
| NLP | NLTK |
| AI | Gemini API |
| Forecasting | Prophet |
| Alerts | Slack Webhooks |
| Visualization | Matplotlib, WordCloud |
| Data Source | NewsAPI |

---

## 🧠 System Workflow

1. Collects real-time news using **NewsAPI**
2. Cleans and analyzes sentiment using **NLTK**
3. Generates AI insights using **Gemini**
4. Stores processed data into CSV datasets
5. Forecasts trends using **Prophet**
6. Sends automated alerts to Slack
7. Visualizes insights on the web dashboard

---

## 📂 Project Structure

```text
FinalProject/
├── app.py
├── data_fetch_and_analyse.py
├── forecast_and_slack.py
├── data/
│   ├── raw.csv
│   └── analyzed.csv
├── plots/
│   ├── distribution.png
│   ├── prophet_sentiment_forecast.png
│   ├── timeline.png
│   └── wordcloud.png
├── templates/
│   ├── index.html
│   └── data_view.html
└── README.md
```

---

## ⚙️ Installation & Setup

```bash
git clone https://github.com/Rajath-MS-2006/FinalProject.git
cd FinalProject
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

NEWS_API_KEY=your_newsapi_key
GEMINI_API_KEY=your_gemini_key
SLACK_WEBHOOK_URL=your_slack_webhook
```

---

## ▶️ Run the Application

```bash
python app.py
```

Open in browser: http://localhost:5000

---

## 👨‍💻 Team Contribution

- Technical Team Member  
- Implemented NLP, forecasting & Slack alert systems  
- Designed and integrated Flask dashboard  
- Delivered final working system  

---

## 🎓 Internship

Infosys Springboard Internship 6.0 (B3)

---

## ⭐ Future Enhancements

- Database integration (PostgreSQL / MongoDB)  
- Live API dashboards  
- Multi-platform alert systems  
- Advanced forecasting models  
- User authentication  
