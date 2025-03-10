QA-BOT 🛠️🎤
A real-time voice agent evaluation system built for HackSpirit 6.0

Overview
QA-BOT is a hackathon project developed for HackSpirit 6.0, a 24-hour hackathon. The bot analyzes and evaluates the real-time performance of voice agents by processing their interactions, identifying key metrics (such as accuracy, sentiment, and responsiveness), and providing actionable insights to enhance customer service quality.

This project leverages Python Whisper and various Python libraries to achieve accurate and efficient real-time analysis. we are unable to complete all the features requried 
in problem statement and we have to train the model more to get precise output.

Problem Statement
Description
Develop a QA-BOT that processes and evaluates voice agents' real-time interactions to measure performance metrics and improve customer service.

Input:
📢 Real-time audio data from voice agents’ customer interactions.
📝 Call transcripts & metadata (sentiment, keywords).
📊 Predefined performance metrics (tone, accuracy).
Output:
✅ Performance scores & insights for each voice agent.
🚨 Alerts for critical issues (e.g., negative comments).
Tech Stack
Programming Language: Python 🐍
Speech Recognition: OpenAI Whisper
Libraries Used:
whisper - For speech-to-text transcription
pydub - For audio processing
nltk - For sentiment analysis
matplotlib - For visualization
scipy - For audio signal processing
spacy - For keyword extraction

How It Works
Real-time audio processing 📡
Captures and transcribes customer-agent interactions using Whisper.
Sentiment and Tone Analysis 🎭
Analyzes tone, sentiment, and keywords using NLP techniques.
Alert System 🚨
Flags issues like negative sentiment, silence detection, or incorrect responses.
Dashboard/Visualization 📈
Provides actionable insights and detailed reports.
Setup & Installation
Prerequisites
Make sure you have Python 3.8+ installed. Then install the required dependencies:

![QA-BOT UI](https://raw.githubusercontent.com/asheesh109/QA-BOT/main/chatbot_img.jpg) 

Future Enhancements 🚀
Integration with AI-powered chatbots
Multi-language support for sentiment analysis
Real-time dashboard with interactive graphs
Advanced deep-learning models for better accuracy

License
This project is licensed under the MIT License.
