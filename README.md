#  Fake News Detector
This project is a simple Fake News Detection web application using Python, Flask, and JavaScript. It uses a machine learning model (TF-IDF + Logistic Regression) to classify whether a news headline is Fake or Real.

##  Features

- Train a model on a sample dataset or load an existing model
- Predict news authenticity using a simple web interface
- Frontend built with HTML/CSS + JavaScript
- Backend API using Flask
- Easily extensible with a larger dataset

##  Project Structure
fake-news-detector/
│
├── backend/
│ ├── app.py # Flask backend API
│ ├── model.pkl # Trained model
│ ├── vectorizer.pkl # TF-IDF vectorizer
│ └── train_model.py # Model training script
│
├── frontend/
│ ├── index.html # Main UI
│ └── script.js # JS to call API
│
├── .gitignore
└── README.md

