# 🎬 Movie Review Sentiment Predictor

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Accuracy](https://img.shields.io/badge/Accuracy-88.7%25-brightgreen)
![Flask](https://img.shields.io/badge/Flask-2.3.3-lightgrey)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

A machine learning web application that predicts whether a movie review is **positive** or **negative** using **TF‑IDF** and **Logistic Regression**. The model is deployed as a lightweight **Flask** web app and can be run locally or inside a **Docker** container.

---

## 📌 Overview

- **Problem**: Manually reading thousands of movie reviews to gauge public sentiment is time‑consuming. An automated classifier can provide instant feedback.  
- **Solution**: A Logistic Regression model trained on 50,000 IMDB reviews, achieving **88.7% accuracy**.  
- **Deployment**: The model is served via a simple HTML form + Flask API, containerized with Docker for easy replication.

---

## 🚀 Features

- **Text Preprocessing**: HTML tag removal, punctuation stripping, lowercase conversion.  
- **Feature Extraction**: TF‑IDF vectorization (5,000 most frequent terms, English stop words removed).  
- **Classification**: Logistic Regression (scikit‑learn).  
- **Web Interface**: User‑friendly form to type or paste a review, get an instant sentiment prediction.  
- **Docker Support**: Ready‑to‑run container with all dependencies.

---

## 📊 Model Performance

Evaluated on a hold‑out test set of 10,000 reviews (20% of the dataset).

| Metric      | Negative (0) | Positive (1) | Average |
|-------------|--------------|--------------|---------|
| Precision   | 0.90         | 0.87         | 0.89    |
| Recall      | 0.87         | 0.91         | 0.89    |
| F1‑score    | 0.88         | 0.89         | 0.89    |

**Overall accuracy**: **0.887**

---

## 📈 Exploratory Data Analysis (EDA)
Key insights from the training notebook:

Balanced dataset: 25,000 positive & 25,000 negative reviews.

Word count distribution: Positive reviews tend to be slightly longer.

Word clouds:

Positive reviews: great, best, amazing, wonderful.

Negative reviews: worst, awful, boring, terrible.

TF‑IDF features: The most discriminative terms align with sentiment.
