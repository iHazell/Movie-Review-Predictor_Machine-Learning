from flask import Flask, render_template, request, jsonify
import joblib
import re

app = Flask(__name__)


model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')


def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_review = data['review']  
    
    
    cleaned_review = clean_text(user_review)
    vectorized_review = vectorizer.transform([cleaned_review])
    
    
    prediction = model.predict(vectorized_review)[0]
    
    
    sentiment_result = "Positive 😊" if prediction == 1 else "Negative 😡"
    
    return jsonify({'prediction': sentiment_result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)