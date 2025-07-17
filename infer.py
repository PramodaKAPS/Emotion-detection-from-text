import numpy as np
from transformers import AutoTokenizer, TFDistilBertForSequenceClassification

def load_model_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

def predict_emotion(text, tokenizer, model, emotions):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
    logits = model(inputs).logits
    prediction = np.argmax(logits, axis=1)[0]
    return emotions[prediction]
