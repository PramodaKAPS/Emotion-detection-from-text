from infer import load_model_tokenizer, predict_emotion

model_path = "/content/drive/MyDrive/emotion_model"
emotions = ["anger", "sadness", "joy", "disgust", "fear", "surprise", "gratitude", "remorse", "curiosity", "neutral"]

tokenizer, model = load_model_tokenizer(model_path)

while True:
    text = input("Enter a sentence to detect its emotion (or just press Enter to exit): ")
    if text.strip() == "":
        print("Exiting.")
        break
    predicted_emotion = predict_emotion(text, tokenizer, model, emotions)
    print(f"The detected emotion is: {predicted_emotion}")
