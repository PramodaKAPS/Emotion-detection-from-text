# Emotion Detection with DistilBERT

A modular emotion detection system using DistilBERT trained on the GoEmotions dataset.

## Features

- 10 emotion classification (anger, sadness, joy, disgust, fear, surprise, gratitude, remorse, curiosity, neutral)
- Modular, reusable code structure
- Google Colab optimized
- Class balancing with oversampling
- Interactive inference interface

## Quick Start in Google Colab

1. **Install dependencies and restart runtime:**
from setup import install_requirements
install_requirements()

Then restart runtime: Runtime > Restart runtime


2. **Run training:**
from main_colab import main
main()


3. **Use for inference:**
from inference import interactive_emotion_detection

model_path = "/content/drive/MyDrive/emotion_model"
emotions_list = ["anger", "sadness", "joy", "disgust", "fear", "surprise", "gratitude", "remorse", "curiosity", "neutral"]

interactive_emotion_detection(model_path, emotions_list)


## Project Structure

- `setup.py`: Environment setup and dependency management
- `data_utils.py`: Data loading, filtering, and preprocessing
- `model_utils.py`: Model creation, training, and evaluation utilities
- `train.py`: Main training pipeline
- `inference.py`: Inference utilities and interactive interface
- `main_colab.py`: Google Colab entry point
- `requirements.txt`: Package dependencies

## Training Configuration

- **Model**: DistilBERT (distilbert-base-uncased)
- **Dataset**: GoEmotions (simplified version)
- **Training samples**: 800 (customizable)
- **Epochs**: 1 (optimized for ~1 hour training)
- **Batch size**: 16
- **Learning rate**: 2e-5

## Usage Examples

### Training
from train import train_emotion_model

train_emotion_model(
cache_dir="/content/huggingface_cache",
save_path="/content/drive/MyDrive/emotion_model",
selected_emotions=["anger", "joy", "sadness", "fear", "surprise"],
num_train=800,
num_epochs=1
)


### Inference
from inference import EmotionDetector

detector = EmotionDetector(model_path, emotions_list)
result = detector.predict_emotion_with_confidence("I'm so happy today!")
print(result)


## License

MIT License
