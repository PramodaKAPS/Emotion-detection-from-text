import os
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from data_utils import load_and_filter_goemotions, oversample_training_data, prepare_tokenized_datasets
from model_utils import create_tf_datasets, setup_model_and_optimizer, compile_and_train, save_model_and_tokenizer, create_rnn_model, evaluate_model

def train_emotion_model(cache_dir, save_path, emotions, num_train=5000, epochs=5, batch_size=16, learning_rate=2e-5, model_type="DistilBERT", input_length=100, vocab_size=10000, embedding_dim=128):
    print(f"Starting training for {model_type}...")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache directory ready: {cache_dir}")

    train_df, valid_df, test_df, sel_indices = load_and_filter_goemotions(cache_dir, emotions, num_train)
    oversampled_train_df = oversample_training_data(train_df)
    
    is_distilbert = model_type == "DistilBERT"
    
    mapping = {old: new for new, old in enumerate(sel_indices)}  # Label mapping for all models
    
    if is_distilbert:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", cache_dir=cache_dir)
        tokenized_train, tokenized_valid, tokenized_test = prepare_tokenized_datasets(tokenizer, oversampled_train_df, valid_df, test_df)
        tf_train, tf_val, tf_test = create_tf_datasets(tokenized_train, tokenized_valid, tokenized_test, tokenizer, sel_indices, batch_size)
        model, optimizer = setup_model_and_optimizer("distilbert-base-uncased", len(emotions), tf_train, epochs, learning_rate, cache_dir=cache_dir)
    else:
        # For RNN models, use Keras Tokenizer
        tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
        tokenizer.fit_on_texts(oversampled_train_df["text"])
        
        train_sequences = pad_sequences(tokenizer.texts_to_sequences(oversampled_train_df["text"]), maxlen=input_length, padding='post')
        valid_sequences = pad_sequences(tokenizer.texts_to_sequences(valid_df["text"]), maxlen=input_length, padding='post')
        test_sequences = pad_sequences(tokenizer.texts_to_sequences(test_df["text"]), maxlen=input_length, padding='post')
        
        # Remap labels to 0-9 range
        train_labels = np.array([mapping[label] for label in oversampled_train_df["label"].values])
        valid_labels = np.array([mapping[label] for label in valid_df["label"].values])
        test_labels = np.array([mapping[label] for label in test_df["label"].values])
        
        tf_train = tf.data.Dataset.from_tensor_slices((train_sequences, train_labels)).shuffle(len(train_sequences)).batch(batch_size)
        tf_val = tf.data.Dataset.from_tensor_slices((valid_sequences, valid_labels)).batch(batch_size)
        tf_test = tf.data.Dataset.from_tensor_slices((test_sequences, test_labels)).batch(batch_size)
        
        model = create_rnn_model(model_type, vocab_size, embedding_dim, input_length, len(emotions))
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model = compile_and_train(model, optimizer, tf_train, tf_val, epochs)

    save_model_and_tokenizer(model, tokenizer, save_path, is_distilbert)

    metrics = evaluate_model(model, tf_test)
    
    return metrics

if __name__ == "__main__":
    cache_dir = "/root/huggingface_cache"
    save_path = "/root/emotion_model"
    emotions = ["anger", "sadness", "joy", "disgust", "fear", "surprise", "gratitude", "remorse", "curiosity", "neutral"]

    # Train multiple models and collect metrics
    model_types = ["Bi-LSTM", "LSTM", "Bi-GRU", "GRU", "DistilBERT"]
    model_results = []

    for model_type in model_types:
        print(f"\nTraining {model_type}...")
        metrics = train_emotion_model(cache_dir, save_path, emotions, model_type=model_type)
        model_results.append({"Model": model_type, **metrics})

    # Print Models Comparison Table
    print("\nModels Evaluation Metrics Comparison")
    print("| Models | Accuracy | F1 Score | Precision | Recall |")
    print("|--------|----------|----------|-----------|--------|")
    for res in model_results:
        print(f"| {res['Model']} | {res['accuracy']:.2f} | {res['f1']:.2f} | {res['precision']:.2f} | {res['recall']:.2f} |")
