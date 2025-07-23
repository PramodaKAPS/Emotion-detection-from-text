import os
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFDebertaV3ForSequenceClassification, create_optimizer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from focal_loss import SparseCategoricalFocalLoss  # Install via pip install focal-loss
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from data_utils import load_and_filter_goemotions, oversample_training_data, prepare_tokenized_datasets
from model_utils import create_tf_datasets, save_model_and_tokenizer, evaluate_model

def train_emotion_model(cache_dir, save_path, emotions, num_train=2000, epochs=10, batch_size=64, learning_rate=3e-5, model_type="DeBERTa-v3-large"):
    print(f"Starting training for {model_type} with {num_train} samples...")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache directory ready: {cache_dir}")

    # Load and filter GoEmotions dataset, limit to num_train for testing
    train_df, valid_df, test_df, sel_indices = load_and_filter_goemotions(cache_dir, emotions, num_train)
    if num_train > 0:
        train_df = train_df.head(num_train)  # Limit to 2000 samples for quick check

    if train_df.empty:
        raise ValueError("Train DataFrame is empty after filtering. Check selected emotions or dataset loading.")

    # Oversample to balance classes
    oversampled_train_df = oversample_training_data(train_df)

    # Tokenization for DeBERTa-v3-large
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large", cache_dir=cache_dir)
    tokenized_train, tokenized_valid, tokenized_test = prepare_tokenized_datasets(tokenizer, oversampled_train_df, valid_df, test_df)

    # Create TF datasets with label remapping
    mapping = {old: new for new, old in enumerate(sel_indices)}
    tf_train, tf_val, tf_test = create_tf_datasets(tokenized_train, tokenized_valid, tokenized_test, tokenizer, sel_indices, batch_size)

    # Setup DeBERTa-v3-large model and optimizer
    model = TFDebertaV3ForSequenceClassification.from_pretrained("microsoft/deberta-v3-large", num_labels=len(emotions), cache_dir=cache_dir)
    steps = len(tf_train) * epochs
    optimizer, schedule = create_optimizer(learning_rate, 0, steps)

    # Compile with focal loss
    model.compile(optimizer=optimizer, loss=SparseCategoricalFocalLoss(gamma=2), metrics=["accuracy"])

    # Callbacks: Early stopping and LR scheduler
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-7)

    # Train the model
    history = model.fit(tf_train, validation_data=tf_val, epochs=epochs, callbacks=[early_stopping, lr_scheduler])

    # Ensemble expansion placeholder: Train multiple models and average predictions
    # Example: models = [model] + [train_additional_model() for _ in range(2)]  # Expand here for ensemble
    # For now, single model; in ensemble, predict via np.mean([m.predict(...) for m in models], axis=0)

    # Save model and tokenizer
    save_model_and_tokenizer(model, tokenizer, save_path, is_distilbert=False)

    # Evaluate and print metrics to check if it works
    metrics = evaluate_model(model, tf_test)
    print("\nTest Metrics on 2000-sample run:")
    print(f"Accuracy: {metrics['accuracy']:.2f} | F1 Score: {metrics['f1']:.2f} | Precision: {metrics['precision']:.2f} | Recall: {metrics['recall']:.2f}")

    return metrics

if __name__ == "__main__":
    cache_dir = "/root/huggingface_cache"
    save_path = "/root/emotion_model_deberta"
    emotions = ["anger", "sadness", "joy", "disgust", "fear", "surprise", "gratitude", "remorse", "curiosity", "neutral"]

    # Run training with 2000 samples to check functionality
    metrics = train_emotion_model(cache_dir, save_path, emotions, num_train=2000, epochs=10, batch_size=64, learning_rate=3e-5, model_type="DeBERTa-v3-large")
    print("\nTraining completed successfully with 2000 samples! Metrics indicate the script is working.")

