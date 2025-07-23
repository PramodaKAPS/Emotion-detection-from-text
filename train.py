"""
Updated train.py for PyTorch-based DeBERTa-v3-large emotion detection training.
Integrates focal loss, early stopping, learning rate scheduler, and ensemble placeholder.
Uses 2000 samples from GoEmotions for initial testing.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW  # Import AdamW from torch.optim
from transformers import AutoTokenizer, get_scheduler
from transformers import DebertaV2ForSequenceClassification  # Use DebertaV2 for v3 models
from focal_loss.focal_loss import FocalLoss  # pip install focal_loss_torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from datasets import Dataset
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from datasets import load_dataset

# Section 1: Dataset Loading and Filtering (from data_utils)
def load_and_filter_goemotions(cache_dir, selected_emotions, num_train=0):
    print("Loading GoEmotions dataset...")
    dataset = load_dataset("go_emotions", "simplified", cache_dir=cache_dir)

    train_df = pd.DataFrame(dataset["train"])
    valid_df = pd.DataFrame(dataset["validation"])
    test_df = pd.DataFrame(dataset["test"])

    emotion_names = dataset["train"].features["labels"].feature.names
    selected_indices = [emotion_names.index(e) for e in selected_emotions if e in emotion_names]

    print("Selected emotions:", selected_emotions)
    print("Selected indices:", selected_indices)

    def filter_emotions(df):
        df = df.copy()
        df["labels"] = df["labels"].apply(lambda x: [label for label in x if label in selected_indices])
        df = df[df["labels"].apply(lambda x: len(x) > 0)]
        return df

    train_df = filter_emotions(train_df)
    valid_df = filter_emotions(valid_df)
    test_df = filter_emotions(test_df)

    train_df["label"] = train_df["labels"].apply(lambda lbls: lbls[0] if lbls else -1)
    valid_df["label"] = valid_df["labels"].apply(lambda lbls: lbls[0] if lbls else -1)
    test_df["label"] = test_df["labels"].apply(lambda lbls: lbls[0] if lbls else -1)

    train_df = train_df[train_df["label"] != -1]
    valid_df = valid_df[valid_df["label"] != -1]
    test_df = test_df[test_df["label"] != -1]

    train_df = train_df[["text", "label"]]
    valid_df = valid_df[["text", "label"]]
    test_df = test_df[["text", "label"]]

    if num_train > 0:
        train_df = train_df.head(num_train)

    print(f"Filtered train data shape: {train_df.shape}")
    print(f"Filtered validation data shape: {valid_df.shape}")
    print(f"Filtered test data shape: {test_df.shape}")
    
    if train_df.empty:
        raise ValueError("Filtered train data is empty. Check selected emotions match dataset labels.")
    
    return train_df, valid_df, test_df, selected_indices

# Section 2: Oversampling (from data_utils)
def oversample_training_data(train_df):
    X = train_df["text"].values.reshape(-1, 1)
    y = train_df["label"]
    ros = RandomOverSampler(sampling_strategy='not majority', random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    df_resampled = pd.DataFrame({"text": X_resampled.flatten(), "label": y_resampled})
    print("Oversampled training data distribution:")
    print(df_resampled["label"].value_counts())
    return df_resampled

# Section 3: Tokenization (from data_utils, adapted for PyTorch)
def prepare_tokenized_datasets(tokenizer, train_df, valid_df, test_df):
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding=True, return_tensors="pt")

    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    test_dataset = Dataset.from_pandas(test_df)

    tokenized_train = train_dataset.map(tokenize, batched=True)
    tokenized_valid = valid_dataset.map(tokenize, batched=True)
    tokenized_test = test_dataset.map(tokenize, batched=True)

    tokenized_train.set_format("torch")
    tokenized_valid.set_format("torch")
    tokenized_test.set_format("torch")

    return tokenized_train, tokenized_valid, tokenized_test

# Training function (updated for PyTorch)
def train_emotion_model(cache_dir, save_path, emotions, num_train=2000, epochs=10, batch_size=64, learning_rate=3e-5, model_type="DeBERTa-v3-large"):
    print(f"Starting training for {model_type} with {num_train} samples...")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache directory ready: {cache_dir}")

    # Load and filter data
    train_df, valid_df, test_df, sel_indices = load_and_filter_goemotions(cache_dir, emotions, num_train)
    if num_train > 0:
        train_df = train_df.head(num_train)

    if train_df.empty:
        raise ValueError("Train DataFrame is empty after filtering. Check selected emotions or dataset loading.")

    # Oversample
    oversampled_train_df = oversample_training_data(train_df)

    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large", cache_dir=cache_dir)
    tokenized_train, tokenized_valid, tokenized_test = prepare_tokenized_datasets(tokenizer, oversampled_train_df, valid_df, test_df)

    # Label remapping
    mapping = {old: new for new, old in enumerate(sel_indices)}
    tokenized_train = tokenized_train.map(lambda x: {"label": mapping[x["label"]]})
    tokenized_valid = tokenized_valid.map(lambda x: {"label": mapping[x["label"]]})
    tokenized_test = tokenized_test.map(lambda x: {"label": mapping[x["label"]]})

    # PyTorch DataLoaders
    train_loader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(tokenized_valid, batch_size=batch_size)
    test_loader = DataLoader(tokenized_test, batch_size=batch_size)

    # Model setup
    model = DebertaV2ForSequenceClassification.from_pretrained("microsoft/deberta-v3-large", num_labels=len(emotions), cache_dir=cache_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = epochs * len(train_loader)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    criterion = FocalLoss(gamma=2)  # Focal loss for class imbalance

    # Training loop with early stopping
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
            labels = batch['label'].to(device)
            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {total_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
                labels = batch['label'].to(device)
                outputs = model(**inputs)
                val_loss += criterion(outputs.logits, labels).item()
        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Ensemble placeholder: For expansion, train multiple models and average predictions
    # e.g., models = [model] + [train_additional_model() for _ in range(2)]
    # Then, for prediction: avg_logits = sum(m(**inputs).logits for m in models) / len(models)

    # Save model and tokenizer
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved at {save_path}")

    # Evaluation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
            labels = batch['label'].cpu().numpy()
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            y_true.extend(labels)
            y_pred.extend(preds)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average='weighted'),
        "precision": precision_score(y_true, y_pred, average='weighted'),
        "recall": recall_score(y_true, y_pred, average='weighted')
    }
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

