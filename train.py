import os
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, DebertaV3ForSequenceClassification, AdamW, get_scheduler
from torch.utils.data import DataLoader
from focal_loss.focal_loss import FocalLoss  # pip install focal_loss_torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from data_utils import load_and_filter_goemotions, oversample_training_data, prepare_tokenized_datasets
from model_utils import create_tf_datasets, save_model_and_tokenizer, evaluate_model  # Update these for PyTorch if needed

def train_emotion_model(cache_dir, save_path, emotions, num_train=2000, epochs=10, batch_size=64, learning_rate=3e-5, model_type="DeBERTa-v3-large"):
    print(f"Starting training for {model_type} with {num_train} samples...")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    train_df, valid_df, test_df, sel_indices = load_and_filter_goemotions(cache_dir, emotions, num_train)
    if num_train > 0:
        train_df = train_df.head(num_train)

    if train_df.empty:
        raise ValueError("Train DataFrame is empty after filtering.")

    oversampled_train_df = oversample_training_data(train_df)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large", cache_dir=cache_dir)
    tokenized_train, tokenized_valid, tokenized_test = prepare_tokenized_datasets(tokenizer, oversampled_train_df, valid_df, test_df)

    # PyTorch DataLoaders (replace create_tf_datasets)
    train_loader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(tokenized_valid, batch_size=batch_size)
    test_loader = DataLoader(tokenized_test, batch_size=batch_size)

    model = DebertaV3ForSequenceClassification.from_pretrained("microsoft/deberta-v3-large", num_labels=len(emotions), cache_dir=cache_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = epochs * len(train_loader)
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    criterion = FocalLoss(gamma=2)  # Focal loss

    # Training loop with early stopping (manual implementation)
    best_val_loss = float('inf')
    patience = 3
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(device)
                outputs = model(**inputs)
                val_loss += criterion(outputs.logits, labels).item()
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break

    # Save and evaluate (adapt evaluate_model for PyTorch)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # Evaluation (simplified; adapt as needed)
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].cpu().numpy()
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
    print("\nTest Metrics:", metrics)
    return metrics

if __name__ == "__main__":
    cache_dir = "/root/huggingface_cache"
    save_path = "/root/emotion_model_deberta"
    emotions = ["anger", "sadness", "joy", "disgust", "fear", "surprise", "gratitude", "remorse", "curiosity", "neutral"]
    metrics = train_emotion_model(cache_dir, save_path, emotions, num_train=2000)


