import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from data_utils import load_and_filter_goemotions, oversample_training_data, prepare_tokenized_datasets
from model_utils import create_tf_datasets, setup_model_and_optimizer, compile_and_train, save_model_and_tokenizer

def train_emotion_model(cache_dir, save_path, emotions, num_train=5000, epochs=5, batch_size=16, learning_rate=2e-5):
    print("Starting training...")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache directory ready: {cache_dir}")

    train_df, valid_df, test_df, sel_indices = load_and_filter_goemotions(cache_dir, emotions, num_train)
    oversampled_train_df = oversample_training_data(train_df)
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", cache_dir=cache_dir)
    
    tokenized_train, tokenized_valid, tokenized_test = prepare_tokenized_datasets(tokenizer, oversampled_train_df, valid_df, test_df)

    tf_train, tf_val, tf_test = create_tf_datasets(tokenized_train, tokenized_valid, tokenized_test, tokenizer, sel_indices, batch_size)

    model, optimizer = setup_model_and_optimizer("distilbert-base-uncased", len(emotions), tf_train, epochs, lr=learning_rate, cache_dir=cache_dir)

    model = compile_and_train(model, optimizer, tf_train, tf_val, epochs)

    save_model_and_tokenizer(model, tokenizer, save_path)

    # Evaluate with full metrics
    y_true = np.concatenate([y for x, y in tf_test], axis=0)
    y_pred = np.argmax(model.predict(tf_test).logits, axis=1)
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

if __name__ == "__main__":
    cache_dir = "/root/huggingface_cache"
    save_path = "/root/emotion_model"
    emotions = ["anger", "sadness", "joy", "disgust", "fear", "surprise", "gratitude", "remorse", "curiosity", "neutral"]

    # Run multiple experiments like in the report
    configs = [
        {"model_name": "Model-1", "num_train": 5000, "epochs": 2, "learning_rate": 5e-5, "batch_size": 16},
        {"model_name": "Model-2", "num_train": 5000, "epochs": 4, "learning_rate": 5e-5, "batch_size": 16},
        {"model_name": "Model-3", "num_train": 5000, "epochs": 4, "learning_rate": 3e-5, "batch_size": 32},
    ]

    results = []
    for conf in configs:
        print(f"\nRunning {conf['model_name']} with epochs={conf['epochs']}, lr={conf['learning_rate']}, batch={conf['batch_size']}")
        metrics = train_emotion_model(cache_dir, save_path, emotions, **conf)
        results.append({**conf, **metrics})

    # Print table like in the report
    print("\nModels Evaluation Metrics Comparison")
    print("Model | Epochs | Learning Rate | Batch Size | Accuracy | F1 Score | Precision | Recall")
    for res in results:
        print(f"{res['model_name']} | {res['epochs']} | {res['learning_rate']} | {res['batch_size']} | {res['accuracy']:.2f} | {res['f1']:.2f} | {res['precision']:.2f} | {res['recall']:.2f}")


