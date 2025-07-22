import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
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
    
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall, "y_true": y_true, "y_pred": y_pred}

def generate_confusion_matrix(y_true, y_pred, labels, normalize=None, save_path=None):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)), normalize=normalize)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()

if __name__ == "__main__":
    cache_dir = "/root/huggingface_cache"
    save_path = "/root/emotion_model"
    emotions = ["anger", "sadness", "joy", "disgust", "fear", "surprise", "gratitude", "remorse", "curiosity", "neutral"]

    # Run multiple experiments
    configs = [
        {"model_name": "Model-1", "num_train": 5000, "epochs": 2, "learning_rate": 5e-5, "batch_size": 16},
        {"model_name": "Model-2", "num_train": 5000, "epochs": 4, "learning_rate": 5e-5, "batch_size": 16},
        {"model_name": "Model-3", "num_train": 5000, "epochs": 4, "learning_rate": 3e-5, "batch_size": 32},
    ]

    results = []
    for conf in configs:
        model_name = conf.pop('model_name')
        print(f"\nRunning {model_name} with epochs={conf['epochs']}, lr={conf['learning_rate']}, batch={conf['batch_size']}")
        metrics = train_emotion_model(cache_dir, save_path, emotions, **conf)
        results.append({"model_name": model_name, **conf, **metrics})

    # Print table
    print("\nModels Evaluation Metrics Comparison")
    print("Model | Epochs | Learning Rate | Batch Size | Accuracy | F1 Score | Precision | Recall")
    for res in results:
        print(f"{res['model_name']} | {res['epochs']} | {res['learning_rate']} | {res['batch_size']} | {res['accuracy']:.2f} | {res['f1']:.2f} | {res['precision']:.2f} | {res['recall']:.2f}")

    # Generate confusion matrix for the best model (highest accuracy)
    best_res = max(results, key=lambda x: x['accuracy'])
    print(f"\nGenerating confusion matrix for best model: {best_res['model_name']} (Accuracy: {best_res['accuracy']:.2f})")
    generate_confusion_matrix(best_res['y_true'], best_res['y_pred'], emotions, normalize=None, save_path=os.path.join(save_path, 'confusion_matrix.png'))



