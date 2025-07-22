import os
from data_utils import load_and_filter_goemotions, oversample_training_data, prepare_tokenized_datasets
from model_utils import create_tf_datasets, setup_model_and_optimizer, compile_and_train, save_model_and_tokenizer

def train_emotion_model(cache_dir, save_path, selected_emotions, num_train=5000, epochs=5, batch_size=16):
    print("Starting training...")
    # Create cache directory
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache directory ready: {cache_dir}")

    train_df, valid_df, test_df, sel_indices = load_and_filter_goemotions(cache_dir, selected_emotions, num_train)
    oversampled_train_df = oversample_training_data(train_df)
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", cache_dir=cache_dir)
    
    tokenized_train, tokenized_valid, tokenized_test = prepare_tokenized_datasets(tokenizer, oversampled_train_df, valid_df, test_df)

    tf_train, tf_val, tf_test = create_tf_datasets(tokenized_train, tokenized_valid, tokenized_test, tokenizer, sel_indices, batch_size)

    model, optimizer = setup_model_and_optimizer("distilbert-base-uncased", len(selected_emotions), tf_train, epochs, cache_dir=cache_dir)

    model = compile_and_train(model, optimizer, tf_train, tf_val, epochs)

    save_model_and_tokenizer(model, tokenizer, save_path)

    results = model.evaluate(tf_test)
    print(f"Evaluation results: {results}")
    return results

if __name__ == "__main__":
    cache_dir = "/root/huggingface_cache"  # Local path on Droplet
    save_path = "/root/emotion_model"  # Local save path
    emotions = ["anger", "sadness", "joy", "disgust", "fear", "surprise", "gratitude", "remorse", "curiosity", "neutral"]

    train_emotion_model(cache_dir, save_path, emotions, num_train=5000, epochs=5, batch_size=16)

