"""
Main training script for emotion detection model
"""
from data_utils import load_and_filter_goemotions, oversample_training_data, prepare_datasets_for_training
from model_utils import (
    load_tokenizer, create_tensorflow_datasets, create_model_and_optimizer,
    compile_and_train_model, evaluate_and_save_model
)

def train_emotion_model(cache_dir, save_path, selected_emotions, num_train=800, 
                       num_epochs=1, batch_size=16):
    """
    Complete training pipeline for emotion detection model
    
    Args:
        cache_dir (str): Directory for caching datasets and models
        save_path (str): Path to save trained model
        selected_emotions (list): List of emotion names to train on
        num_train (int): Number of training samples
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
    """
    try:
        print("Starting emotion detection model training...")
        
        # Load and filter dataset
        train_df, valid_df, test_df, selected_indices = load_and_filter_goemotions(
            cache_dir, selected_emotions, num_train
        )
        
        # Oversample training data
        emotions_train = oversample_training_data(train_df)
        
        # Load tokenizer
        model_checkpoint = "distilbert-base-uncased"
        tokenizer = load_tokenizer(model_checkpoint, cache_dir)
        
        # Prepare datasets
        tokenized_train, tokenized_val, tokenized_test = prepare_datasets_for_training(
            emotions_train, valid_df, test_df, tokenizer
        )
        
        # Create TensorFlow datasets
        tf_train_dataset, tf_validation_dataset, tf_test_dataset = create_tensorflow_datasets(
            tokenized_train, tokenized_val, tokenized_test, 
            tokenizer, selected_indices, batch_size
        )
        
        # Create model and optimizer
        model, optimizer = create_model_and_optimizer(
            model_checkpoint, cache_dir, len(selected_emotions), 
            tf_train_dataset, num_epochs
        )
        
        # Train model
        model = compile_and_train_model(
            model, optimizer, tf_train_dataset, tf_validation_dataset, num_epochs
        )
        
        # Evaluate and save model
        test_results = evaluate_and_save_model(
            model, tf_test_dataset, tokenizer, save_path
        )
        
        print("✅ Training completed successfully!")
        return test_results
        
    except Exception as e:
        print(f"❌ An error occurred during training: {e}")
        raise

if __name__ == "__main__":
    # Training configuration
    cache_dir = "/content/huggingface_cache"
    os.makedirs(cache_dir, exist_ok=True)
    #save_path = "/content/drive/MyDrive/emotion_model"
    save_dir = "./emotion_model"  # Local folder on droplet
    os.makedirs(save_dir, exist_ok=True)  # Create if not exists
    selected_emotions = [
        "anger", "sadness", "joy", "disgust", "fear", 
        "surprise", "gratitude", "remorse", "curiosity", "neutral"
    ]
    
    # Run training
    train_emotion_model(
        cache_dir=cache_dir,
        save_path=save_path,
        selected_emotions=selected_emotions,
        num_train=800,
        num_epochs=1,
        batch_size=16
    )
