"""
Main script for running emotion detection training in Google Colab
"""
from setup import setup_cache_directory, mount_google_drive
from train import train_emotion_model

def main():
    """
    Main function to run emotion detection training in Google Colab
    """
    print("🚀 Starting Emotion Detection Training in Google Colab")
    print("=" * 60)
    
    # Mount Google Drive
    mount_google_drive()
    
    # Setup cache directory
    cache_dir = setup_cache_directory()
    
    # Training configuration
    save_path = "./emotion_model"  # Local folder on droplet
    os.makedirs(save_dir, exist_ok=True)  # Create if not exists
    selected_emotions = [
        "anger", "sadness", "joy", "disgust", "fear", 
        "surprise", "gratitude", "remorse", "curiosity", "neutral"
    ]
    
    # Training parameters
    config = {
        "num_train": 800,
        "num_epochs": 1,
        "batch_size": 16
    }
    
    print(f"📊 Training Configuration:")
    print(f"   - Cache directory: {cache_dir}")
    print(f"   - Save path: {save_path}")
    print(f"   - Selected emotions: {selected_emotions}")
    print(f"   - Training samples: {config['num_train']}")
    print(f"   - Epochs: {config['num_epochs']}")
    print(f"   - Batch size: {config['batch_size']}")
    print("-" * 60)
    
    # Run training
    try:
        test_results = train_emotion_model(
            cache_dir=cache_dir,
            save_path=save_path,
            selected_emotions=selected_emotions,
            **config
        )
        
        print("\n🎉 Training completed successfully!")
        print(f"📊 Final test results: {test_results}")
        print(f"💾 Model saved to: {save_path}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
