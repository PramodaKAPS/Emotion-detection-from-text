import os
import sys
from google.colab import drive


# Setup cache directory
cache_dir = "/content/huggingface_cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Created cache directory: {cache_dir}")

# Import your modules
from data_utils import load_and_filter_goemotions, oversample_train_df
from model_utils import (
    tokenize_df, create_tf_datasets, get_optimizer, train_and_save
)
from transformers import AutoTokenizer, DataCollatorWithPadding, TFDistilBertForSequenceClassification

# Parameters
selected_emotions = ["anger", "sadness", "joy", "disgust", "fear", "surprise", "gratitude", "remorse", "curiosity", "neutral"]
model_ckpt = "distilbert-base-uncased"
num_epochs = 1
batch_size = 16
save_path = "/content/drive/MyDrive/emotion_model"

try:
    # Prepare data
    train_df, valid_df, test_df, selected_indices = load_and_filter_goemotions(cache_dir, selected_emotions, num_train=800)
    emotions_train = oversample_train_df(train_df)
    
    print("Class distribution after oversampling:")
    print(emotions_train["label"].value_counts())
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, cache_dir=cache_dir)
    
    # Tokenization
    tokenized_train = tokenize_df(emotions_train, tokenizer)
    tokenized_valid = tokenize_df(valid_df, tokenizer)
    tokenized_test = tokenize_df(test_df, tokenizer)
    
    # Map labels to 0-9
    emotion_mapping_dict = {idx: i for i, idx in enumerate(selected_indices)}
    tokenized_train = tokenized_train.map(lambda x: {"label": emotion_mapping_dict[x["label"]]})
    tokenized_valid = tokenized_valid.map(lambda x: {"label": emotion_mapping_dict[x["label"]]})
    tokenized_test = tokenized_test.map(lambda x: {"label": emotion_mapping_dict[x["label"]]})
    
    # Create TF datasets
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
    tf_train_dataset, tf_validation_dataset, tf_test_dataset = create_tf_datasets(
        tokenized_train, tokenized_valid, tokenized_test, data_collator, batch_size)
    
    # Load and train model
    model = TFDistilBertForSequenceClassification.from_pretrained(model_ckpt, num_labels=10, cache_dir=cache_dir)
    optimizer = get_optimizer(num_train_steps=len(tf_train_dataset) * num_epochs)
    
    # Save tokenizer as well
    tokenizer.save_pretrained(save_path)
    
    train_and_save(model, optimizer, tf_train_dataset, tf_validation_dataset, num_epochs, save_path)
    
    # Evaluate on test set
    test_results = model.evaluate(tf_test_dataset)
    print("Test set results:", test_results)
    
except Exception as e:
    print(f"An error occurred: {e}")
    print("Please check your library versions and restart the runtime if needed.")

