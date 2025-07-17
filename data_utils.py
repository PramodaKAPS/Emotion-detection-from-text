"""
Data utilities for emotion detection
"""
import pandas as pd
from datasets import load_dataset, Dataset
from imblearn.over_sampling import RandomOverSampler
import os

def load_and_filter_goemotions(cache_dir, selected_emotions, num_train=800):
    """
    Load and filter GoEmotions dataset for selected emotions
    
    Args:
        cache_dir (str): Directory for caching datasets
        selected_emotions (list): List of emotion names to filter
        num_train (int): Number of training samples to use
    
    Returns:
        tuple: (train_df, valid_df, test_df, selected_indices)
    """
    print("Loading GoEmotions dataset...")
    
    # Load the GoEmotions dataset with 'simplified' configuration
    dataset = load_dataset("go_emotions", "simplified", cache_dir=cache_dir)
    
    # Convert to DataFrame for easier filtering
    train_df = pd.DataFrame(dataset["train"])
    valid_df = pd.DataFrame(dataset["validation"])
    test_df = pd.DataFrame(dataset["test"])
    
    # Filter for the selected emotions
    emotion_mapping = dataset["train"].features["labels"].feature.names
    selected_indices = [emotion_mapping.index(emotion) for emotion in selected_emotions]
    
    def filter_emotions(df):
        """Filter rows with at least one of the selected emotions"""
        df = df.copy()
        df["labels"] = df["labels"].apply(lambda x: [label for label in x if label in selected_indices])
        df = df[df["labels"].apply(len) > 0]  # Keep rows with at least one selected label
        return df
    
    # Apply filtering
    train_df = filter_emotions(train_df)
    valid_df = filter_emotions(valid_df)
    test_df = filter_emotions(test_df)
    
    # Convert labels to single-label format (take the first matching label)
    train_df["label"] = train_df["labels"].apply(lambda x: x[0])
    valid_df["label"] = valid_df["labels"].apply(lambda x: x[0])
    test_df["label"] = test_df["labels"].apply(lambda x: x[0])
    
    # Drop the original 'labels' column
    train_df = train_df[["text", "label"]]
    valid_df = valid_df[["text", "label"]]
    test_df = test_df[["text", "label"]]
    
    # Subset the training data
    train_df = train_df.head(num_train)
    
    print(f"Filtered training data shape: {train_df.shape}")
    print(f"Filtered validation data shape: {valid_df.shape}")
    print(f"Filtered test data shape: {test_df.shape}")
    
    return train_df, valid_df, test_df, selected_indices

def oversample_training_data(train_df):
    """
    Oversample training data to balance classes
    
    Args:
        train_df (pd.DataFrame): Training dataframe with 'text' and 'label' columns
    
    Returns:
        pd.DataFrame: Oversampled training dataframe
    """
    print("Oversampling training data...")
    
    X_train = train_df["text"].values.reshape(-1, 1)
    y_train = train_df["label"]
    
    ros = RandomOverSampler(sampling_strategy="not majority", random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    
    emotions_train = pd.DataFrame({
        "text": X_resampled.flatten(), 
        "label": y_resampled
    })
    
    print("Class distribution after oversampling:")
    print(emotions_train["label"].value_counts())
    
    return emotions_train

def prepare_datasets_for_training(train_df, valid_df, test_df, tokenizer):
    """
    Prepare datasets for model training
    
    Args:
        train_df, valid_df, test_df (pd.DataFrame): DataFrames with text and labels
        tokenizer: Hugging Face tokenizer
    
    Returns:
        tuple: (tokenized_train, tokenized_val, tokenized_test)
    """
    print("Preparing datasets for training...")
    
    def tokenize_function(example):
        return tokenizer(example["text"], padding=True, truncation=True)
    
    # Convert DataFrames to Dataset format for tokenization
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Tokenize the datasets
    tokenized_train = train_dataset.map(tokenize_function, batched=True, batch_size=None)
    tokenized_val = valid_dataset.map(tokenize_function, batched=True, batch_size=None)
    tokenized_test = test_dataset.map(tokenize_function, batched=True, batch_size=None)
    
    return tokenized_train, tokenized_val, tokenized_test

