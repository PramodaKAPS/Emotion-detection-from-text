import os
import pandas as pd
from datasets import load_dataset, Dataset
from imblearn.over_sampling import RandomOverSampler

def load_and_filter_goemotions(cache_dir, selected_emotions, num_train=800):
    """
    Load and filter GoEmotions dataset with error handling for pattern issues
    """
    try:
        # First attempt: Try loading with simplified configuration
        dataset = load_dataset("go_emotions", "simplified", cache_dir=cache_dir)
    except ValueError as e:
        if "Invalid pattern" in str(e):
            print("Attempting alternative loading method...")
            try:
                # Second attempt: Try without explicit cache_dir
                dataset = load_dataset("go_emotions", "simplified")
            except:
                # Third attempt: Try without simplified configuration
                dataset = load_dataset("go_emotions")
        else:
            raise e
    
    # Rest of your existing code remains the same
    train_df = pd.DataFrame(dataset["train"])
    valid_df = pd.DataFrame(dataset["validation"])
    test_df = pd.DataFrame(dataset["test"])
    emotion_names = dataset["train"].features["labels"].feature.names
    selected_indices = [emotion_names.index(e) for e in selected_emotions]

    def filter_emotions(df):
        df["labels"] = df["labels"].apply(lambda x: [label for label in x if label in selected_indices])
        df = df[df["labels"].apply(len) > 0]
        return df

    train_df = filter_emotions(train_df)
    valid_df = filter_emotions(valid_df)
    test_df = filter_emotions(test_df)
    
    train_df["label"] = train_df["labels"].apply(lambda x: x[0])
    valid_df["label"] = valid_df["labels"].apply(lambda x: x[0])
    test_df["label"] = test_df["labels"].apply(lambda x: x[0])
    
    train_df = train_df[["text", "label"]]
    valid_df = valid_df[["text", "label"]]
    test_df = test_df[["text", "label"]]
    
    train_df = train_df.head(num_train)
    return train_df, valid_df, test_df, selected_indices

def oversample_train_df(train_df):
    """
    Oversample training data to balance classes
    """
    X_train = train_df["text"].values.reshape(-1, 1)
    y_train = train_df["label"]
    ros = RandomOverSampler(sampling_strategy="not majority", random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    emotions_train = pd.DataFrame({"text": X_resampled.flatten(), "label": y_resampled})
    return emotions_train

