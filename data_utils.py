import pandas as pd
from datasets import load_dataset, Dataset
from imblearn.over_sampling import RandomOverSampler
import os

def load_and_filter_goemotions(cache_dir, selected_emotions, num_train=800):
    """
    Load GoEmotions dataset with multiple fallback methods
    """
    dataset = None
    
    # Method 1: Try with cache_dir and simplified config
    try:
        dataset = load_dataset("go_emotions", "simplified", cache_dir=cache_dir)
        print("✅ Loaded with cache_dir and simplified config")
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
        
        # Method 2: Try without cache_dir
        try:
            dataset = load_dataset("go_emotions", "simplified")
            print("✅ Loaded with simplified config (no cache)")
        except Exception as e2:
            print(f"❌ Method 2 failed: {e2}")
            
            # Method 3: Try without simplified config
            try:
                dataset = load_dataset("go_emotions")
                print("✅ Loaded without simplified config")
            except Exception as e3:
                print(f"❌ Method 3 failed: {e3}")
                
                # Method 4: Try with streaming mode
                try:
                    dataset = load_dataset("go_emotions", streaming=True)
                    # Convert streaming dataset to regular dataset
                    train_data = list(dataset["train"].take(5000))
                    valid_data = list(dataset["validation"].take(1000))
                    test_data = list(dataset["test"].take(1000))
                    
                    dataset = {
                        "train": Dataset.from_list(train_data),
                        "validation": Dataset.from_list(valid_data),
                        "test": Dataset.from_list(test_data)
                    }
                    print("✅ Loaded with streaming mode")
                except Exception as e4:
                    print(f"❌ All methods failed. Last error: {e4}")
                    raise Exception("Unable to load GoEmotions dataset with any method")
    
    if dataset is None:
        raise Exception("Dataset loading failed")
    
    # Rest of your existing filtering code...
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
