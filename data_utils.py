def load_and_filter_goemotions(cache_dir, selected_emotions, num_train=0):
    print("Loading GoEmotions dataset...")
    dataset = load_dataset("go_emotions", "simplified", cache_dir=cache_dir)

    train_df = pd.DataFrame(dataset["train"])
    valid_df = pd.DataFrame(dataset["validation"])
    test_df = pd.DataFrame(dataset["test"])

    emotion_names = dataset["train"].features["labels"].feature.names
    selected_indices = [emotion_names.index(e) for e in selected_emotions if e in emotion_names]  # Fix: Skip if emotion not in dataset

    print("Selected emotions:", selected_emotions)
    print("Selected indices:", selected_indices)

    def filter_emotions(df):
        df = df.copy()
        df["labels"] = df["labels"].apply(lambda x: [label for label in x if label in selected_indices])
        df = df[df["labels"].apply(lambda x: len(x) > 0)]
        return df

    train_df = filter_emotions(train_df)
    valid_df = filter_emotions(valid_df)
    test_df = filter_emotions(test_df)

    train_df["label"] = train_df["labels"].apply(lambda lbls: lbls[0] if lbls else -1)  # Fix: Handle empty labels
    valid_df["label"] = valid_df["labels"].apply(lambda lbls: lbls[0] if lbls else -1)
    test_df["label"] = test_df["labels"].apply(lambda lbls: lbls[0] if lbls else -1)

    train_df = train_df[train_df["label"] != -1]  # Remove invalid
    valid_df = valid_df[valid_df["label"] != -1]
    test_df = test_df[test_df["label"] != -1]

    train_df = train_df[["text", "label"]]
    valid_df = valid_df[["text", "label"]]
    test_df = test_df[["text", "label"]]

    # Handle full dataset
    if num_train > 0:
        train_df = train_df.head(num_train)
    # Else use all

    print(f"Filtered train data shape: {train_df.shape}")
    print(f"Filtered validation data shape: {valid_df.shape}")
    print(f"Filtered test data shape: {test_df.shape}")
    
    if train_df.empty:
        raise ValueError("Filtered train data is empty. Check selected emotions match dataset labels.")
    
    return train_df, valid_df, test_df, selected_indices
