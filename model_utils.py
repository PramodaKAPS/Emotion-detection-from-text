"""
Model utilities for emotion detection
"""
import os
import tensorflow as tf
from transformers import (
    AutoTokenizer, 
    DataCollatorWithPadding, 
    TFDistilBertForSequenceClassification, 
    create_optimizer
)

def load_tokenizer(model_checkpoint, cache_dir):
    """
    Load tokenizer from Hugging Face Hub
    
    Args:
        model_checkpoint (str): Model checkpoint name
        cache_dir (str): Cache directory
    
    Returns:
        tokenizer: Hugging Face tokenizer
    """
    print(f"Loading tokenizer: {model_checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, cache_dir=cache_dir)
    return tokenizer

def create_tensorflow_datasets(tokenized_train, tokenized_val, tokenized_test, 
                              tokenizer, selected_indices, batch_size=16):
    """
    Create TensorFlow datasets for training
    
    Args:
        tokenized_train, tokenized_val, tokenized_test: Tokenized datasets
        tokenizer: Hugging Face tokenizer
        selected_indices (list): List of selected emotion indices
        batch_size (int): Batch size for training
    
    Returns:
        tuple: (tf_train_dataset, tf_validation_dataset, tf_test_dataset)
    """
    print("Creating TensorFlow datasets...")
    
    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
    
    # Define the label mapping (map original indices to 0-9)
    emotion_mapping_dict = {idx: i for i, idx in enumerate(selected_indices)}
    
    # Remap labels in the tokenized datasets
    tokenized_train = tokenized_train.map(lambda x: {"label": emotion_mapping_dict[x["label"]]})
    tokenized_val = tokenized_val.map(lambda x: {"label": emotion_mapping_dict[x["label"]]})
    tokenized_test = tokenized_test.map(lambda x: {"label": emotion_mapping_dict[x["label"]]})
    
    # Convert to TensorFlow datasets
    tf_train_dataset = tokenized_train.to_tf_dataset(
        columns=["attention_mask", "input_ids"],
        label_cols=["label"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    
    tf_validation_dataset = tokenized_val.to_tf_dataset(
        columns=["attention_mask", "input_ids"],
        label_cols=["label"],
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    
    tf_test_dataset = tokenized_test.to_tf_dataset(
        columns=["attention_mask", "input_ids"],
        label_cols=["label"],
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    
    return tf_train_dataset, tf_validation_dataset, tf_test_dataset

def create_model_and_optimizer(model_checkpoint, cache_dir, num_labels, 
                              tf_train_dataset, num_epochs=1, learning_rate=2e-5):
    """
    Create model and optimizer for training
    
    Args:
        model_checkpoint (str): Model checkpoint name
        cache_dir (str): Cache directory
        num_labels (int): Number of emotion labels
        tf_train_dataset: TensorFlow training dataset
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimizer
    
    Returns:
        tuple: (model, optimizer)
    """
    print(f"Loading model: {model_checkpoint}")
    
    # Load the model with specified number of labels
    model = TFDistilBertForSequenceClassification.from_pretrained(
        model_checkpoint, 
        num_labels=num_labels, 
        cache_dir=cache_dir
    )
    
    # Define optimizer and learning rate schedule
    num_train_steps = len(tf_train_dataset) * num_epochs
    optimizer, schedule = create_optimizer(
        init_lr=learning_rate,
        num_warmup_steps=0,
        num_train_steps=num_train_steps,
    )
    
    return model, optimizer

def compile_and_train_model(model, optimizer, tf_train_dataset, 
                           tf_validation_dataset, num_epochs=1):
    """
    Compile and train the emotion detection model
    
    Args:
        model: TensorFlow model
        optimizer: Model optimizer
        tf_train_dataset: Training dataset
        tf_validation_dataset: Validation dataset
        num_epochs (int): Number of training epochs
    
    Returns:
        model: Trained model
    """
    print("Compiling model...")
    
    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    
    print("Starting model training...")
    
    # Train the model
    model.fit(
        tf_train_dataset,
        validation_data=tf_validation_dataset,
        epochs=num_epochs,
    )
    
    return model

def evaluate_and_save_model(model, tf_test_dataset, tokenizer, save_path):
    """
    Evaluate model and save to specified path
    
    Args:
        model: Trained model
        tf_test_dataset: Test dataset
        tokenizer: Hugging Face tokenizer
        save_path (str): Path to save model
    
    Returns:
        test_results: Model evaluation results
    """
    print("Evaluating model on test set...")
    
    # Evaluate on the test set
    test_results = model.evaluate(tf_test_dataset)
    print("Test set results:", test_results)
    
    # Create the directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save the model and tokenizer
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to {save_path}")
    
    return test_results
