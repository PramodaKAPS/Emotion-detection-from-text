import os
import numpy as np
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification, create_optimizer, DataCollatorWithPadding, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, GRU, Dense

# Section 1: Dataset Creation Utilities
def create_tf_datasets(tokenized_train, tokenized_valid, tokenized_test, tokenizer, selected_indices, batch_size=16):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
    mapping = {old: new for new, old in enumerate(selected_indices)}

    tokenized_train = tokenized_train.map(lambda x: {"label": mapping[x["label"]]})
    tokenized_valid = tokenized_valid.map(lambda x: {"label": mapping[x["label"]]})
    tokenized_test = tokenized_test.map(lambda x: {"label": mapping[x["label"]]})

    tf_train = tokenized_train.to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols=["label"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    tf_val = tokenized_valid.to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols=["label"],
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    tf_test = tokenized_test.to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols=["label"],
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    return tf_train, tf_val, tf_test

# Section 2: Model Creation Utilities
def create_rnn_model(model_type, vocab_size, embedding_dim, input_length, num_labels):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=input_length))
    if model_type == "Bi-LSTM":
        model.add(Bidirectional(LSTM(128)))
    elif model_type == "LSTM":
        model.add(LSTM(128))
    elif model_type == "Bi-GRU":
        model.add(Bidirectional(GRU(128)))
    elif model_type == "GRU":
        model.add(GRU(128))
    model.add(Dense(num_labels, activation='softmax'))
    return model

def setup_model_and_optimizer(model_name, num_labels, tf_train_dataset, epochs=1, lr=2e-5, cache_dir=None):
    model = TFDistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, cache_dir=cache_dir)
    steps = len(tf_train_dataset) * epochs
    optimizer, schedule = create_optimizer(lr, 0, steps)
    return model, optimizer

# Section 3: Model Compilation and Training
def compile_and_train(model, optimizer, tf_train_dataset, tf_val_dataset, epochs=1):
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    model.fit(tf_train_dataset, validation_data=tf_val_dataset, epochs=epochs)
    return model

# Section 4: Model Saving
def save_model_and_tokenizer(model, tokenizer, path, is_distilbert=False):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    if is_distilbert:
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
    else:
        model.save(path)
    print(f"Model saved at {path}")

# Section 5: Evaluation Metrics
def evaluate_model(model, tf_test):
    y_true = np.concatenate([y for x, y in tf_test], axis=0)
    y_pred = np.argmax(model.predict(tf_test).logits, axis=1) if hasattr(model.predict(tf_test), 'logits') else np.argmax(model.predict(tf_test), axis=1)
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

    


