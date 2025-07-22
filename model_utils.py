import os
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification, create_optimizer, DataCollatorWithPadding, AutoTokenizer

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

def setup_model_and_optimizer(model_name, num_labels, tf_train_dataset, epochs=1, lr=2e-5, cache_dir=None):
    model = TFDistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, cache_dir=cache_dir)
    steps = len(tf_train_dataset) * epochs
    optimizer, schedule = create_optimizer(lr, 0, steps)
    return model, optimizer

def compile_and_train(model, optimizer, tf_train_dataset, tf_val_dataset, epochs=1):
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    model.fit(tf_train_dataset, validation_data=tf_val_dataset, epochs=epochs)
    return model

def save_model_and_tokenizer(model, tokenizer, path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Model and tokenizer saved at {path}")

