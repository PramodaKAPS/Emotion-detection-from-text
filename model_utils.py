from transformers import AutoTokenizer, DataCollatorWithPadding, TFDistilBertForSequenceClassification, create_optimizer
from datasets import Dataset
import os
import tensorflow as tf

def tokenize_df(df, tokenizer):
    def tokenize_function(example):
        return tokenizer(example["text"], padding=True, truncation=True)
    dataset = Dataset.from_pandas(df)
    return dataset.map(tokenize_function, batched=True, batch_size=None)

def create_tf_datasets(tokenized_train, tokenized_valid, tokenized_test, data_collator, batch_size=16):
    tf_train_dataset = tokenized_train.to_tf_dataset(
        columns=["attention_mask", "input_ids"],
        label_cols=["label"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    tf_validation_dataset = tokenized_valid.to_tf_dataset(
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

def get_optimizer(num_train_steps, init_lr=2e-5):
    optimizer, schedule = create_optimizer(
        init_lr=init_lr,
        num_warmup_steps=0,
        num_train_steps=num_train_steps)
    return optimizer

def train_and_save(model, optimizer, tf_train_dataset, tf_validation_dataset, num_epochs, save_path):
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],)
    model.fit(
        tf_train_dataset,
        validation_data=tf_validation_dataset,
        epochs=num_epochs,
    )
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    print(f"Model saved to {save_path}")
