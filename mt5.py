from sklearn.model_selection import train_test_split
import logging
import pandas as pd
import torch
from datasets import load_dataset
from transformers import (
    T5TokenizerFast,
    MT5ForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
)
from huggingface_hub import HfFolder, notebook_login

from datasets import load_dataset, Dataset, ClassLabel, DatasetDict

from sklearn.metrics import accuracy_score, f1_score
import evaluate
import numpy as np

# Load dataset
data = pd.read_csv('/mnt/disk2/arshia.yousefinezhad/emotion_detection/data/preprocess_labelencoding_data.csv')
data = data.sample(5000)


# Split dataset to train test validation
call_data, data_val = train_test_split(data,test_size=0.15,  random_state=42 , stratify=data.emotion)
data_train, data_test = train_test_split(call_data,test_size=0.1 ,  random_state=42 , stratify=call_data.emotion)

# convert in to huggingface dataset
train_dataset = Dataset.from_pandas(data_train)
train_dataset = train_dataset.remove_columns(["__index_level_0__"])

val_dataset = Dataset.from_pandas(data_val)
val_dataset = val_dataset.remove_columns(["__index_level_0__"])

test_dataset = Dataset.from_pandas(data_test)
test_dataset = test_dataset.remove_columns(["__index_level_0__"])


# convert dataset to pandas dataset
df_train = train_dataset.to_pandas()
df_val = val_dataset.to_pandas()
df_test = test_dataset.to_pandas()


#  unique labels
unique_labels_train = list(sorted(df_train['emotion'].unique()))

# ClassLabels emotion
class_label_feature = ClassLabel(names=unique_labels_train)


def label_str_to_int_call(example):
    example['emotion'] = class_label_feature.str2int(example['emotion'])
    return example

# calls dataset
train_dataset = train_dataset.map(label_str_to_int_call)
train_dataset = train_dataset.cast_column('emotion', class_label_feature)

val_dataset = val_dataset.map(label_str_to_int_call)
val_dataset = val_dataset.cast_column('emotion', class_label_feature)

test_dataset = test_dataset.map(label_str_to_int_call)
test_dataset = test_dataset.cast_column('emotion', class_label_feature)


df = df_train.copy()

mt5_id = "/mnt/disk2/LanguageModels/mt5_large"

label_list = list(sorted(df['emotion'].unique()))

label2id = {label: i for i, label in enumerate(label_list)}
id2label = {v: k for k, v in label2id.items()}

# Update the model's configuration with the id2label mapping
config = AutoConfig.from_pretrained(mt5_id)
config.update({"id2label": id2label})


# Preprocessing
tokenizer = T5TokenizerFast.from_pretrained(mt5_id)

# This function tokenizes the input text using the RoBERTa tokenizer.
# It applies padding and truncation to ensure that all sequences have the same length (256 tokens).
# https://huggingface.co/learn/nlp-course/chapter5/3?fw=pt#the-map-methods-superpowers
def tokenize(batch):
    return tokenizer(batch["combined_text"], padding=True, truncation=True, max_length=256)

train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
val_dataset = val_dataset.map(tokenize, batched=True, batch_size=len(val_dataset))
test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(test_dataset))

train_dataset = train_dataset.remove_columns(["combined_text"])
val_dataset = val_dataset.remove_columns(["combined_text"])
test_dataset = test_dataset.remove_columns(["combined_text"])

train_dataset = train_dataset.rename_column("emotion", "labels")
val_dataset = val_dataset.rename_column("emotion", "labels")
test_dataset = test_dataset.rename_column("emotion", "labels")


# Set dataset format
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])




accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    f1 = f1_score(labels, predictions, average="weighted")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1}


# Model
model = MT5ForSequenceClassification.from_pretrained(mt5_id, num_labels=5)

# TrainingArguments
training_args = TrainingArguments(
    output_dir= mt5_id,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    report_to=['tensorboard'],
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Fine-tune the model

logging.basicConfig(filename=f'/mnt/disk2/arshia.yousefinezhad/emotion_detection/assets/logs/{model.__class__.__name__}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info(trainer.train())

logging.info(trainer.evaluate())

logging.info(trainer.predict(test_dataset))


model.save_pretrained("assets/finetuned_models/mt5")