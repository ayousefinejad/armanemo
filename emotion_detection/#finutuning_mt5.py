import logging
import pandas as pd

from transformers import (
    T5TokenizerFast,
    AutoTokenizer,
    T5EncoderModel,
    TrainingArguments,
    Trainer,
    AutoConfig,
)

from loader.huggingface_loader import huggingface_split_data, huggingface_label2int
from preprocess.huggingface import huggingface_proceed
from models.helper import huggingface_compute_metrics
from configuration import BaseConfig


if __name__ == '__main__':
    config = BaseConfig().get_config()

    DATA = pd.read_csv(config.processed_data_dir)
    DATA = DATA.sample(1000)

    TRAIN_DATASET, VAL_DATASET, TEST_DATASET = huggingface_split_data(DATA)

    # convert dataset to pandas dataset
    DF_TRAIN = TRAIN_DATASET.to_pandas()
    DF_VAL = VAL_DATASET.to_pandas()
    DD_TEST = TEST_DATASET.to_pandas()

    #  unique labels
    UNIQUE_LABELS_TRAIN = list(sorted(DF_TRAIN['emotion'].unique()))
    
    TRAIN_DATASET, VAL_DATASET, TEST_DATASET = huggingface_label2int(TRAIN_DATASET, VAL_DATASET, TEST_DATASET, UNIQUE_LABELS_TRAIN)

    LABEL_LIST = list(sorted(DF_TRAIN['emotion'].unique()))

    LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
    ID2LABEL = {v: k for k, v in LABEL2ID.items()}

    # Update the model's configuration with the id2label mapping
    CONFIG = AutoConfig.from_pretrained(config.model_name)
    CONFIG.update({"id2label": ID2LABEL})

    # Preprocessing
    TOKENIZER = AutoTokenizer.from_pretrained(config.model_name)
    TRAIN_DATASET, VAL_DATASET, TEST_DATASET = huggingface_proceed(TRAIN_DATASET, VAL_DATASET, TEST_DATASET, TOKENIZER)

    # Model
    MODEL = T5EncoderModel.from_pretrained(config.model_name, num_labels=5)

    # TrainingArguments
    TRAINING_ARGS = TrainingArguments(
        output_dir= config.model_name,
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
    TRAINER = Trainer(
        model=MODEL,
        args=TRAINING_ARGS,
        train_dataset=TRAIN_DATASET,
        eval_dataset=VAL_DATASET,
        compute_metrics=huggingface_compute_metrics,
    )


    # Fine-tune the model
    logging.basicConfig(filename=config.log_mt5_dir, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(TRAINER.train())
    logging.info(TRAINER.evaluate())
    logging.info(TRAINER.predict(TEST_DATASET))
    # MODEL.save_pretrained(config.finetuned_mt5_dir)
    config.finetuned_mt5_dir

    
    



