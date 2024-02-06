from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset, ClassLabel, DatasetDict


def huggingface_split_data(data):
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
    return train_dataset, test_dataset, val_dataset



def huggingface_label2int(train_dataset, test_dataset, val_dataset, unique_labels_train):
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

    return train_dataset, test_dataset, val_dataset 
