import torch
from sklearn.metrics import accuracy_score, f1_score
import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")

def classes_weight(data, device):
    """
    Calculate class weights based on the distribution of classes in the dataset.

    Args:
        data (DataFrame): Pandas DataFrame containing a column named 'emotion' with class labels.
        device (torch.device): The device (CPU or GPU) where the PyTorch tensor should be placed.

    Returns:
        torch.Tensor: A tensor containing class weights.
    """
    label_list = list(set(data.emotion))
    label2id = {label: i for i, label in enumerate(label_list)}
    numeric_labels = [label2id[label] for label in data.emotion.tolist()]
    label_list_tensor = torch.tensor(numeric_labels).to(device)
    class_counts = torch.bincount(label_list_tensor)
    total_samples = len(label_list)
    class_weights = total_samples / (len(class_counts) * class_counts.float())
    class_weights = class_weights / class_weights.sum()
    return class_weights


def huggingface_compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    f1 = f1_score(labels, predictions, average="weighted")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1}
