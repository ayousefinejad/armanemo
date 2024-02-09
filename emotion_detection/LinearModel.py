import pandas as pd
import torch
import time
import logging
from models.models_DL import TextClassificationLinear, train, evaluate
from preprocess.process import process_length_words
from loader.data_loader import split_data
from loader.data_loader_deepmodel import create_data_loader, tokens_ids


logging.basicConfig(filename=f'/Users/arshiayousefi/Desktop/ML/armanemo/assets/logs/Linear_model.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
if __name__ == '__name__':
    device = "cpu"
    embedd_dim = 512
    hidden_dim = 256
    pad_idx = 0
    n_epochs = 10
    # Hyperparameters
    EPOCHS = 10  # epoch
    lr = 1e-1  # learning rate
    batch_size = 128
    MAX_VOCAB = 50_000

    data = pd.read_csv("/Users/arshiayousefi/Desktop/ML/armanemo/data/train_process2.csv")
    data = process_length_words(data)
    label_list = list(sorted(data['label'].unique()))
    train_data, valid_data, test_data = split_data(data)

    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {v: k for k, v in label2id.items()}
    
    # Assuming 'class_counts' is a list or array with the number of samples for each class
    class_counts = train_data.label.value_counts().to_dict()

    total_samples = sum(class_counts.values())
    class_weights = {class_label: total_samples / class_count for class_label, class_count in enumerate(class_counts.values())}

    train_data, valid_data, test_data, token2id = tokens_ids(train_data, valid_data, test_data, MAX_VOCAB)
    train_loader, valid_loader, test_loader = create_data_loader(train_data, valid_data, test_data, label2id, batch_size, device)
    vocab_size = len(token2id) + 1

    output_dim = len(label2id)
    model = model = TextClassificationLinear(vocab_size, embedd_dim, hidden_dim, output_dim, pad_idx).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(list(class_weights.values())).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    total_accu = None

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        train(train_loader)
        accu_train, loss_train = evaluate(train_loader)
        accu_val, loss_val = evaluate(valid_loader)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
        logging.info("-" * 59)
        logging.info(
            "| end of epoch {:3d} | time: {:5.2f}s | "
            " | train accuracy {:8.3f} | loss train {:8.3f} | valid accuracy {:8.3f} | loss validation {:8.3f}".format(
                epoch, time.time() - epoch_start_time, accu_train, loss_train, accu_val, loss_val
            )
        )
        logging.info("-" * 59)