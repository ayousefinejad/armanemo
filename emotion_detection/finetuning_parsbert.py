import pandas as pd
import torch
from configuration import BaseConfig
from models.pars_bert import TextClassificationParsBert, train, evaluate
from loader.data_loader import create_data_loader, split_data
from transformers import BertConfig, BertTokenizer, BertModel
import time
from tqdm import tqdm
import logging

if __name__ == '__main__':
    logging.basicConfig(filename=f'/Users/arshiayousefi/Desktop/ML/armanemo/assets/logs/ML_Models.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    config = BaseConfig().get_config()

    data = pd.read_csv(config.processed_data_dir)
    data.dropna(axis=0, inplace=True)

    label_list = list(sorted(data['label'].unique()))
    train, valid, test = split_data(data, 'label')

    x_train, y_train = train['text'].to_numpy(), train['label'].to_numpy()
    x_valid, y_valid = valid['text'].to_numpy(), valid['label'].to_numpy()
    x_test, y_test = test['text'].to_numpy(), test['label'].to_numpy()


    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {v: k for k, v in label2id.items()}

    tokenizer = BertTokenizer.from_pretrained(config.parsbert_model)
    bertcofig = BertConfig.from_pretrained(
        config.parsbert_model, **{
            'label2id': label2id,
            'id2label': id2label,
        })

    model = TextClassificationParsBert(bertcofig, config.parsbert_model)
    torch.save(model.state_dict(), '/Users/arshiayousefi/Desktop/ML/armanemo/assets/logs/pretrained_model/parsbert_model.pt')
    model = model.to(config.device)

    train_data_loader = create_data_loader(x_train, y_train, tokenizer, config.max_len, config.train_batch_size, label_list)
    valid_data_loader = create_data_loader(x_valid, y_valid, tokenizer, config.max_len, config.valid_batch_size, label_list)
    test_data_loader = create_data_loader(x_test, y_test, tokenizer, config.max_len, config.test_batch_size, label_list)


    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    total_accu = None

    for epoch in tqdm(range(1, config.num_epochs + 1), desc="Epochs... "):
        epoch_start_time = time.time()
        train(train_data_loader)
        accu_train, loss_train =  evaluate(train_data_loader)
        accu_val, loss_val = evaluate(valid_data_loader)

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