import torch
import time 
import logging
from tqdm import tqdm


class pytorch_model:
    def __init__(self, model, criterion, optimizer, scheduler):
        """
        Initialize the Emotion Detection Model.

        Parameters:
            - model: The neural network model for emotion detection.
            - criterion: The loss criterion used for training the model.
            - optimizer: The optimization algorithm employed during training.
            - scheduler: Learning rate scheduler for adaptive learning rate adjustments.
            - device: The device (CPU or GPU) on which the model will be trained and evaluated.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.total_accu = None

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def _train(self, dataloader):
        """
        Training method for the emotion detection model.

        Parameters:
            - dataloader: DataLoader containing training data.
            - epoch: Current epoch number.
        """
        self.model.train()
        total_acc, total_count = 0, 0
        log_interval = 500
        start_time = time.time()

        for batch_index, batch_data in enumerate(tqdm(dataloader)):
            # Extract input features and labels from the batch_data dictionary
            input_ids = batch_data['input_ids'].to(self.device)
            attention_mask = batch_data['attention_mask'].to(self.device)
            token_type_ids = batch_data['token_type_ids'].to(self.device)
            targets = batch_data['targets'].to(self.device)


            self.optimizer.zero_grad()
            predicted_targets = self.model(input_ids, attention_mask, token_type_ids)

            loss = self.criterion(predicted_targets, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)

            self.optimizer.step()

            total_acc += (predicted_targets.argmax(1) == targets).sum().item()
            total_count += targets.size(0)

            if batch_index % log_interval == 0 and batch_index > 0:
                elapsed = time.time() - start_time
                logging.info(
                    "| epoch {:3d} | {:5d}/{:5d} batches | time: {:5.2f}s "
                    "| accuracy {:8.3f}".format(
                        epoch, batch_index, len(dataloader), elapsed, total_acc / total_count 
                    )
                )
                total_acc, total_count = 0, 0
                start_time = time.time()


    def _evaluate(self, dataloader):
        """
        Evaluation method for the emotion detection model.

        Parameters:
            - dataloader: DataLoader containing validation or test data.

        Returns:
            - Accuracy on the provided dataset.
        """
        self.model.eval()
        total_acc, total_count = 0, 0

        with torch.no_grad():
            for _ , batch_data in enumerate(tqdm(dataloader)):
                input_ids = batch_data['input_ids'].to(self.device)
                attention_mask = batch_data['attention_mask'].to(self.device)
                token_type_ids = batch_data['token_type_ids'].to(self.device)
                targets = batch_data['targets'].to(self.device)

                predicted_targets = self.model(input_ids, attention_mask, token_type_ids)
                total_acc += (predicted_targets.argmax(1) == targets).sum().item()
                total_count += targets.size(0)
        return total_acc / total_count



    def trainer(self, train_data_loader, valid_data_loader, test_data_loader, epoch):
        """
        Training routine for the emotion detection model.

        Parameters:
            - train_data_loader: DataLoader for the training dataset.
            - valid_data_loader: DataLoader for the validation dataset.
            - test_data_loader: DataLoader for the test dataset.
            - epochs: Number of training epochs.
        """
        # Configure logging
        logging.basicConfig(filename=f'/mnt/disk2/arshia.yousefinezhad/emotion_detection/assets/logs/{self.model.__class__.__name__}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        for _epoch in tqdm(range(1, 1 + epoch), desc="Epochs... "):
            epoch_start_time = time.time()
            self._train(train_data_loader)
            accu_val = self._evaluate(valid_data_loader)

            if self.total_accu is not None and self.total_accu > accu_val:
              self.scheduler.step()
            else:
              self.total_accu = accu_val
            
            logging.info("-" * 59)
            logging.info(
                "| end of epoch {:3d} | time: {:5.2f}s | "
                "valid accuracy {:8.3f} ".format(
                    _epoch, time.time() - epoch_start_time, accu_val
                )
            )
            logging.info("-" * 59)

        logging.info("Checking the results of test dataset.")
        accu_test = self._evaluate(test_data_loader)
        logging.info("test accuracy {:8.3f}".format(accu_test))
        
        # Save Model
        torch.save(self.model.state_dict(), '/mnt/disk2/arshia.yousefinezhad/emotion_detection/assets/pretrained_models/emotion_detection_parsbert.pt')
        logging.info("Model saved successfully.")


    def predict(self, text, tokenizer):
        """
        Make predictions on a given text input.

        Parameters:
            - text: Input text for emotion prediction.
            - tokenizer: Tokenizer used to tokenize the input text.

        Returns:
            - Predicted emotion class.
        """
        with torch.no_grad():
            marked_text = "[CLS] " + text + " [SEP]"
            tokenized_text = tokenizer.tokenize(marked_text)
            indexed_token = tokenizer.convert_tokens_to_ids(tokenized_text)

            input_ids = [1] * len(tokenized_text)
            token_tensor = torch.tensor([indexed_token])
            input_tensors = torch.tensor([input_ids])
            output = self.model(token_tensor, input_tensors)
            return output.argmax(1).item()
