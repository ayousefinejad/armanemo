import argparse
from pathlib import Path


# ============================ Third Party libs ============================
import argparse
from pathlib import Path


class BaseConfig:
    """
        BaseConfig:
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("--device", default="cpu",
                                 help="device to train models on it")
        
        self.parser.add_argument("--model_name", default="/mnt/disk2/LanguageModels/mt5_large",
                                 help="parsbert model pretrained")

        self.parser.add_argument("--max_len", type=int, default=128)

        self.parser.add_argument("--train_batch_size", type=int, default=64)
        self.parser.add_argument("--valid_batch_size", type=int, default=16)
        self.parser.add_argument("--test_batch_size", type=int, default=16)

        self.parser.add_argument("--num_epochs", type=int, default=60)
        self.parser.add_argument("--lr", type=int, default=2e-5)

    def add_path(self) -> None:
        """
        function to add path

        Returns:
            None

        """
        # input dirs
        self.parser.add_argument("--dataset_dir", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/data/train.csv")

        self.parser.add_argument("--processed_data_dir", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/data/train_process.csv")
        
        # output dirs
        # self.parser.add_argument("--parsbert_dir", type=str,
        #             default=Path(__file__).parents[
        #                         2].__str__() + "/assets/pretrained_models/parsbert_model.pth",\
        #                             help="You can download pretraind model from 'HooshvareLab/bert-fa-base-uncased'")
        
        self.parser.add_argument("--finetuned_mt5_dir", type=str,
                    default=Path(__file__).parents[
                                2].__str__() + "/assets/finetuned_models/mt5")
        
        self.parser.add_argument("--log_mt5_dir", type=str,
                    default=Path(__file__).parents[
                                2].__str__() + "/assets/logs/mt5_model.log")


    def get_config(self):
        """

        Returns:

        """
        self.add_path()
        return self.parser.parse_args()
