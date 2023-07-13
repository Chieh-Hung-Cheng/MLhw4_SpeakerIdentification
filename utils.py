import torch
import numpy as np
from Config import Config
from torch.utils.data import random_split
import os

def fix_randomness(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.seed(seed)
        torch.cuda.manual_seed_all(seed)
    print("Randomness Fixed")


def split_train_valid(train_dataset):
    original_train_length = len(train_dataset)
    actual_valid_length = int(original_train_length * Config.valid_ratio)
    actual_train_length = original_train_length - actual_valid_length
    return random_split(train_dataset, [actual_train_length, actual_valid_length],
                        generator=torch.Generator().manual_seed(Config.seed))

def load_model(model_name):
    Config.model.load_state_dict(torch.load(os.path.join(Config.save_path, f"model_{model_name}.ckpt")))
    print(f"Loaded model_{model_name}.ckpt into Config.model")
