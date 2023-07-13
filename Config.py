import torch
import os
from torch.utils.data import DataLoader

class Config:
    # Time & Randomness
    time_string = None
    seed = 3141592

    # Paths
    base_path = None
    data_path = None
    save_path = None
    output_path = None

    # Load Models
    load_ckpt = False
    load_name = None

    # Dataset / DataLoader
    train_loader = None
    valid_loader = None
    test_loader = None
    valid_ratio = 0.2

    # Training Related
    learning_rate = 3e-4
    epochs = 10000
    batch_size = 128
    early_stop = 50

    model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = None

    @classmethod
    def set_path_time(cls, timenow):
        cls.base_path = os.getcwd()

        cls.save_path = os.path.join(cls.base_path, ".model")
        cls.output_path = os.path.join(cls.base_path, "output")
        if not os.path.isdir(cls.save_path):
            os.mkdir(cls.save_path)
        if not os.path.isdir(cls.output_path):
            os.mkdir(cls.output_path)

        cls.time_string = f"{timenow.hour:02d}{timenow.minute:02d}{timenow.month:02d}{timenow.day:02d}"

        print(f"{Config.base_path=}")
        print(f"{Config.data_path=}")
        print(f"{Config.save_path=}")
        print(f"{Config.output_path=}")
        print(f"{Config.time_string=}")

    @classmethod
    def set_train_valid_loader(cls, train_dataset, valid_dataset):
        cls.train_loader = DataLoader(train_dataset, batch_size=cls.batch_size, shuffle=True)
        cls.valid_loader = DataLoader(valid_dataset, batch_size=cls.batch_size, shuffle=True)
        print("Train, Valid DataLoader Complete")


    @classmethod
    def set_model_criterion_optimizer(cls, model_class):
        if cls.load_ckpt:
            cls.load_model(cls.load_name, model_class)
        else:
            cls.model = model_class().to(cls.device)
            print("Training a new model...")
        cls.criterion = torch.nn.CrossEntropyLoss()
        cls.optimizer = torch.optim.Adam(cls.model.parameters(), lr=cls.learning_rate, weight_decay=1e-5)
        print("Model, Criterion, Optimizer Complete")

    @classmethod
    def load_model(cls, model_name, model_class=None):
        if cls.model is None:
            cls.model = model_class().to(cls.device)
        cls.model.load_state_dict(torch.load(os.path.join(cls.save_path, f"model_{model_name}.ckpt")))
        print(f"Model name {model_name} Loaded")

if __name__ == "__main__":
    print(Config.seed)