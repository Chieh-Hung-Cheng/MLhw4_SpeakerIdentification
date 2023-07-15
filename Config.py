import torch


class Config:
    # Time & Randomness
    time_string = None
    seed = 87

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
    valid_ratio = 0.1
    num_worker = 8

    # Training Related
    learning_rate = 1e-3
    epochs = 70000
    batch_size = 8
    early_stop = 50
    valid_cycle = 2
    warmup_steps = 1000

    model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = None
    scheduler = None



if __name__ == "__main__":
    print(Config.seed)
