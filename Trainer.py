from Config import Config
from tqdm import tqdm
import torch
import os
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self):
        pass

    def train(self):
        best_loss = None
        best_accuracy = -1.0
        early_stop_count = 0
        writer = SummaryWriter()

        epochs_pbar = tqdm(range(Config.epochs), desc=f"model_{Config.time_string}")
        for epoch in epochs_pbar:
            # train
            Config.model.train()
            loss_record = []
            for x_b, y_b in Config.train_loader:
                x_b = x_b.to(Config.device)
                y_b = y_b.to(Config.device)
                y_pred = Config.model(x_b)

                loss = Config.criterion(y_pred, y_b)
                loss.backward()
                Config.optimizer.step()
                Config.scheduler.step()
                Config.optimizer.zero_grad()


                loss_record.append(loss.detach().item())

            mean_train_loss = sum(loss_record) / len(loss_record)
            writer.add_scalar("TrainLoss", mean_train_loss, epoch)

            # valid
            if epoch % Config.valid_cycle == 0:
                Config.model.eval()
                loss_record = []
                correct_count = 0
                item_count = 0
                for x_b, y_b in Config.valid_loader:
                    x_b = x_b.to(Config.device)
                    y_b = y_b.to(Config.device)
                    y_pred = Config.model(x_b)

                    loss = Config.criterion(y_pred, y_b)
                    loss_record.append(loss.detach().item())

                    label_pred = torch.argmax(y_pred, dim=1)
                    label_truth = y_b
                    # label_truth = torch.argmax(y_b, dim=1)
                    correct_count += torch.sum(label_pred == label_truth)
                    item_count += len(y_b)

                mean_valid_loss = sum(loss_record) / len(loss_record)
                writer.add_scalar("ValidLoss", mean_valid_loss, epoch)

                # Update tqdm
                epochs_pbar.set_postfix({"running_train_loss": f"{mean_train_loss:.4f}",
                                         "running_valid_loss": f"{mean_valid_loss:.4f}",
                                         "early_countdown": f"{Config.early_stop - early_stop_count}",
                                         "accuracy": f"{correct_count / item_count:.3%}",
                                         "best_accuracy": f"{best_accuracy:.3%}"
                                         })

                # Saving and Early Stopping
                if (best_loss is None) or (mean_valid_loss < best_loss):
                    best_loss = mean_valid_loss
                    best_accuracy = correct_count / item_count
                    self.save_current_model(best_loss)
                    early_stop_count = 0
                else:
                    early_stop_count += 1
                    if early_stop_count >= Config.early_stop:
                        print(f"Early Stopped at {epoch=}")
                        break


    def save_current_model(self, loss):
        torch.save(Config.model.state_dict(),
                   os.path.join(Config.save_path, f"model_{Config.time_string}.ckpt"))
