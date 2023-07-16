from Config import Config
from tqdm import tqdm
import torch
import os
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self):
        self.epoch_num = 0
        self.best_loss = float("inf")
        self.best_accuracy = -1.0
        self.strolling = 0

    def train(self):
        # train mode
        Config.model.train()
        # statistic needs
        loss_record = []

        pbar = tqdm(Config.train_loader,
                             desc=f"Train model_{Config.time_string}: {self.epoch_num}/{Config.epochs}")
        for x_b, y_b in pbar:
            # Predict
            x_b = x_b.to(Config.device)
            y_b = y_b.to(Config.device)
            y_pred = Config.model(x_b)
            # Calculate Loss
            loss = Config.criterion(y_pred, y_b)
            # Update Parameter
            Config.optimizer.zero_grad()
            loss.backward()
            Config.optimizer.step()
            Config.scheduler.step()
            # Update Process Bar
            pbar.update()
            pbar.set_postfix({"train_loss": f"{loss:.3f}"})

            loss_record.append(loss.detach().item())
        pbar.close()
        mean_train_loss = sum(loss_record) / len(loss_record)
        return mean_train_loss


    def validate(self):
        # evaluation mode
        Config.model.eval()
        # statistic needs
        loss_record = []
        correct_count = 0
        item_count = 0
        pbar = tqdm(Config.valid_loader,
                             desc=f"Valid model_{Config.time_string}: {self.epoch_num}/{Config.epochs}")
        for x_b, y_b in pbar:
            # Predict
            x_b = x_b.to(Config.device)
            y_b = y_b.to(Config.device)
            y_pred = Config.model(x_b)
            # Calculate Loss
            loss = Config.criterion(y_pred, y_b)
            loss_record.append(loss.detach().item())
            # Calculate Accuracy
            label_pred = torch.argmax(y_pred, dim=1)
            correct_count += torch.sum(label_pred == y_b)
            item_count += len(y_b)
            # Update process bar
            pbar.update()
            pbar.set_postfix({"valid_loss": f"{loss:.3f}",
                              "valid_accuracy": f"{correct_count / item_count:.3%}"})
        pbar.close()
        mean_valid_loss = sum(loss_record) / len(loss_record)
        overall_accuracy = correct_count / item_count
        return mean_valid_loss, overall_accuracy

    def summary(self, train_loss, valid_loss, valid_accuracy):
        def print_Info():
            tqdm.write(f"{train_loss=:.3f}, {valid_loss=:.3f}, {valid_accuracy=:.3%}, "
                  f"best_accuracy={self.best_accuracy:.3%}, strolling={self.strolling}")

        if valid_loss < self.best_loss:
            self.best_loss = valid_loss
            self.best_accuracy = valid_accuracy
            self.save_current_model()
            self.strolling = 0
            print_Info()
            return False
        else:
            self.strolling += 1
            print_Info()
            if self.strolling >= Config.early_stop:
                tqdm.write(f"Early Stopped at epoch {self.epoch_num}")
                return True
            else:
                return False

    def train_loop(self):
        for epoch in range(Config.epochs):
            self.epoch_num = epoch
            # Train
            train_loss = self.train()
            # Validate
            valid_loss, valid_accuracy = self.validate()
            # Summary and decide terminate or not
            if self.summary(train_loss, valid_loss, valid_accuracy):
                break

    def save_current_model(self):
        torch.save(Config.model.state_dict(),
                   os.path.join(Config.save_path, f"model_{Config.time_string}.ckpt"))
