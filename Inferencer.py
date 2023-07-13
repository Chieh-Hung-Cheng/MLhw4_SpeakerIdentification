import torch
import os
import tqdm
import csv
from Config import Config

class Inferencer:
    def __init__(self):
        pass

    def infer(self):
        Config.model.eval()
        preds = []
        ids = []
        for x_b, id in tqdm.tqdm(Config.test_loader):
            x_b = x_b.to(Config.device)
            ids.append(id)
            with torch.no_grad():
                y_pred = Config.model(x_b)
                y_pred = torch.argmax(y_pred, dim=1)
                preds.append(y_pred.detach().cpu())

        ids = [id for sublist in ids for id in sublist]
        preds = torch.cat(preds, dim=0).numpy()
        self.save_pred(preds, ids, os.path.join(Config.output_path, f"pred_{Config.time_string}.csv"))

    def save_pred(self, preds, ids, file):
        ''' Save predictions to specified file '''
        with open(file, 'w', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerow(['Id', 'Category'])
            for i in range(len(preds)):
                writer.writerow([ids[i], preds[i]])


if __name__ == "__main__":
    pass