import torch
import os
import tqdm
import csv
from Config import Config
import json

class Inferencer:
    def __init__(self):
        with open(os.path.join(Config.data_path, "mapping.json")) as f:
            # mapping_dict["speaker2id"], mapping_dict["id2Speaker"]: speaker:"idxxxx" , id:(0~600)
            self.mapping_dict = json.load(f)

    def infer(self):
        Config.model.eval()
        preds = []
        filename_lst = []
        for x_b, filename_b in tqdm.tqdm(Config.test_loader):
            x_b = x_b[0].unsqueeze(0).to(Config.device)
            filename_lst.append(filename_b[0])
            with torch.no_grad():
                y_pred = Config.model(x_b)
                y_pred = torch.argmax(y_pred, dim=1)
                preds.append(y_pred.detach().cpu())

        preds = torch.cat(preds, dim=0).numpy()
        self.save_pred(preds, filename_lst, os.path.join(Config.output_path, f"pred_{Config.time_string}.csv"))

    def save_pred(self, preds, filenames, file):
        ''' Save predictions to specified file '''
        with open(file, 'w', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerow(['Id', 'Category'])
            for i in range(len(preds)):
                speaker = self.mapping_dict["id2speaker"][str(preds[i])]
                writer.writerow([filenames[i], speaker])


if __name__ == "__main__":
    pass