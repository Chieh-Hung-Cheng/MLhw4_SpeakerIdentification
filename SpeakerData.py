import random

from Config import Config
import json
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class SpeakerTrainValidDataset(Dataset):
    def __init__(self):
        with open(os.path.join(Config.data_path, "mapping.json")) as f:
            # mapping_dict["speaker2id"], mapping_dict["id2Speaker"]: speaker:"idxxxx" , id:(0~600)
            self.mapping_dict = json.load(f)
        with open(os.path.join(Config.data_path, "metadata.json")) as f:
            # Training: speaker id:str to feature_paths:list in metadata["speakers"]
            self.metadata_dict = json.load(f)

        self.filepaths = []
        self.labels = []
        for speaker, feature_path_dict_list in self.metadata_dict["speakers"].items():
            id = int(self.mapping_dict["speaker2id"][speaker])
            for feature_path_dict in feature_path_dict_list:
                self.filepaths.append(os.path.join(Config.data_path, feature_path_dict["feature_path"]))
                self.labels.append(id)

    def __len__(self):
        assert(len(self.filepaths) == len(self.labels))
        return len(self.filepaths)

    def __getitem__(self, item):
        vector = torch.load(self.filepaths[item])
        label = self.labels[item]
        if len(vector) > 128:
            start_idx = random.randint(0, len(vector)-128)
            return vector[start_idx: start_idx+128], label
        else:
            return vector, label


class SpeakerTestDataset(Dataset):
    def __init__(self):
        with open(os.path.join(Config.data_path, "testdata.json")) as f:
            self.test_dict = json.load(f)

        self.test_filenames = []
        self.vector_tensors = []
        for uttr_dict in tqdm(self.test_dict["utterances"]):
            filename = uttr_dict["feature_path"]
            self.test_filenames.append(filename)
            self.vector_tensors.append(torch.load(os.path.join(Config.data_path, filename)))

    def __len__(self):
        return len(self.test_filenames)

    def __getitem__(self, item):
        filename = self.test_filenames[item]
        vector_tensor = self.vector_tensors[item]
        return vector_tensor, filename


def collate_function(batch):
    vector_tensors, label_tensors = zip(*batch)
    vector_tensors = pad_sequence(vector_tensors, batch_first=True, padding_value=-20)
    label_tensors = torch.LongTensor(label_tensors)
    return vector_tensors, label_tensors


def test_collate_function(batch):
    vector_tensors, filenames = zip(*batch)
    return vector_tensors, filenames


if __name__ == "__main__":
    Config.data_path = r"D:\ML_Dataset\HW4\Dataset"
    x = SpeakerTrainValidDataset()
    pass
