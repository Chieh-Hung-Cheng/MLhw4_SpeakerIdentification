import random

from Config import Config
import json
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class SpeakerFileParser():
    def __init__(self):
        with open(os.path.join(Config.data_path, "mapping.json")) as f:
            # mapping_dict["speaker2id"], mapping_dict["id2Speaker"]: speaker:"idxxxx" , id:(0~600)
            self.mapping_dict = json.load(f)
        with open(os.path.join(Config.data_path, "metadata.json")) as f:
            # Training: speaker id:str to feature_paths:list in metadata["speakers"]
            self.metadata_dict = json.load(f)
        with open(os.path.join(Config.data_path, "testdata.json")) as f:
            # test_dict["utterances"]
            self.test_dict = json.load(f)

    def parse_train_data(self):
        vector_list = []
        label_list = []
        for speaker, feature_path_dict_list in tqdm(self.metadata_dict["speakers"].items()):
            id = int(self.mapping_dict["speaker2id"][speaker])
            for feature_path_dict in feature_path_dict_list:
                feature_vector = torch.load(os.path.join(Config.data_path, feature_path_dict["feature_path"]))
                vector_list.append(feature_vector)
                label_list.append(id)
        return vector_list, label_list

    def parse_test_data(self):
        pass

    def my_dict_tester(self):
        ret_int = 0
        num_data = 0
        for speaker, feature_path_dict_list in tqdm(self.metadata_dict["speakers"].items()):
            id = int(self.mapping_dict["speaker2id"][speaker])
            for feature_path_dict in feature_path_dict_list:
                num_data+=1
                if feature_path_dict["mel_len"]<128:
                    print(feature_path_dict["mel_len"])
                    ret_int+=1
        return ret_int, num_data


class SpeakerDataset(Dataset):
    def __init__(self, vector_list, label_list, segment_length=128):
        self.vector_tensors = vector_list
        self.segment_length = segment_length
        if label_list is not None:
            self.label_tensors = torch.LongTensor(label_list)
        else:
            self.label_tensors = None

    def __len__(self):
        if self.label_tensors is not None:
            assert len(self.label_tensors) == len(self.vector_tensors)
        return len(self.vector_tensors)

    def __getitem__(self, item):
        if self.label_tensors is not None:
            # Training and Validation
            vector_length = len(self.vector_tensors[item])
            if vector_length > self.segment_length:
                start_idx = random.randint(0, vector_length - self.segment_length)
            else:
                start_idx = 0
            return self.vector_tensors[item][start_idx:], self.label_tensors[item]
        else:
            # Testing
            return self.vector_tensors[item]


def collate_function(batch):
    vector_tensors, label_tensors = zip(*batch)
    vector_tensors = pad_sequence(vector_tensors, batch_first=True, padding_value=-20)
    label_tensors = torch.LongTensor(label_tensors)
    return vector_tensors, label_tensors


if __name__ == "__main__":
    Config.data_path = r"D:\ML_Dataset\HW4\Dataset"
    sfp = SpeakerFileParser()
    sfp.parse_test_data()
