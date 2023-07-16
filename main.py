from Config import Config
from Utils import Utils
from SpeakerData import SpeakerTrainValidDataset, SpeakerTestDataset, collate_function, test_collate_function
from SpeakerClassifier import SpeakerClassifier
from Trainer import Trainer


def main():
    # initialize
    Utils.initialize_time_path(local_flag=True)
    Utils.fix_randomness()
    # data
    trainset = SpeakerTrainValidDataset()
    trainset, validset = Utils.split_train_valid_dataset(trainset)
    Utils.set_train_valid_loader(trainset, validset, collate_function)
    # model
    Utils.set_model_related(SpeakerClassifier)
    # train
    trainer = Trainer()
    trainer.train_loop()


if __name__ == "__main__":
    main()