import os
import pandas as pd
import pickle as pkl
from torch.utils.data import Dataset


class AssetSerialData(Dataset):
    def __init__(self, data_file, time_series_length):
        self.data_file = data_file
        self.time_series_length = time_series_length

        self.construct_data()

    def construct_data(self):
        with open(self.data_file, 'rb') as reader:
            self.data = pkl.load(reader)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
