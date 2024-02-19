import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class LeetCodeDataset(Dataset):
    """
    LeetCode dataset that contains over 2000 Python LeetCode examples. The dataset
    can be found here: https://huggingface.co/datasets/RayBernard/leetcode/tree/main
    """

    def __init__(self, data_file='../raw_data/leetcodecomplete.jsonl'):
        """
        Initialize the datset.
        :param data_file: jsonl file containing LeetCode dataset
        """
        self.data_file = data_file
        self.data_df = pd.read_json(path_or_buf=data_file, lines=True)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        pass
        # Todo after we figure out how to feed data to microsofts Phi model