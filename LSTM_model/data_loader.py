import torch 
from torch.utils.data import Dataset 
import pandas as pd

class StockData(Dataset):
    def __init__(self, data_dir, company, concerned_price):
        super(StockData, self).__init__()

        data = pd.read_csv(data_dir)
        data = data[data["Stock"] == company][concerned_price].values
        # data = data / da
        self.data = torch.tensor(data, dtype=torch.float32)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]