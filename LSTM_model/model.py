from torch import nn 
import torch.nn.functional as F

class StockModel(nn.Module):
    def __init__(self, in_features):
        super(StockModel, self).__init__()

        self.lstm_1 = nn.LSTM(input_size=in_features, hidden_size=128, num_layers=1)
        self.lstm_2 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1)
        self.fc_1 = nn.Linear(in_features=64, out_features=25)
        self.fc_2 = nn.Linear(in_features=25, out_features=1)

    def forward(self, x):
        # x: (batch_size, seq_len, dims)
        x = x.permute((1, 0, 2))
        x, _ = F.sigmoid(self.lstm_1(x))
        x, _ = F.sigmoid(self.lstm_2(x))
        x = x.permute((1, 0, 2)) # (batch_size, seq_len, 64)
        x = self.fc_1(x)
        x = self.fc_2(x)

        return x[:, -1].contiguous().view(-1, 1) # (batch_size, 1)
