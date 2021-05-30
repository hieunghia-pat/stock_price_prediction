import torch
from torch.nn import MSELoss
from torch.optim import Adadelta
from LSTM_model.data_loader import StockData
from LSTM_model.model import StockModel

if torch.cuda.is_available():
    print("GPUs are available")
    device = "cuda"
else: 
    print("GPU is not available")
    device = "cpu"

company = "TSLA"
concerned_price = ["Close"]
dataset = StockData("stock_data.csv", company, concerned_price)
split_index = int(len(dataset)*0.9)
train_dataset = dataset.data[:split_index]
test_dataset = dataset.data[split_index:]

x_train = []
y_train = []
seq_len = 10
for i in range(seq_len, train_dataset.shape[0]):
    x_train.append(train_dataset[i-seq_len:i].numpy())
    y_train.append(train_dataset[i].numpy())

x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

model = StockModel(len(concerned_price)).to(device)

loss_object = MSELoss().to(device)
optimizer = Adadelta(model.parameters(), lr=2*10e-4)
total_loss = 0

num_dim = len(concerned_price)
for epoch in range(10):
    total_loss = 0
    print(f"Epoch {epoch+1}:")
    for batch, (x, y) in enumerate(zip(x_train, y_train), 1):
        x = x.contiguous().view((1, seq_len, num_dim))
        y = y.contiguous().view(1, 1)
        optimizer.zero_grad()
        output = model(x) # (batch_size, 1)
        loss = loss_object(output, y)
        total_loss += loss
        loss.backward()
        optimizer.step()

        if batch % 101 == 0:
            print(f"    Batch {batch} - Loss: {loss}")
        
    print(f"Total loss: {total_loss / batch}")
    print("==============")

torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict()
}, f"checkpoints/{company}_{epoch+1}_{total_loss / batch}.pth")