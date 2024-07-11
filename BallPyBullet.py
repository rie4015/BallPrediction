import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch  # Library for implementing Deep Neural Network
import torch.nn as nn
import torch.optim as optim
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

n_steps = 24 * 2 - 1


def var2str(variable):

    return [global_var for global_var in globals() if id(variable) == id(globals()[global_var])]


def load(data):
    return np.load(f'{var2str(data)}.npy')


X_train = []
X2_train = []
X3_train = []
X_test = []
X2_test = []
X3_test = []
y_train =[]
y2_train = []
y3_train =[]
y_test =[]
y2_test = []
y3_test = []



# scaling dataset

scaler = MinMaxScaler(feature_range=(0,1))

X_train = load(X_train)
X2_train = load(X2_train)
X3_train = load(X3_train)
X_test = load(X_test)
X2_test = load(X2_test)
X3_test = load(X3_test)
y_train = load(y_train)
y2_train = load(y2_train)
y3_train = load(y3_train)
y_test = load(y_test)
y2_test = load(y2_test)
y3_test = load(y3_test)





# x = X[500:1000,0]
# y = X[500:1000:,1]
# z = X[500:1000:,2]
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(x, y, z, label='parametric curve')
# plt.show()

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float).to(device)
        self.y = torch.tensor(y, dtype=torch.float).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Создание объектов Dataset для обучающего и тестового наборов


train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)

train_dataset2 = CustomDataset(X2_train, y2_train)
test_dataset2 = CustomDataset(X2_test, y2_test)

train_dataset3 = CustomDataset(X3_train, y3_train)
test_dataset3 = CustomDataset(X3_test, y3_test)

batch_size = 8

# Создание DataLoader для обучающего и тестового наборов
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

train_loader2 = DataLoader(train_dataset2, batch_size=batch_size, shuffle=True)
test_loader2 = DataLoader(test_dataset2, batch_size=batch_size, shuffle=False)

train_loader3 = DataLoader(train_dataset3, batch_size=batch_size, shuffle=True)
test_loader3 = DataLoader(test_dataset3, batch_size=batch_size, shuffle=False)

print()



class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(NeuralNetwork, self).__init__()
        # self.dis = nn.PairwiseDistance(p=2, keepdim=True)
        # self.flat = nn.Flatten()
        self.input_layer = nn.Linear(n_steps * input_size, hidden_size1)
        # self.hidden1 = nn.Linear(hidden_size1, hidden_size1)
        # self.hidden2 = nn.Linear(input_size * n_steps, hidden_size2)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size1, output_size)

    def forward(self, x):
        x.view(-1, 15)
        # print(x)
        # x = self.flat(x)
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.output(x)
        return x


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.dropout = nn.Dropout(dropout_rate)
        self.batchnorm = nn.BatchNorm1d(n_steps)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        out = self.dropout(out)
        out = self.batchnorm(out)
        out = self.fc2(out)
        return out


class LSTMModel1(nn.Module):
    # input_size : number of features in input at each time step
    # hidden_size : Number of LSTM units
    # num_layers : number of LSTM layers
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel1, self).__init__()  # initializes the parent class nn.Module
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):  # defines forward pass of the neural network
        out, _ = self.lstm(x)
        out = self.linear(out)
        return out


input_size = 1
output_size = 1
hidden_size = 96
num_layers = 2
dropout_rate = 0.1

# Инициализация модели
# model = NeuralNetwork(input_size, hidden_size1,hidden_size2,output_size).to(device)
# model = NeuralNetwork1(input_size, n_steps).to(device)
model1 = LSTMModel1(input_size, hidden_size, num_layers).to(device)
print(model1)
model2 = LSTMModel1(input_size, hidden_size, num_layers).to(device)
model3 = LSTMModel1(input_size, hidden_size, num_layers).to(device)

learning_rate = 1e-3
momentum = 0.1

# Определение функции потерь и оптимизатора
criterion = nn.L1Loss(reduction='mean')
optimizer1 = optim.Adam(model1.parameters(), lr=learning_rate)
optimizer2 = optim.Adam(model2.parameters(), lr=learning_rate)
optimizer3 = optim.Adam(model3.parameters(), lr=learning_rate)

num_epochs = 15
print('batch_size:', batch_size)
print('num_epochs:', num_epochs)

def train(model, train_loader, test_loader, optimizer):
    train_hist = []
    test_hist = []
    for epoch in range(num_epochs):
        start = time.time()
        total_loss = 0.0
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(train_loader)
        train_hist.append(average_loss)

        # print(f'Эпоха {epoch+1}, Потери: {loss.item()}')
        model.eval()
        with torch.no_grad():
            total_test_loss = 0.0
            for inputs, targets in test_loader:
                outputs = model(inputs)
                test_loss = criterion(outputs, targets)
                total_test_loss += test_loss.item()
            # Calculate average test loss and accuracy
            average_test_loss = total_test_loss / len(test_loader)
            test_hist.append(average_test_loss)
        end = time.time() - start
        print(
            f'Epoch [{epoch + 1}/{num_epochs}] - Training Loss: {average_loss:.4f}, Test Loss: {average_test_loss:.4f}', 'time:', end)
        # print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss.item()}, Test Loss: {test_loss.item()}')

    print('Обучение завершено!')
    return train_hist, test_hist




def showg(train,test):
    x = np.linspace(1, num_epochs, num_epochs)
    plt.plot(x, train, scalex=True, label="Training loss")
    plt.plot(x, test, label="Test loss")
    plt.legend()
    plt.savefig(f'{var2str(train)}.png')
    plt.show()
    return

train_hist1, test_hist1 = train(model1, train_loader, test_loader, optimizer1)

showg(train_hist1, test_hist1)

train_hist2, test_hist2 = train(model2, train_loader2, test_loader2, optimizer2)

showg(train_hist2, test_hist2)
train_hist3, test_hist3 = train(model3, train_loader3, test_loader3, optimizer3)

showg(train_hist3, test_hist3)

torch.save(model1.state_dict(), 'model1.pth')
# torch.save(model.state_dict(), 'modelw.pth')
torch.save(model2.state_dict(), 'model2.pth')
torch.save(model3.state_dict(), 'model3.pth')

torch.save(model1, 'modelfull1.pth')
torch.save(model2, 'modelfull2.pth')
torch.save(model3, 'modelfull3.pth')


test_hist = []



model1.eval()
model2.eval()
model3.eval()# Переводим модель в режим оценки

num_forecast_steps = 480



#sequence_to_plot = []
df = pd.read_csv('simulation_dataset.csv')

def pir(X_test, Y_test):
    x_test = torch.tensor(X_test, dtype=torch.float)
    y_test = torch.tensor(Y_test, dtype=torch.float)
    # Convert to NumPy and remove singleton dimensions
    sequence_to_plot = x_test.squeeze().cpu().numpy()
    # test_data = df[0:480][['pos_x']].values
    # test_time = df[0:480][['time_step']].values
    sequence_to_plot2 = y_test.squeeze().cpu().numpy()




    #print(sequence_to_plot[-1])
    # Use the last 30 data points as the starting point
    historical_data = sequence_to_plot[18]
    test_data = historical_data
    print(historical_data.shape)

    # Initialize a list to store the forecasted values
    forecasted_values = []

    # Use the trained model to forecast future values
    with torch.no_grad():
        for _ in range(num_forecast_steps * 2):
            # Prepare the historical_data tensor
            historical_data_tensor = torch.as_tensor(historical_data).view(1, -1, 1).float().to(device)
            # Use the model to predict the next value
            predicted_value = model1(historical_data_tensor).cpu().numpy()[0, 0]

            # Append the predicted value to the forecasted_values list
            forecasted_values.append(predicted_value[0])

            # Update the historical_data sequence by removing the oldest value and adding the predicted value
            historical_data = np.roll(historical_data, shift=-1)
            historical_data[-1] = predicted_value
    # Generate futute dates
    print(historical_data.shape)
    last_date = len(sequence_to_plot[18]) - 1
    future_dates = []
    future_dates2 = []

    dates = []
    for i in range(num_forecast_steps * 2):
        future_dates.append(i)
    for i in range(num_forecast_steps):
        future_dates2.append(i + 1)
    for i in range (num_forecast_steps * 2):
        dates.append(last_date + i)

    #scaler = joblib.load('std_scaler.bin')
    #(historical_data.flatten()).reshape(-1, 1)

    #historical_data = scaler.inverse_transform(x2)
    print(forecasted_values)


    original_cases = historical_data
    #    (sequence_to_plot)[-1]

    #plt.plot(test_time, test_data, label= 'test')

    plt.plot(future_dates, test_data, label='test_data',color='green')

    plt.plot(future_dates, sequence_to_plot2[18], label='test_data')

    plt.plot(future_dates, original_cases, label= 'historical_data')

    #plt.plot(dates, sequence_to_plot[-1], label= 'historical_data',color='green')
    #print(sequence_to_plot2[18],sequence_to_plot2[18])
    plt.plot(dates, forecasted_values,
             label= 'forecasted_values', color='red')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


pir(X_test, y_test)
pir(X2_test, y2_test)
pir(X3_test, y3_test)



X_test = torch.tensor(X_test, dtype=torch.float)
sequence_to_plot = X_test.squeeze().cpu().numpy()
last_date = len(sequence_to_plot[-1]) - 1
future_dates = []
dates = []
for i in range(47):
    future_dates.append(last_date + i)
for i in range(num_forecast_steps * 2):
    dates.append(last_date + i - num_forecast_steps + 1)



plt.plot(future_dates, sequence_to_plot[-1], label= 'historical_data',color='green')
plt.show()




