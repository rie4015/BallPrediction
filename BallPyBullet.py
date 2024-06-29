import sklearn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib.dates as mdates # Formatting dates
import seaborn as sns # Visualization
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch # Library for implementing Deep Neural Network
from torch.utils.data import Dataset, DataLoader


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


df = pd.read_csv('simulation_dataset.csv')

# Определение количества предыдущих координат для входа
n_steps = 48
print('loaded')

# Преобразование датасета
def create_sequencesx(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps][['pos_x']].values)
        y.append(data[i+1:i + n_steps +1][['pos_x']].values)
        # iloc
        # print(i, '/', len(data) - n_steps)
    return np.array(X), np.array(y)


def create_sequencesy(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps][['pos_y']].values)
        y.append(data[i+1:i + n_steps +1][['pos_y']].values)
        # iloc
        # print(i, '/', len(data) - n_steps)
    return np.array(X), np.array(y)


def create_sequencesz(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps][['pos_z']].values)
        y.append(data[i+1:i + n_steps +1][['pos_z']].values)
        # iloc
        # print(i, '/', len(data) - n_steps)
    return np.array(X), np.array(y)


Xx, yx = create_sequencesx(df, n_steps)
Xy, yy = create_sequencesy(df, n_steps)
Xz, yz = create_sequencesz(df, n_steps)




print("X shape:", Xx.shape)  # Форма входных данных
print("y shape:", yx.shape)  # Форма целевых данных

train_coordinates = Xx
train_trajectories = yx
# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(Xx, yx, test_size=0.2, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(Xy, yy, test_size=0.2, random_state=42)
X3_train, X3_test, y3_train, y3_test = train_test_split(Xz, yz, test_size=0.2, random_state=42)

print("Размер обучающего набора X:", X_train.shape)
print("Размер тестового набора X:", X_test.shape)
print("Размер обучающего набора y:", y_train.shape)
print("Размер тестового набора y:", y_test.shape)

#x = X[500:1000,0]
#y = X[500:1000:,1]
#z = X[500:1000:,2]
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot(x, y, z, label='parametric curve')
#plt.show()

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

# Создание DataLoader для обучающего и тестового наборов
train_loader = DataLoader(train_dataset, batch_size =8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

train_loader2 = DataLoader(train_dataset2, batch_size =8, shuffle=True)
test_loader2 = DataLoader(test_dataset2, batch_size=8, shuffle=False)

train_loader3 = DataLoader(train_dataset3, batch_size =8, shuffle=True)
test_loader3 = DataLoader(test_dataset3, batch_size=8, shuffle=False)


print()





class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1,hidden_size2, output_size):
        super(NeuralNetwork, self).__init__()
        #self.dis = nn.PairwiseDistance(p=2, keepdim=True)
        #self.flat = nn.Flatten()
        self.input_layer = nn.Linear(n_steps*input_size, hidden_size1)
        #self.hidden1 = nn.Linear(hidden_size1, hidden_size1)
        #self.hidden2 = nn.Linear(input_size * n_steps, hidden_size2)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size1, output_size)

    def forward(self, x):
        x.view(-1, 15)
        #print(x)
        #x = self.flat(x)
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
hidden_size = 64
num_layers = 2
dropout_rate = 0.1

# Инициализация модели
#model = NeuralNetwork(input_size, hidden_size1,hidden_size2,output_size).to(device)
#model = NeuralNetwork1(input_size, n_steps).to(device)
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


num_epochs = 30


def train(model,train_loader,test_loader,optimizer):
    train_hist = []
    test_hist = []
    for epoch in range(num_epochs):
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

        #print(f'Эпоха {epoch+1}, Потери: {loss.item()}')
        model.eval()
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
        print(f'Epoch [{epoch + 1}/{num_epochs}] - Training Loss: {average_loss:.4f}, Test Loss: {average_test_loss:.4f}')
        #print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss.item()}, Test Loss: {test_loss.item()}')

    print('Обучение завершено!')
    return train_hist,test_hist


train_hist1, test_hist1 = train(model1,train_loader,test_loader,optimizer1)

train_hist2, test_hist2 = train(model2,train_loader2,test_loader2,optimizer2)
train_hist3, test_hist3 = train(model3,train_loader3,test_loader3,optimizer3)



torch.save(model1, 'modelfull1.pth')
# torch.save(model.state_dict(), 'modelw.pth')
torch.save(model1, 'modelfull2.pth')
torch.save(model1, 'modelfull3.pth')

x = np.linspace(1, num_epochs, num_epochs)
plt.plot(x,train_hist1,scalex=True, label="Training loss")
plt.plot(x, test_hist1, label="Test loss")
plt.legend()
plt.show()

x = np.linspace(1,num_epochs,num_epochs)
plt.plot(x,train_hist2,scalex=True, label="Training loss")
plt.plot(x, test_hist2, label="Test loss")
plt.legend()
plt.show()

x = np.linspace(1,num_epochs,num_epochs)
plt.plot(x,train_hist3,scalex=True, label="Training loss")
plt.plot(x, test_hist3, label="Test loss")
plt.legend()
plt.show()


model1.eval()  # Переводим модель в режим оценки





num_forecast_steps = 30

# Convert to NumPy and remove singleton dimensions
sequence_to_plot = X_test.squeeze().gpu().numpy()

# Use the last 30 data points as the starting point
historical_data = sequence_to_plot[-1]
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
last_date = X_test.index[-1]

# Generate the next 30 dates
future_dates = pd.date_range(start=last_date + pd.DateOffset(1), periods=30)

# Concatenate the original index with the future dates
combined_index = X_test.index.append(future_dates)

# set the size of the plot
plt.rcParams['figure.figsize'] = [14, 4]

# Test data
plt.plot(X_test.index[-100:-30], X_test.Open[-100:-30], label="test_data", color="b")
# reverse the scaling transformation
original_cases = StandardScaler().inverse_transform(np.expand_dims(sequence_to_plot[-1], axis=0)).flatten()

# the historical data used as input for forecasting
plt.plot(X_test.index[-30:], original_cases, label='actual values', color='green')

# Forecasted Values
# reverse the scaling transformation
forecasted_cases = StandardScaler().inverse_transform(np.expand_dims(forecasted_values, axis=0)).flatten()
# plotting the forecasted values
plt.plot(combined_index[-60:], forecasted_cases, label='forecasted values', color='red')

plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.title('Time Series Forecasting')
plt.grid(True)





