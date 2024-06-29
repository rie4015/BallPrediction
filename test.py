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


df = pd.read_csv('simulation_dataset11.csv')

# Определение количества предыдущих координат для входа
n_steps = 5
print('loaded')

# Преобразование датасета
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps][['pos_x', 'pos_y', 'pos_z']].values)
        y.append(data[i+1:i + n_steps +1][['pos_x', 'pos_y', 'pos_z']].values)
        # iloc
        # print(i, '/', len(data) - n_steps)
    return np.array(X), np.array(y)

X, y = create_sequences(df, n_steps)




print("X shape:", X.shape)  # Форма входных данных
print("y shape:", y.shape)  # Форма целевых данных

train_coordinates = X
train_trajectories = y
# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Создание DataLoader для обучающего и тестового наборов
train_loader = DataLoader(train_dataset, batch_size =16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

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
    # input_size : number of features in input at each time step
    # hidden_size : Number of LSTM units
    # num_layers : number of LSTM layers
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()  # initializes the parent class nn.Module
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):  # defines forward pass of the neural network
        out, _ = self.lstm(x)
        out = self.linear(out)
        return out






# Определение параметров модели
input_size = 3  # Размер входных данных (pos_x, pos_y, pos_z)
hidden_size = 64 # Размер скрытого слоя
hidden_size2 = 15
output_size = 3  # Размер выходных данных (предсказание следующию координату)
num_layers = 4

# Инициализация модели
#model = NeuralNetwork(input_size, hidden_size1,hidden_size2,output_size).to(device)
#model = NeuralNetwork1(input_size, n_steps).to(device)
model = LSTMModel(input_size, hidden_size,num_layers,output_size).to(device)
print(model)


learning_rate = 0.0001
momentum = 0.1

# Определение функции потерь и оптимизатора
criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


num_epochs = 50
train_hist =[]
test_hist =[]

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

    # print(f'Эпоха {epoch+1}, Потери: {loss.item()}')
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
    # print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss.item()}, Test Loss: {test_loss.item()}')

print('Обучение завершено!')

torch.save(model, 'modelfull1.pth')
# torch.save(model.state_dict(), 'modelw.pth')

x = np.linspace(1,num_epochs,num_epochs)
plt.plot(x,train_hist,scalex=True, label="Training loss")
plt.plot(x, test_hist, label="Test loss")
plt.legend()
plt.show()


model.eval()  # Переводим модель в режим оценки

total_loss = 0.0
predictions = []

with torch.no_grad():
    for batch in test_loader:
        X_batch, y_batch = batch
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        total_loss += loss.item()

        predictions.extend(outputs)  # Сохраняем прогнозы модели



average_loss = total_loss / len(test_loader)
print(f'Средние потери на тестовой выборке: {average_loss}')

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
        predicted_value = model(historical_data_tensor).cpu().numpy()[0, 0]

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


A = torch.tensor([[[14.161222660299964,4.6976924605192,11.365504739004148],
    [-14.189024798435142,4.692473245812175,11.372844079650065],
    [-14.216792998269533,4.687236119188637,11.38001344669554]]])

B = torch.tensor([[-3.246398204604679,-2.366775425377713,3.965173483425942],
[-2.853556205148696,-1.648785863503928,4.559857278221214],
[-2.478209676698014,-0.9580682188321658,5.111891504090092],
[-2.122676100166702,-0.2989621513274291,5.627999421009871],
[-1.7845206847922428,0.3325845681723355,6.118468257571938]]).to(device)

outputs = model(B)
print(outputs)

'14.244528055414136,4.681981192639867,11.387012928936239'
',[0.0,9.0,-1.45887055892039,0.9449651519560566,6.585664705209591]'





