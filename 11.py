import numpy as np
import pandas as pd
import torch
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


model = torch.load('modelfull.pth').to(device)


df = pd.read_csv('simulation_dataset.csv')

# Определение количества предыдущих координат для входа
num_forecast_steps = 30
n_steps= 30
print('loaded')

# Преобразование датасета
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps - 231953):
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
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float).to(device)
        self.y = torch.tensor(y, dtype=torch.float).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
# Создание объектов Dataset для обучающего и тестового наборов


test_dataset = CustomDataset(X, y)

# Создание DataLoader для обучающего и тестового наборов
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

sequence_to_plot = X_test.squeeze().cpu().numpy()
historical_data = sequence_to_plot[-1]
print(historical_data.shape)

forecasted_values = []



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

last_date =X.index[-1]

# Generate the next 30 dates
future_dates = pd.date_range(start=last_date + pd.DateOffset(1), periods=30)

# Concatenate the original index with the future dates
combined_index = X.index.append(future_dates)






num_forecast_steps = 30

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

# set the size of the plot
plt.rcParams['figure.figsize'] = [14, 4]
test_data = X
# Test data
plt.plot(test_data.index[-100:-30], test_data.Open[-100:-30], label="test_data", color="b")
# reverse the scaling transformation
original_cases = StandardScaler().inverse_transform(np.expand_dims(sequence_to_plot[-1], axis=0)).flatten()

# the historical data used as input for forecasting
plt.plot(test_data.index[-30:], original_cases, label='actual values', color='green')

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

B = torch.tensor([[-3.246398204604679,-2.366775425377713,3.965173483425942],
[-2.853556205148696,-1.648785863503928,4.559857278221214],
[-2.478209676698014,-0.9580682188321658,5.111891504090092],
[-2.122676100166702,-0.2989621513274291,5.627999421009871],
[-1.7845206847922428,0.3325845681723355,6.118468257571938]])

outputs = model(B)
print(outputs)





#input1 = torch.tensor(x[1, 0])
#input2 = torch.tensor(x[1, 1])
#input3 = torch.tensor(x[1, 2])
#input4 = torch.tensor(x[1, 3])
#input5 = torch.tensor(x[1, 4])
#x1 = input1 - input2
#x2 = input2 - input3
#x3 = input3 - input4
#x4 = input4 - input5
# x = torch.cat([x1, x2, x3, x4], 0)
# x = x.view(-1, input_size * (n_steps - 1))
#xf = x1.clone().detach()
#.dot(x2)
#xf.dot(x3)
#xf.dot(x4)
#xf = xf.clone().detach() / 4
#xf = xf.clone().detach() + x[1, 4].clone().detach()
#xf = self.hidden2(xf)
