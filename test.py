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
n_steps = 5
print('loaded')

# Преобразование датасета
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps - 319705):
        X.append(data[i:i + n_steps][['pos_x', 'pos_y', 'pos_z']].values)
        y.append(data.iloc[i + n_steps][['pos_x', 'pos_y', 'pos_z']].values)

        print(i, '/', len(data) - n_steps)
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
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print()





class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.dis = nn.PairwiseDistance()
        self.hidden = nn.Linear(input_size * (n_steps - 1), hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        input1 = x[1, 0]
        input2 = x[1, 1]
        input3 = x[1, 2]
        input4 = x[1, 3]
        input5 = x[1, 4]
        print('i',input1)
        print('i', input2)
        print('i', input3)
        print('i', input4)
        print('i', input5)
        for i in range(5)
            x1 = self.dis(input1, input2)
            x2 = self.dis(input2, input3)
            x3 = self.dis(input3, input4)
            x4 = self.dis(input4, input5)
        print('x[1]',x[1])
        print(x1)
        x = torch.cat([x1, x2, x3, x4], 0)
        print(x)
        x = x.view(-1, input_size * (n_steps - 1))
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x


class TrajectoryPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(TrajectoryPredictor, self).__init__()
        self.rnn = nn.RNN(input_size * n_steps, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(-1, input_size * n_steps)
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out





# Определение параметров модели
input_size = 3  # Размер входных данных (pos_x, pos_y, pos_z)
hidden_size = 64 # Размер скрытого слоя
output_size = 3  # Размер выходных данных (предсказание следующию координату)
num_layers = 2

# Инициализация модели
model = NeuralNetwork(input_size, hidden_size,output_size).to(device)
print(model)


learning_rate = 1e-4
momentum = 0.1

# Определение функции потерь и оптимизатора
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        X_batch, y_batch = batch

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    print(f'Эпоха {epoch+1}, Потери: {loss.item()}')

print('Обучение завершено!')

torch.save(model, 'modelfull.pth')
# torch.save(model.state_dict(), 'modelw.pth')


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

A = torch.tensor([[[14.161222660299964,4.6976924605192,11.365504739004148],
    [-14.189024798435142,4.692473245812175,11.372844079650065],
    [-14.216792998269533,4.687236119188637,11.38001344669554]]])

outputs = model(A)
print(outputs)

'14.244528055414136,4.681981192639867,11.387012928936239'






