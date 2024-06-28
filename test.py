from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F


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
n_steps = 3

# Преобразование датасета
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps][['pos_x', 'pos_y', 'pos_z']].values)
        y.append(data.iloc[i + n_steps][['pos_x', 'pos_y', 'pos_z']].values)
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
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Пример итерации по обучающему DataLoader





class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.input_layer = nn.Linear(input_size * input_size, hidden_size)
        self.relu = nn.ReLU()
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(-1,3 * 3)
        x = self.input_layer(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x





# Определение параметров модели
input_size = 3  # Размер входных данных (pos_x, pos_y, pos_z)
hidden_size = 64 # Размер скрытого слоя
output_size = 3  # Размер выходных данных (предсказание следующих координат)

# Инициализация модели
model = NeuralNetwork(input_size, hidden_size, output_size).to(device)
print(model)


learning_rate = 1e-6
momentum =0.5

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


