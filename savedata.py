import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch  # Library for implementing Deep Neural Network
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import joblib



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
n_steps = 240*4
print('loaded')

#scaler = MinMaxScaler(feature_range=(0,1))
#scaled_train = scaler.fit_transform(df)
#print(scaler)
#joblib.dump(scaler, 'std_scaler.bin', compress=True)
# Преобразование датасета
def create_sequencesx(data, n_steps,string):
    X, y = [], []
    l = 0
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps][[string]].values)
        y.append(data[i + 1:i + n_steps + 1][[string]].values)
        # iloc
        # print(i, '/', len(data) - n_steps)
        if (i + n_steps + 2) % (240*4) == 0:
            i = i + n_steps + 1
            l += 1
            print('step', l)
        # print(i)
    return np.array(X), np.array(y)


Xx, yx = create_sequencesx(df, n_steps,'pos_x')
Xy, yy = create_sequencesx(df, n_steps,'pos_y')
Xz, yz = create_sequencesx(df, n_steps,'pos_z')




print("X shape:", Xx.shape)  # Форма входных данных
print("y shape:", yx.shape)  # Форма целевых данных


# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(Xx, yx, test_size=0.2, random_state=42)
print("Размер обучающего набора X:", X_train.shape)
print("Размер тестового набора X:", X_test.shape)
print("Размер обучающего набора y:", y_train.shape)
print("Размер тестового набора y:", y_test.shape)
X2_train, X2_test, y2_train, y2_test = train_test_split(Xy, yy, test_size=0.2, random_state=42)
X3_train, X3_test, y3_train, y3_test = train_test_split(Xz, yz, test_size=0.2, random_state=42)


def var2str(variable):

    return [global_var for global_var in globals() if id(variable) == id(globals()[global_var])]

def save(data):
    #f = open(f'{var2str(data)}.txt', 'w')
    np.save(f'{var2str(data)}.npy', data)
    #f.write(str(data))
    #f.close()


save(X_train)
save(X2_train)
save(X3_train)
save(X_test)
save(X2_test)
save(X3_test)
save(y_train)
save(y2_train)
save(y3_train)
save(y_test)
save(y2_test)
save(y3_test)


save(Xy)
save(Xz)
save(yx)
save(yy)
save(yz)
save(Xx)
save(Xy)
save(Xz)
save(yx)
save(yy)
save(yz)