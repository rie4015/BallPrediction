from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import os
import pandas as pd
from torchvision.io import read_image




class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return len(self.img_labels)


    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label



transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

dataset = CustomImageDataset('simulation_dataset.csv','./data/', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
print(dataloader)


last_date = test_data.index[-1]

# Generate the next 30 dates
future_dates = pd.date_range(start=last_date + 1, periods=30)

# Concatenate the original index with the future dates
combined_index = test_data.index.append(future_dates)

# set the size of the plot
plt.rcParams['figure.figsize'] = [14, 4]

# Test data
plt.plot(test_data.index[-100:-30], test_data.index[-100:-30], label="test_data", color="b")
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