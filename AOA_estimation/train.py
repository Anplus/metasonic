import os
import struct
import mmap
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class ReadData:
    """
    A class to read binary data, labels, and offsets from preprocessed dataset files.
    """

    def __init__(self, prefix_path):
        """
        Initialize the data reader, load offsets, and labels from the header and label files.

        Args:
        prefix_path (str): The prefix path to `.header`, `.data`, and `.label` files.
        """
        self.offset_dict = {}
        # Read offsets from the header file
        for line in open(prefix_path + '.header', 'rb'):
            key, val_pos, val_len = line.split('\t'.encode('ascii'))
            self.offset_dict[key] = (int(val_pos), int(val_len))

        # Memory-map the data file for fast random access
        self.fp = open(prefix_path + '.data', 'rb')
        self.m = mmap.mmap(self.fp.fileno(), 0, access=mmap.ACCESS_READ)

        # Load labels from the label file
        print('Loading labels...')
        self.label = {}
        for line in open(prefix_path + '.label', 'rb'):
            key, label = line.split(b'\t')
            label = label.decode().replace('\n', '').split(',')
            self.label[key] = float(label[0])
        print('Finished loading data:', len(self.label))

    def get_data(self, key):
        """
        Retrieve binary data corresponding to a key.

        Args:
        key (bytes): The unique identifier for the data.

        Returns:
        bytes: Binary data corresponding to the key.
        """
        p = self.offset_dict.get(key, None)
        if p is None:
            return None
        val_pos, val_len = p
        return self.m[val_pos:val_pos + val_len]

    def get_label(self, key):
        """
        Retrieve the label corresponding to a key.

        Args:
        key (bytes): The unique identifier for the data.

        Returns:
        float: The label value.
        """
        return self.label.get(key)

    def get_keys(self):
        """
        Get all keys from the label file.

        Returns:
        list: A list of keys.
        """
        return self.label.keys()


class AudioDataset(Dataset):
    """
    PyTorch Dataset for loading audio data and corresponding labels.
    """

    def __init__(self, data_path):
        """
        Initialize the dataset by loading data and labels.

        Args:
        data_path (str): Path to the dataset file (prefix of `.data`, `.header`, `.label` files).
        """
        self.readData = ReadData(data_path)
        self.keys = list(self.readData.get_keys())

    def __len__(self):
        """
        Get the total number of data points in the dataset.

        Returns:
        int: Total number of data points.
        """
        return len(self.keys)

    def __getitem__(self, idx):
        """
        Get the data and label for a given index.

        Args:
        idx (int): Index of the data point.

        Returns:
        tuple: Processed data and corresponding label.
        """
        key = self.keys[idx]
        data = self.readData.get_data(key)
        data = list(struct.unpack('%sd' % 526, data))
        data = np.array(data).reshape((1, 526)).astype(np.float32)
        label = self.readData.get_label(key)
        label = np.array(label).astype(np.float32).reshape(1)
        return data, label


class CNN(nn.Module):
    """
    A 1D Convolutional Neural Network for regression tasks on audio data.
    """

    def __init__(self, n_channels1=32, n_channels2=64, n_channels3=64, n_channels4=128, kernel_size1=21, kernel_size2=5, kernel_size3=7, kernel_size4=5):
        """
        Initialize the CNN layers.
        """
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, n_channels1, kernel_size=kernel_size1, padding=kernel_size1 // 2)
        self.conv2 = nn.Conv1d(n_channels1, n_channels2, kernel_size=kernel_size2, padding=kernel_size2 // 2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(n_channels2, n_channels3, kernel_size=kernel_size3, padding=kernel_size3 // 2)
        self.conv4 = nn.Conv1d(n_channels3, n_channels4, kernel_size=kernel_size4, padding=kernel_size4 // 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(n_channels4 * 131, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Define the forward pass of the CNN.
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        # Map the output range from [0, 1] to [-50, 57]
        x = x * (57 + 50) - 50
        return x


def train_and_test_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=1000, test_interval=1, save_path='cnn/models/model_21'):
    """
    Train and evaluate the CNN model.

    Args:
    model (nn.Module): The CNN model.
    train_loader (DataLoader): DataLoader for training data.
    test_loader (DataLoader): DataLoader for test data.
    criterion: Loss function.
    optimizer: Optimizer.
    device: Device to run the model on ('cuda' or 'cpu').
    num_epochs (int): Number of epochs for training.
    test_interval (int): Interval for evaluating the model on test data.
    save_path (str): Path to save the best model.
    """
    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Training loop
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        # Test loop
        if (epoch + 1) % test_interval == 0:
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
            test_loss /= len(test_loader)
            print(f'Test [{epoch+1}/{num_epochs}], Loss: {test_loss:.4f}')

            # Save the best model
            if test_loss < best_loss:
                best_loss = test_loss
                best_epoch = epoch + 1
                torch.save(model.state_dict(), save_path + f'best_model_epoch_{best_epoch}.pth')
                print(f'Saved best model at epoch {best_epoch} with loss {best_loss:.4f}')

    print(f'Best model saved at epoch {best_epoch} with loss {best_loss:.4f}')


if __name__ == '__main__':
    # Configuration
    batch_size = 40
    num_epochs = 1000
    learning_rate = 0.8e-3
    test_interval = 5
    save_path = 'model/model_1/'

    # Ensure the save directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(f"Folder '{save_path}' has been checked and created if it did not exist.")

    # Load datasets
    train_dataset = AudioDataset('train_four_anchors_AOA')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = AudioDataset('test_four_anchors_AOA')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train and evaluate the model
    train_and_test_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs, test_interval, save_path)
