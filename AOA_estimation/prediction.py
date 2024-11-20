import librosa
import numpy as np
import torch
import torch.nn as nn
from scipy.fftpack import fft
from scipy.signal import chirp, correlate
import argparse

class CNN(nn.Module):
    """
    A 1D Convolutional Neural Network for regression tasks on audio spectrum data.
    """
    def __init__(self, n_channels1=32, n_channels2=64, n_channels3=64, n_channels4=128, kernel_size1=21, kernel_size2=5, kernel_size3=7, kernel_size4=5):
        """
        Initialize the CNN model layers.
        """
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, n_channels1, kernel_size=kernel_size1, padding=kernel_size1//2)
        self.conv2 = nn.Conv1d(n_channels1, n_channels2, kernel_size=kernel_size2, padding=kernel_size2//2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(n_channels2, n_channels3, kernel_size=kernel_size3, padding=kernel_size3//2)
        self.conv4 = nn.Conv1d(n_channels3, n_channels4, kernel_size=kernel_size4, padding=kernel_size4//2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(n_channels4 * 131, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Define the forward pass of the CNN model.
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

        # Map the Sigmoid output range from [0, 1] to [-50, 57]
        x = x * (57 + 50) - 50
        return x


def preprocess_audio(file_pre, audio_file, target_length=4800, sr=48000):
    """
    Preprocess an audio file to extract the target spectrum.

    Args:
    file_pre (str): Prefix path for the audio files.
    audio_file (str): Path to the audio file.
    target_length (int): Desired length of the processed audio segment.
    sr (int): Sampling rate to load the audio.

    Returns:
    np.ndarray: Target spectrum in the specified frequency range.
    """
    # Set an offset for additional padding
    offset = 1500

    # Load the audio file using librosa
    full_path = file_pre + audio_file
    wav, _ = librosa.load(full_path, sr=sr)

    # Generate a chirp signal from 16kHz to 20kHz over 0.1 seconds
    t = np.linspace(0, 0.1, int(sr * 0.1))
    chirp_signal = chirp(t, f0=16000, f1=20000, t1=0.1, method='linear')

    # Perform cross-correlation to detect the chirp in the audio
    correlation = correlate(wav, chirp_signal, mode='full')

    # Find the peak index in the correlation result
    peak_index = np.argmax(np.abs(correlation))

    # Extract the most similar 0.1-second segment around the chirp
    start_index = peak_index - len(chirp_signal) // 2 - offset
    end_index = start_index + len(chirp_signal) + offset
    matched_segment = wav[start_index:end_index]

    # Ensure the segment length matches the target length
    if len(matched_segment) < target_length + offset:
        matched_segment = np.pad(matched_segment, (0, target_length + offset - len(matched_segment)), 'constant')
    else:
        matched_segment = matched_segment[:target_length + offset]

    # Convert the time-domain signal to the frequency domain
    freq_domain = fft(matched_segment)
    freq = np.fft.fftfreq(len(matched_segment), d=1/sr)

    # Extract the amplitude spectrum in the frequency range 16kHz to 20kHz
    amplitude_spectrum = np.abs(freq_domain)
    idx = np.where((freq >= 16000) & (freq <= 20000))
    target_spectrum = amplitude_spectrum[idx]
    return target_spectrum


def infer(model, file_pre, audio_path, device):
    """
    Perform inference on an audio file using the trained model.

    Args:
    model (nn.Module): Trained CNN model.
    file_pre (str): Prefix path for the audio files.
    audio_path (str): Path to the audio file for inference.
    device (torch.device): Device to run the model on ('cuda' or 'cpu').

    Returns:
    np.ndarray: Predicted output from the model.
    """
    data = preprocess_audio(file_pre, audio_path)
    data = np.array(data).reshape((1, 1, 526)).astype(np.float32)
    data = torch.tensor(data).to(device)
    model.eval()
    with torch.no_grad():
        output = model(data)
    return output.cpu().numpy().flatten()


def load_true_values(file_path):
    """
    Load true values from a file for evaluation.

    Args:
    file_path (str): Path to the file containing true values.

    Returns:
    tuple: Lists of numeric IDs, file names, and true values.
    """
    true_values = []
    file_name_list = []
    num_list = []

    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split('\t')
            file_name = parts[0].split('/')[-1]
            value = float(parts[-1])  # Assuming the true value is a single float
            true_values.append(value)
            file_name_list.append(file_name)

            # Extract numeric IDs from the file name
            if len(file_name) == 48:
                num = int(file_name[6])
            elif len(file_name) == 49:
                num = int(file_name[6:8])
            elif len(file_name) == 50:
                num = int(file_name[6:9])
            else:
                num = int(file_name[6:10])
            num_list.append(num)

    return num_list, file_name_list, true_values


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Inference using a trained CNN model.")
    parser.add_argument('--file_pre', type=str, required=True, help="Prefix path for audio files.")
    args = parser.parse_args()

    # Load the trained model
    model_path = 'model/model_sigmod/best_model_epoch_40.pth'  # Path to the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path))

    # Load true values for evaluation
    true_values_file = 'test_list.txt'
    num_list, file_name_list, true_values = load_true_values(true_values_file)

    # Perform inference and calculate errors
    errors = []
    with open("error.txt", "w") as file:
        file.write("True Pred Error\n")
        for i in range(len(true_values)):
            audio_file_path = f"data_small_room_same_points_compare/{num_list[i]}/{file_name_list[i]}"
            result = infer(model, args.file_pre, audio_file_path, device)
            error = np.abs(result - true_values[i])  # Absolute error
            errors.append(error)
            print(f"Predicted: {result}, True: {true_values[i]}, Error: {error}")
            file.write(f"{true_values[i]} {result[0]} {error}\n")

    # Output mean error and all errors
    print(f"Mean Error: {np.mean(errors)}")
    print(errors)
