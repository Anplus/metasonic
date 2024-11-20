import os
import struct
import uuid
import librosa
import numpy as np
from tqdm import tqdm
from scipy.fftpack import fft
from scipy.signal import chirp, correlate
import argparse

class DataSetWriter:
    """
    A utility class for writing data, headers, and labels to binary files for dataset storage.
    """
    def __init__(self, prefix):
        """
        Initialize file handlers for data, header, and label files.
        
        Args:
        prefix (str): The prefix for output files.
        """
        self.data_file = open(prefix + '.data', 'wb')  # Binary file for storing audio features
        self.header_file = open(prefix + '.header', 'wb')  # File for storing headers with metadata
        self.label_file = open(prefix + '.label', 'wb')  # File for storing labels
        self.offset = 0  # Offset tracker for data positions
        self.header = ''  # Header content

    def add_data(self, key, data):
        """
        Add processed data to the data file and record metadata in the header file.

        Args:
        key (str): Unique identifier for the data.
        data (bytes): Processed audio data in binary format.
        """
        # Write the key and its length to the data file
        self.data_file.write(struct.pack('I', len(key)))
        self.data_file.write(key.encode('ascii'))
        # Write the data and its length to the data file
        self.data_file.write(struct.pack('I', len(data)))
        self.data_file.write(data)
        
        # Update offset and write metadata to the header file
        self.offset += 4 + len(key) + 4
        self.header = key + '\t' + str(self.offset) + '\t' + str(len(data)) + '\n'
        self.header_file.write(self.header.encode('ascii'))
        self.offset += len(data)

    def add_label(self, label):
        """
        Add label information to the label file.

        Args:
        label (str): Label information corresponding to the data.
        """
        self.label_file.write(label.encode('ascii') + '\n'.encode('ascii'))


def preprocess_audio(file_pre, audio_file, target_length=4800, sr=48000):
    """
    Preprocess an audio file by extracting a target spectrum.

    Args:
    file_pre (str): Prefix path for the audio files.
    audio_file (str): Path to the audio file to process.
    target_length (int): The desired length of the processed audio segment.
    sr (int): The sampling rate to resample the audio.

    Returns:
    np.ndarray: Target spectrum extracted from the specified frequency range.
    """
    offset = 1500  # Offset to account for padding around the matched chirp
    full_path = os.path.join(file_pre, audio_file)

    # Load the audio file with the specified sampling rate
    wav, _ = librosa.load(full_path, sr=sr)
    
    # Generate a linear chirp signal for cross-correlation
    t = np.linspace(0, 0.1, int(sr * 0.1))
    chirp_signal = chirp(t, f0=16000, f1=20000, t1=0.1, method='linear')
    
    # Perform cross-correlation to detect the chirp in the audio
    correlation = correlate(wav, chirp_signal, mode='full')
    peak_index = np.argmax(np.abs(correlation))
    
    # Define the start and end indices for the matched segment
    start_index = peak_index - len(chirp_signal) // 2 - offset
    end_index = start_index + len(chirp_signal) + offset
    
    # Extract the matched segment
    matched_segment = wav[start_index:end_index]

    # Pad or trim the segment to the target length
    if len(matched_segment) < target_length + offset:
        matched_segment = np.pad(matched_segment, (0, target_length + offset - len(matched_segment)), 'constant')
    else:
        matched_segment = matched_segment[:target_length + offset]

    # Perform FFT to transform the segment into the frequency domain
    freq_domain = fft(matched_segment)
    freq = np.fft.fftfreq(len(matched_segment), d=1/sr)
    amplitude_spectrum = np.abs(freq_domain)
    
    # Extract the amplitude spectrum in the target frequency range (18 kHz to 20 kHz)
    idx = np.where((freq >= 16000) & (freq <= 20000))
    target_spectrum = amplitude_spectrum[idx]
    return target_spectrum


def convert_data(file_pre, data_list_path, output_prefix):
    """
    Convert a dataset of audio files into a structured format with data, headers, and labels.

    Args:
    file_pre (str): Prefix path for the audio files.
    data_list_path (str): Path to the text file containing audio file paths and labels.
    output_prefix (str): Prefix for the output files.
    """
    data_list = open(data_list_path, "r").readlines()
    writer = DataSetWriter(output_prefix)  # Initialize the DataSetWriter
    
    # Process each audio file in the dataset
    for record in tqdm(data_list):
        path, label = record.strip().split('\t')
        amplitude_spectrum = preprocess_audio(file_pre, path)
        
        # Skip if the spectrum is empty
        if len(amplitude_spectrum) == 0:
            continue
        
        # Pack the spectrum data and add it to the dataset
        data = struct.pack('%sd' % len(amplitude_spectrum), *amplitude_spectrum)
        key = str(uuid.uuid1())  # Generate a unique identifier for the data
        writer.add_data(key, data)
        writer.add_label('\t'.join([key, label]))


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Convert audio data into binary dataset format.")
    parser.add_argument('--file_pre', type=str, required=True, help="Prefix path for audio files.")
    parser.add_argument('--train_list', type=str, required=True, help="Path to the training data list file.")
    parser.add_argument('--val_list', type=str, required=True, help="Path to the validation data list file.")
    parser.add_argument('--output_prefix', type=str, required=True, help="Output prefix for the dataset files.")
    args = parser.parse_args()

    # Convert training data
    convert_data(args.file_pre, args.train_list, args.output_prefix + '_train')
    # Convert validation data
    convert_data(args.file_pre, args.val_list, args.output_prefix + '_val')