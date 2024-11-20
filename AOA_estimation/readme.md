# Audio Signal Processing and Prediction Pipeline

This repository provides a pipeline to preprocess audio signals, train a model, and make predictions on the data. Follow the steps below to get started.

---

## Step 1: Prepare the Dataset

You need to prepare the following dataset files. Their format is as shown in the examples:
- train_list.txt: Contains the paths to the training audio files and their corresponding labels.
- val_list.txt: Contains the paths to the validation audio files and their corresponding labels.
- test_list.txt: Contains the paths to the testing audio files and their corresponding labels.

## Step 2: Preprocess the Audio Signals

Run the convert_data.py script to preprocess the audio signals and convert them into a binary dataset format.

### Command:
```bash
python convert_data.py --file_pre "path_to_audio_files" --train_list "train_list.txt" --val_list "val_list.txt" --output_prefix "output_dataset"
--file_pre: The prefix path to the audio files.
--train_list: Path to the training data list file.
--val_list: Path to the validation data list file.
--output_prefix: Prefix for the generated dataset files (e.g., train_dataset, val_dataset).
```

## Step 3: Train the data
```bash
python train.py --train_data "output_dataset_train.data" --val_data "output_dataset_val.data" --output_model "trained_model.pth"
```
## Step 4: Predict the results
```bash
python prediction.py --file_pre "path_to_audio_files"
```
