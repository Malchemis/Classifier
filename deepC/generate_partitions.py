import argparse
import yaml
import os 
import pickle

import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, Resample

def verifiy_data(labels, config): 
    # Compare the number of unique filenames and raw number of rows to see if there are any duplicates.
    unique = labels[config['data']['header'][0]].nunique()
    length = labels.shape[0]
    if not np.array_equal(unique, length):
        print("There are duplicates in the dataset.")
    else:
        print(f"There are {length} unique filenames in the dataset.")

    # Verify that the classes are the correct ones
    unique_classes = labels[config['data']['header'][1]].unique()
    unique_classes.sort()
    if not np.array_equal(config['data']['classes'], unique_classes):
        print('The classes are not the same as the ones in the annotations file.')
        print('Classes in annotations file: {}'.format(unique_classes))
        print('Classes in classes variable: {}'.format(config['data']['classes']))
        raise ValueError('The classes are not the same as the ones in the annotations file.')
    
def generate_split(labels, config, split_size= [0.7, 0.1, 0.2], generate_test=True):
    """
    Generate the train, validation and test splits.
    The train split will be used to train the model, the validation split 
    will be used to evaluate the model during training and the test split will be used
    to evaluate the model after training.
    
    Args:
        labels (pandas dataframe): The dataframe containing the annotations.
        config (dict): The configuration dictionary.
        split_size (list): The size of the splits [train, val, test]. The default is [0.7, 0.1, 0.2].
    """

    if generate_test: 
        # Split the dataset into train and test
        train, test = train_test_split(labels, test_size=split_size[2], random_state=config['training']['seed'], stratify=labels[config['data']['header'][1]])
        # Split the train dataset into train and validation
        train, val = train_test_split(train, test_size=0.1, random_state=config['training']['seed'], stratify=train[config['data']['header'][1]])
    else: 
        train, val = train_test_split(labels, test_size=split_size[1], random_state=config['training']['seed'], stratify=labels[config['data']['header'][1]])
        test = pd.DataFrame()

    train.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    # Verify the splits
    length = labels.shape[0]
    print(f"Number of unique audio files: {length} = {len(train)} + {len(val)} + {len(test)}") if len(labels) == len(train) + len(val) + len(test) else print('The splits are not correct.')
    print('Number of train files: {}'.format(len(train)))
    print('Number of validation files: {}'.format(len(val)))
    print('Number of test files: {}'.format(len(test)))

    return train, val, test

def generate_partitions(train, val, test, config): 
    """
    Generate the partitions.
    The partitions will be together, identified by 3 keys (train, val, test) and the values will be the couple (filename, class).
    
    Args:
        train (pandas dataframe): The dataframe containing the train annotations.
        val (pandas dataframe): The dataframe containing the validation annotations.
        test (pandas dataframe): The dataframe containing the test annotations.
        config (dict): The configuration dictionary.
    """

    # Convert classes to be in range [0, num_classes - 1]
    # This is necessary for the cross-entropy loss.
    train[config['data']['header'][1]] = train[config['data']['header'][1]].apply(lambda x: config['data']['classes'].index(x))
    val[config['data']['header'][1]] = val[config['data']['header'][1]].apply(lambda x: config['data']['classes'].index(x))
    test[config['data']['header'][1]] = test[config['data']['header'][1]].apply(lambda x: config['data']['classes'].index(x))

    # Get the partitions
    if not os.path.exists(config['data']['partition']):
        partitions = {'train': [], 'val': [], 'test': []}

        partitions['train'] = [(filename, class_value) for filename, class_value in zip(train[config['data']['header'][0]], train[config['data']['header'][1]])]
        partitions['val'] = [(filename, class_value) for filename, class_value in zip(val[config['data']['header'][0]], val[config['data']['header'][1]])]
        partitions['test'] = [(filename, class_value) for filename, class_value in zip(test[config['data']['header'][0]], test[config['data']['header'][1]])]
        
        # Save the partitions
        with open(config['data']['partition'], 'wb') as f:
            pickle.dump(partitions, f)
    else:
        with open(config['data']['partition'], 'rb') as f:
            partitions = pickle.load(f)

    return partitions

def pad_audio(audio, target_len, fs):
    
    if audio.shape[-1] < target_len:
        audio = torch.nn.functional.pad(
            audio, (0, target_len - audio.shape[-1]), mode="constant"
        )

        padded_indx = [target_len / len(audio)]
        onset_s = 0.000
    
    elif len(audio) > target_len:
        
        rand_onset = random.randint(0, len(audio) - target_len)
        audio = audio[rand_onset:rand_onset + target_len]
        onset_s = round(rand_onset / fs, 3)
        padded_indx = [target_len / len(audio)] 

    else:
        onset_s = 0.000
        padded_indx = [1.0]

    offset_s = round(onset_s + (target_len / fs), 3)
    return audio, onset_s, offset_s, padded_indx

def get_log_melspectrogram(audio, sample_rate, n_fft, win_length, hop_length, n_mels, f_min, f_max):
    """Compute log melspectrogram of an audio signal."""
    # Compute the mel spectrogram
    mel_spectrogram = MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        window_fn=torch.hamming_window,
        wkwargs={"periodic": False},
        power=1,
    )(audio)
    # Convert to dB
    amp_to_db = AmplitudeToDB(stype="amplitude")
    amp_to_db.amin = 1e-5  # amin= 1e-5 as in librosa
    log_melspectrogram = amp_to_db(mel_spectrogram).clamp(min=-50, max=80)
    return log_melspectrogram

def get_log_melspectrogram_set(set, save_path, config): 
    """Compute log melspectrogram of a set of audio signals."""
    for i, (filename, _) in enumerate(set):
        print(f"\rConstructing mel audio {i+1}/{len(set)}", flush=True)
        audio, sr = torchaudio.load(os.path.join(config['data']['data_dir'], filename))
        # Convert to mono if necessary
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        # Resample the audio if necessary
        if sr != config['feats']['sample_rate']:
            resampled_audio = Resample(sr, config['feats']['sample_rate'])(audio)
        # Pad the audio if necessary
        resampled_audio_pad, *_ = pad_audio(
                                    resampled_audio, 
                                    config['feats']['duration']*config['feats']['sample_rate'], 
                                    config['feats']['sample_rate']
                                )
        # Compute the log melspectrogram
        log_melspectrogram = get_log_melspectrogram(
                                resampled_audio_pad, 
                                config['feats']['sample_rate'], 
                                config['feats']['n_fft'], 
                                config['feats']['win_length'],
                                config['feats']['hop_length'], 
                                config['feats']['n_mels'], 
                                config['feats']['f_min'], 
                                config['feats']['f_max']
                            )
        # Create the save path folder if doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(f"{save_path}/{filename.split('/')[-1].replace('.wav', '')}.npy", log_melspectrogram.numpy())

def compute_all_log_melspectrogram(partitions, config): 
    if not os.path.exists(os.path.join(config['data']['data_dir'], 'npy', 'train')):
        print("Constructing mel audio for train set")
        get_log_melspectrogram_set(partitions['train'], os.path.join(config['data']['data_dir'], 'npy', 'train'), config)
    if not os.path.exists(os.path.join(config['data']['data_dir'], 'npy', 'val')):    
        print("Constructing mel audio for val set")
        get_log_melspectrogram_set(partitions['val'], os.path.join(config['data']['data_dir'], 'npy', 'val'), config)
    if not os.path.exists(os.path.join(config['data']['data_dir'], 'npy', 'test')):
        print("Constructing mel audio for test set")
        get_log_melspectrogram_set(partitions['test'], os.path.join(config['data']['data_dir'], 'npy', 'test'), config)

def find_n_frames(partitions, config):
    """Find the number of frames in the dataset."""
    n_frames = []
    for i, (filename, _) in enumerate(partitions['train']):
        log_melspectrogram = np.load(os.path.join(config['data']['data_dir'], 'npy', 'train', filename.split('/')[-1].replace('.wav', '') + '.npy'))
        n_frames.append(log_melspectrogram.shape[2])
    return min(n_frames)

def add_n_frames_in_yaml(n_frames, config):
    """Add the number of frames in the dataset to the config file."""
    with open(config, 'r') as f:
        config_yaml = yaml.safe_load(f)
        config_yaml['training']['n_frames'] = n_frames
    with open(config, 'w') as f:
        yaml.dump(config_yaml, f, sort_keys=False)

if __name__ == "__main__": 

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='conf/conf.yaml', help='Path to config file.')
    args = parser.parse_args()

    # Open the config file 
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load the annotations in a pandas dataframe
    labels = pd.read_csv(config['data']['annotations_path'], sep='\t')

    # Verify the data
    verifiy_data(labels, config)

    # Generate the splits
    if config['data']['dataset'] == 'vehicle':
        train, val, test = generate_split(labels, config, generate_test=True)
    elif config['data']['dataset'] == 'IDMT':
        train, val, _ = generate_split(labels, config, split_size= [0.9, 0.1], generate_test=False) # split size determined according to the IDMT paper
        # Load the test csv file
        test = pd.read_csv(config['data']['test_annotations_path'], sep='\t')
    
    # Generate the partitions
    partitions = generate_partitions(train, val, test, config)

    # Compute the log melspectrogram for each set
    compute_all_log_melspectrogram(partitions, config)

    # Find the number of frames in the dataset and add it to the config file
    n_frames = find_n_frames(partitions, config)
    add_n_frames_in_yaml(n_frames, args.config)