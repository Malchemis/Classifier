import argparse
import glob
import os 
import pandas as pd
import torchaudio

def remove_NaN_filename(csv_file):
    """
    Remove the row where the filename is NaN
    """
    csv_file  = csv_file.dropna(subset=['filepath_wav_30'])
    return csv_file

def remove_0_vehicle_class(csv_file):
    """
    Remove the row where the vehicle_class is 0
    """
    csv_file = csv_file[csv_file.vehicle_class != 0]
    return csv_file

def delete_files_from_folders(csv_file, basedir): 
    """
    Delete the wav files that are not in the csv file from the folders
    """
    # Get all wav files in different directories
    path_wav = os.path.join(basedir, "wav")
    wav_files = glob.glob(path_wav + "/**/*.wav", recursive=True)
    # Set the wav file name in the wav_files as the same as the csv_file
    wav_files = [os.path.join("wav", x.split("/")[-2], x.split("/")[-1]) for x in wav_files]

    # Get the wav file names that are not in the csv_file
    wav_files_to_delete = [x for x in wav_files if x not in csv_file["filepath_wav_30"].tolist()]
    # Get the wav file that should not be deleted
    wav_files_to_keep = [x for x in wav_files if x not in wav_files_to_delete]
    # Get the wav file names that are not in the wav_files
    csv_rows_to_delete = csv_file[~csv_file["filepath_wav_30"].isin(wav_files_to_keep)]
    # Get the csv row that should not be deleted
    csv_rows_to_keep = csv_file[~csv_file["filepath_wav_30"].isin(csv_rows_to_delete["filepath_wav_30"].tolist())]
    
    # Write the csv file
    csv_rows_to_keep.to_csv(path_csv, sep="\t", index=False)
    # Delete the wav files from the folders
    print(f'Found {len(wav_files_to_delete)} wav files to delete')
    for wav_file in wav_files_to_delete:
        print(f'Deleting {wav_file}')
        os.remove(os.path.join(args.basedir, wav_file))

def delete_empty_files(path_csv, basedir): 
    empty_files = []
    csv_file = pd.read_csv(path_csv, sep='\t')  
    for filename in csv_file['filepath_wav_30'].unique(): 
        audio, _ = torchaudio.load(os.path.join(basedir, filename))
        # Get the files that are empty
        if audio.shape[1] == 0:
            empty_files.append(filename)
    if len(empty_files) != 0:
        print(f'Found {len(empty_files)} empty files')
        # Delete the empty files from the csv file
        new_csv = csv_file[~csv_file['filepath_wav_30'].isin(empty_files)]
        # Rewrite the csv
        new_csv.to_csv(path_csv, sep='\t', index=False)
        # Delete the empty files from the folders
        for filename in empty_files: 
            print(f'Deleting {filename}')
            os.remove(os.path.join(basedir, filename))

def delete_short_files(path_csv, basedir, min_duration=25): 
    short_files = []
    csv_file = pd.read_csv(path_csv, sep='\t')  
    for filename in csv_file['filepath_wav_30'].unique(): 
        audio, sr = torchaudio.load(os.path.join(basedir, filename))
        # Get the files that are shorter than min_duration
        if audio.shape[1] < min_duration*sr:
            short_files.append(filename)
    if len(short_files) != 0:
        print(f'Found {len(short_files)} short files')
        # Delete the short files from the csv file
        new_csv = csv_file[~csv_file['filepath_wav_30'].isin(short_files)]
        # Rewrite the csv
        new_csv.to_csv(path_csv, sep='\t', index=False)
        # Delete the short files from the folders
        for filename in short_files: 
            print(f'Deleting {filename}')
            os.remove(os.path.join(basedir, filename))


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", type=str, default="/media/huimin/PortableSSD/KTH/data/SED-single_vehicle-30_sec")
    args = parser.parse_args()

    # Get csv file 
    path_csv = os.path.join(args.basedir, "weak_labels.csv")
    csv_file = pd.read_csv(path_csv, sep="\t")

    csv_file  = remove_NaN_filename(csv_file)
    csv_file = remove_0_vehicle_class(csv_file)

    delete_files_from_folders(csv_file, args.basedir)
    delete_empty_files(path_csv, args.basedir)
    delete_short_files(path_csv, args.basedir)