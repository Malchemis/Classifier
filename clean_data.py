import argparse
import glob
import os 
import pandas as pd


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", type=str, default="/media/huimin/PortableSSD/KTH/data/SED-single_vehicle-30_sec")
    args = parser.parse_args()

    # Get csv file 
    path_csv = os.path.join(args.basedir, "weak_labels.csv")
    csv_file = pd.read_csv(path_csv, sep="\t")

    # Get all wav files in different directories
    path_wav = os.path.join(args.basedir, "wav")
    wav_files = glob.glob(path_wav + "/**/*.wav", recursive=True)

    # Drop row where there is no filepath name
    csv_file  = csv_file.dropna(subset=['filepath_wav_30'])
    # Drop row where vehicle_class is 0 
    csv_file = csv_file[csv_file.vehicle_class != 0]

    # Set the wav file name in the wav_files as the same as the csv_file
    wav_files = [os.path.join("wav", x.split("/")[-2], x.split("/")[-1]) for x in wav_files]

    # Get the wav file name that is not in the csv_file
    wav_files_to_delete = [x for x in wav_files if x not in csv_file["filepath_wav_30"].tolist()]
    # Get the wav file that should not be deleted
    wav_files_to_keep = [x for x in wav_files if x not in wav_files_to_delete]
    # Get the wav file name that is not in the wav_files
    csv_rows_to_delete = csv_file[~csv_file["filepath_wav_30"].isin(wav_files_to_keep)]
    # Get the csv row that should not be deleted
    csv_rows_to_keep = csv_file[~csv_file["filepath_wav_30"].isin(csv_rows_to_delete["filepath_wav_30"].tolist())]
    
    # Write the csv file
    csv_rows_to_keep.to_csv(path_csv, sep="\t", index=False)
    # Delete the wav files to delete
    print(f'Found {len(wav_files_to_delete)} wav files to delete')
    for wav_file in wav_files_to_delete:
        print(f'Deleting {wav_file}')
        os.remove(os.path.join(args.basedir, wav_file))