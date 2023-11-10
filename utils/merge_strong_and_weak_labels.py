import os 
import argparse
import pandas as pd

def merge_strong_and_weak(strong_labels, weak_labels, save_path): 
    """
    Merge the strong and weak labels into a single dataframe. 
    """
    # Extract filename from 'filepath_wav_20' in weak_labels
    weak_labels['filename'] = weak_labels['filepath_wav_20'].apply(lambda x: x.split('/')[-1])

    # Extract filename from 'filename' in strong_labels
    strong_labels['filename_extracted'] = strong_labels['filename'].apply(lambda x: x.split('/')[-1])

    # Filter strong_labels to only include rows where the extracted filename is in weak_labels
    strong_and_weak = strong_labels[strong_labels['filename_extracted'].isin(weak_labels['filename'])].reset_index(drop=True)
    print(f'Number of data both in strong and weak labels: {len(strong_and_weak)}')

    # Merge strong_and_weak with weak_labels on 'filename'
    strong_and_weak = strong_and_weak.merge(weak_labels[['filename', 'vehicle_class', 'speed']], left_on='filename_extracted', right_on='filename', how='left')

    # Drop the columns that is no longer needed
    strong_and_weak = strong_and_weak.drop(columns=['filename_extracted', 'filename_y', 'event_label'])
    # Rename the column 'filename_x' to 'filename'
    strong_and_weak = strong_and_weak.rename(columns={'filename_x': 'filename'})

    # Save the merged dataframe to a csv file
    strong_and_weak.to_csv(save_path, sep='\t', index=False)


if __name__ == "__main__": 

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", type=str, default="/media/huimin/Backup_03TB_651/single_vehicle-20sec_gap15sec")
    args = parser.parse_args()

    strong_path = os.path.join(args.basedir, 'strong_labels.csv')
    weak_path = os.path.join(args.basedir, 'weak_labels.csv')

    strong_labels = pd.read_csv(strong_path, sep='\t')
    weak_labels = pd.read_csv(weak_path, sep='\t')

    save_path = os.path.join(args.basedir, 'strong_and_weak_labels.csv')
    merge_strong_and_weak(strong_labels, weak_labels, save_path)