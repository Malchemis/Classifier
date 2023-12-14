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
    # Rename the column 'vehicle_class' to 'event_label'
    strong_and_weak = strong_and_weak.rename(columns={'vehicle_class': 'event_label'})

    # # Convert the column 'event_label' to string
    # strong_and_weak['event_label'] = strong_and_weak['event_label'].apply(lambda x: 'class_' + str(x))
    
    # # Rename class 1 to motorcycle
    # strong_and_weak['event_label'] = strong_and_weak['event_label'].apply(lambda x: 'motorcycle' if x == 1 else x)
    # Rename class 2 to car
    strong_and_weak['event_label'] = strong_and_weak['event_label'].apply(lambda x: 'car' if x == 2 else x)
    # Rename classes 5 and 12 to lorry 
    strong_and_weak['event_label'] = strong_and_weak['event_label'].apply(lambda x: 'lorry' if x in [5, 12] else x)
    # Remove class 15 
    strong_and_weak = strong_and_weak[strong_and_weak['event_label'] != 15].reset_index(drop=True)
    # Remove class 1 
    strong_and_weak = strong_and_weak[strong_and_weak['event_label'] != 1].reset_index(drop=True)

    # Keep only the audios in which the onset is between 3.5 and 7.4 seconds
    strong_and_weak = strong_and_weak[(strong_and_weak['onset'] >= 3.5) & (strong_and_weak['onset'] <= 7.4)].reset_index(drop=True)
    # And the offset is less than 13 seconds
    strong_and_weak = strong_and_weak[strong_and_weak['offset'] <= 13].reset_index(drop=True)

    print(f'Number of data after filtering: {len(strong_and_weak)}')

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