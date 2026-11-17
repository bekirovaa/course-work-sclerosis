import pandas as pd
import os
file_with_indexes= "C:\COURSEWORK\labels.xlsx"
path_for_index = "index.parquet"
labels_df = pd.read_excel(file_with_indexes)
labels_df = labels_df.iloc[:, [0,1]]
labels_df.columns = ['study_uid', 'label']
mri_index = pd.read_parquet(path_for_index)

def take_case_uid(path_in_zip_file):
    parts=path_in_zip_file.split('/')
    for part in parts:
        if part.startswith('1.2'):
            return part
    return None

mri_index['study_uid'] = mri_index['zip_path'].apply(take_case_uid)
merged_df = pd.merge(mri_index, labels_df, on='study_uid', how='inner')
print(f"Всего снимков: {len(mri_index)}")
print(f"Снимков с известным диагнозом: {len(merged_df)}")
merged_df.to_parquet("dataset_with_labels.parquet")
