import zipfile
import io
import pydicom
import pandas as pd
import os

ZIP = r"C:\Users\Алина\Downloads\SCLEROSIS21_172.zip"
path_for_index = "index.parquet"

def is_valid_dicom(dataset):
    return hasattr(dataset, "SOPInstanceUID")

def extract_metadata(dataset, filepath):
    return {
        "zip_path": filepath,
        "patient_id": str(dataset.get("PatientID", "")),
        "series_uid": str(dataset.get("SeriesInstanceUID", "")),
        "series_desc": str(dataset.get("SeriesDescription", "")).lower().strip(),
        "param_tr": float(dataset.get("RepetitionTime", 0) or 0),
        "param_te": float(dataset.get("EchoTime", 0) or 0),
        "slice_num": int(dataset.get("InstanceNumber", 0) or 0),
    }

def scan_zip(zip_path):
    metadata_list = []
    
    if not os.path.exists(zip_path):
        print(f"Ошибка: архив не найден: {zip_path}")
        return None

    with zipfile.ZipFile(zip_path, 'r') as archive:
        all_files = archive.namelist()
        print(f"Файлов внутри архива {len(all_files)}")
        
        for entry in all_files:
            # Пропускаем директории
            if entry.endswith('/'):
                continue
            
            try:
                # Открываем файл внутри архива в поток
                with archive.open(entry) as f:
                    dicom_data = pydicom.dcmread(
                        io.BytesIO(f.read()), 
                        stop_before_pixels=True, 
                        force=True
                    )
                
                if is_valid_dicom(dicom_data):
                    info = extract_metadata(dicom_data, entry)
                    metadata_list.append(info)
                    
            except Exception:
                continue
                
    return metadata_list

def main():
    collected_data = scan_zip(ZIP)
    
    if collected_data:
        index_df = pd.DataFrame(collected_data)
        
        index_df.to_parquet(path_for_index)
        
        print(f"\nГотово, проиндексировано файлов: {len(index_df)}")
        print(f"Данные сохранены в {path_for_index}")
        
        # Выводим статистику по описаниям
        print("\nСводка")
        description_counts = index_df['series_desc'].value_counts().head(10)
        print(description_counts)
    else:
        print("Ошибка: не удалось собрать данные")

if __name__ == "__main__":
    main()