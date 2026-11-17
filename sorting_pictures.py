import zipfile, io, os, pydicom, pandas as pd, numpy as np
from PIL import Image

ZIP_PATH = r"C:\Users\Алина\Downloads\SCLEROSIS21_172.zip"
LABELED_INDEX = "dataset_with_labels.parquet"
OUTPUT_DIRECTORY = "dataset_sorted"
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

def extract_pixels(ds):
    img = ds.pixel_array.astype(float)
    if img.ndim > 2:
        if img.shape[-1] == 3: 
            img = img[:, :, 0]
        else:                 
            img = img[0]
    
    min_pixel_clr, max_pixel_clr = np.min(img), np.max(img)
    if max_pixel_clr - min_pixel_clr < 1e-5: 
        return None
    # Нормализация яркости от 0 до 255
    img = (img - min_pixel_clr) / (max_pixel_clr - min_pixel_clr) * 255
    return img.astype(np.uint8)

def score_axial_slice(img):
    height, width = img.shape
    brain_pixels = np.sum(img > 40)
    if brain_pixels < (height * width * 0.3): return 0
    cy, cx = height // 2, width // 2
    ch, cw = int(height * 0.15), int(width * 0.15)
    center = img[cy-ch:cy+ch, cx-cw:cx+cw]
    if center.size == 0: return 0
    contrast = np.std(center)
    return contrast

def is_sagittal(img):
    h, w = img.shape
    y_indices, x_indices = np.where(img > 40)
    if len(y_indices) == 0: return False
    
    height = np.max(y_indices) - np.min(y_indices)
    width = np.max(x_indices) - np.min(x_indices)
    
    # Если высота сильно больше ширины - это сагитальный(профиль)
    ratio = max(height, width) / (min(height, width) + 1e-5)
    return ratio > 1.3


def process_series(archive, zip_paths, class_dir, study_uid):
    slices_data = []
    # Читаем все срезы серии
    for file_name in zip_paths:
        try:
            with archive.open(file_name) as f:
                ds = pydicom.dcmread(io.BytesIO(f.read()), force=True)
                if not hasattr(ds, 'pixel_array'): continue
                img = extract_pixels(ds)
                if img is None: continue
                slices_data.append({'img': img, 'inst': ds.InstanceNumber})
        except Exception:
            continue

    if not slices_data: return 0

    saved = 0
    axial_scores = [(i, score_axial_slice(s['img'])) for i, s in enumerate(slices_data)]
    axial_scores.sort(key=lambda x: x[1], reverse=True) # Сортируем по убыванию контраста
    for i in range(min(3, len(axial_scores))):
        idx = axial_scores[i][0]
        if axial_scores[i][1] > 10: # Отсекаем срезы с плохим контрастом
            img = slices_data[idx]['img']
            filename = f"study_{study_uid[-6:]}_axial_{slices_data[idx]['inst']}.png"
            Image.fromarray(img).save(os.path.join(class_dir, filename))
            saved += 1

    #Ищем сагиттальные срезы
    sagittal_found = 0
    for s in slices_data:
        if is_sagittal(s['img']):
            filename = f"study_{study_uid[-6:]}_sagittal_{s['inst']}.png"
            Image.fromarray(s['img']).save(os.path.join(class_dir, filename))
            saved += 1
            sagittal_found += 1
            if sagittal_found >= 2: # Берем только 2 лучших сагиттальных среза
                break

    return saved

def main():
    print("Загружаем данные...")
    df = pd.read_parquet(LABELED_INDEX)
    grouped = df.groupby(['study_uid', 'series_uid', 'label'])
    
    total_saved = 0
    with zipfile.ZipFile(ZIP_PATH, 'r') as archive:
        for (study, series, label), group in grouped:
            class_dir = os.path.join(OUTPUT_DIRECTORY, str(label))
            os.makedirs(class_dir, exist_ok=True)
            
            zip_paths = group.sort_values('slice_num')['zip_path'].tolist()
            saved = process_series(archive, zip_paths, class_dir, study)
            total_saved += saved
            
            if total_saved % 50 == 0 and total_saved > 0:
                print(f"Обработано срезов: {total_saved}...")

    print(f"\nГотово! Сохранено {total_saved} срезов в папку '{OUTPUT_DIRECTORY}'")

if __name__ == "__main__":
    main()