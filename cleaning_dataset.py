import os
import cv2
import shutil
import numpy as np

DATASET_DIRRECTORY = "dataset_sorted"
TRASH_DIRRECTORY = "dataset_trash" 

os.makedirs(TRASH_DIRRECTORY, exist_ok=True)

def is_good_brain_slice(img_path):

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return False
    
    h, w = img.shape
    
    # 1. Бинаризация (делаем мозг белым, фон черным)
    _, thresh = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY)
    
    # 2. Находим контуры (очертания головы)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return False
    
    # Берем самый большой контур (саму голову)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest_contour)
    
    area_ratio = (bw * bh) / (w * h)
    if area_ratio < 0.3 or area_ratio > 0.85:
        return False
        
    aspect_ratio = bw / bh
    if aspect_ratio < 0.65 or aspect_ratio > 1.5:
        return False
        
   
    center_x, center_y = w // 2, h // 2
    contour_center_x = x + bw // 2
    contour_center_y = y + bh // 2

    if abs(center_x - contour_center_x) > (w * 0.2) or abs(center_y - contour_center_y) > (h * 0.2):
        return False

    return True

def main():
    total_kept = 0
    total_trashed = 0
    
    for label in ['0', '1']:
        class_dir = os.path.join(DATASET_DIRRECTORY, label)
        if not os.path.exists(class_dir):
            continue
            
        trash_class_dir = os.path.join(TRASH_DIRRECTORY, label)
        os.makedirs(trash_class_dir, exist_ok=True)
        
        print(f"Чистим папку {label}...")
        
        for f in os.listdir(class_dir):
            if not f.endswith('.png'):
                continue
                
            img_path = os.path.join(class_dir, f)
            
            if is_good_brain_slice(img_path):
                total_kept += 1
            else:
                # Перемещаем в мусорку (чтобы ты могла посмотреть, что удалилось)
                shutil.move(img_path, os.path.join(trash_class_dir, f))
                total_trashed += 1
                
    print(f"\nГотово! Оставлено хороших срезов: {total_kept}")
    print(f"Удалено мусора (глаза, шея, пустоты): {total_trashed}")
    print(f"Мусор перемещен в папку '{TRASH_DIRRECTORY}' (можешь её потом удалить).")

if __name__ == "__main__":
    main()