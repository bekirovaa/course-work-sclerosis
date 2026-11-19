import os
import cv2
import numpy as np
import shutil

DATASET_DIR = "dataset_sorted"
OUTPUT_DIR = "dataset_for_resnet" 
TRASH_DIR = "dataset_mra_trash"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TRASH_DIR, exist_ok=True)


def is_axial(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
    if image is None:
        print("Картинка не найдена")
        return False
    _, bright_mask = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    bright_contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    sosud_found = False
    for cnt in bright_contours:
        area = cv2.contourArea(cnt)
        if area < 100: continue 
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = max(w, h) / float(min(w, h))
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 1.0 
        if aspect_ratio > 2.5 and solidity < 0.7:
            sosud_found = True
            break
        
    h_img, w_img = image.shape
    corners = [image[0:30, 0:30], image[0:30, w_img-30:w_img], 
               image[h_img-30:h_img, 0:30], image[h_img-30:h_img, w_img-30:w_img]]
    mean_corner = np.mean([np.mean(c) for c in corners])
    if sosud_found or mean_corner > 15:
        return False
    filtered_image = cv2.medianBlur(image, 7)
    _, thresh_img = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    if perimeter == 0 or area == 0:
        return False
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = w / float(h)
    is_circle = False
    if circularity > 0.7 and (0.8 <= aspect_ratio <= 1.2):
        is_circle = True

    if is_circle:
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        kernel = np.ones((15, 15), np.uint8)
        brain_mask = cv2.erode(mask, kernel, iterations=1)
        edges = cv2.Canny(image, 50, 150)
        brain_edges = cv2.bitwise_and(edges, edges, mask=brain_mask)
        edge_count = cv2.countNonZero(brain_edges)
        brain_area = cv2.countNonZero(brain_mask)
        if brain_area > 0:
            texture_density = edge_count / brain_area
            if texture_density < 0.04:
                is_circle = False
        else:
            is_circle = False
            
    return is_circle

def is_saggital(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
    if image is None:
        print("Картинка не найдена")
        return False
    _, bright_mask = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    bright_contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    sosud_found = False
    for cnt in bright_contours:
        area = cv2.contourArea(cnt)
        if area < 100: continue 
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = max(w, h) / float(min(w, h)) 
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 1.0 
        if aspect_ratio > 2.5 and solidity < 0.7:
            sosud_found = True
            break
        
    h_img, w_img = image.shape
    corners = [image[0:30, 0:30], image[0:30, w_img-30:w_img], 
               image[h_img-30:h_img, 0:30], image[h_img-30:h_img, w_img-30:w_img]]
    mean_corner = np.mean([np.mean(c) for c in corners])
    
    if sosud_found or mean_corner > 15:
        return False
    filtered_image = cv2.medianBlur(image, 7)
    _, thresh_img = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    if perimeter == 0 or area == 0:
        return False
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    if circularity > 0.75: # Если это почти идеальный круг, то это не профиль
        return False
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = w / float(h)
    hull = cv2.convexHull(largest_contour, returnPoints=False)
    if len(hull) < 4:
        return False
    defects = cv2.convexityDefects(largest_contour, hull)
    deep_defects = 0
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            depth = d / 256.0
          
            if depth > 15: 
                deep_defects += 1
                
    # У профиля должно быть минимум 2 глубоких впадины 
    if deep_defects < 2:
        return False


    cx = x + w // 2
    cy = y + int(h * 0.4)
    
    crop_size = int(min(w, h) * 0.25)
    y1, y2 = max(0, cy - crop_size), min(h_img, cy + crop_size)
    x1, x2 = max(0, cx - crop_size), min(w_img, cx + crop_size)
    center_crop = image[y1:y2, x1:x2]
    
    if center_crop.size == 0:
        return False
    sobel_y = cv2.Sobel(center_crop, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y_8u = np.uint8(255 * np.absolute(sobel_y) / (np.max(np.absolute(sobel_y)) + 1e-5))
    _, sobel_thresh = cv2.threshold(sobel_y_8u, 50, 255, cv2.THRESH_BINARY)
    
    edge_density = np.sum(sobel_thresh > 0) / sobel_thresh.size

    if edge_density < 0.06:
        return False

    return True

SOURCE_DIR = "dataset_sorted"
AXIAL_DIR = "dataset_axial_final"
SAGITTAL_DIR = "dataset_sagittal_final"
IGNORED_DIR = "dataset_ignored"


for folder in [AXIAL_DIR, SAGITTAL_DIR, IGNORED_DIR]:
    os.makedirs(folder, exist_ok=True)

def process_folders():
    stats = {'axial': 0, 'sagittal': 0, 'ignored': 0}

    for label in ['0', '1']:
        src_dir = os.path.join(SOURCE_DIR, label)
        if not os.path.exists(src_dir): continue
            

        axial_class_dir = os.path.join(AXIAL_DIR, label)
        sagittal_class_dir = os.path.join(SAGITTAL_DIR, label)
        ignored_class_dir = os.path.join(IGNORED_DIR, label)
        
        os.makedirs(axial_class_dir, exist_ok=True)
        os.makedirs(sagittal_class_dir, exist_ok=True)
        os.makedirs(ignored_class_dir, exist_ok=True)
        
        print(f"Сортирую папку {label}...")
        
        for f in os.listdir(src_dir):
            if not f.endswith('.png'): continue
            img_path = os.path.join(src_dir, f)
            
            #  Проверяем на аксиальный срез
            if is_axial(img_path):
                shutil.copy(img_path, os.path.join(axial_class_dir, f))
                stats['axial'] += 1
                continue # Переходим к следующей картинке
                
            #  Если не аксиальный, проверяем на сагиттальный
            if is_saggital(img_path):
                shutil.copy(img_path, os.path.join(sagittal_class_dir, f))
                stats['sagittal'] += 1
                continue
                
            # Если не подошел ни туда, ни туда
            shutil.copy(img_path, os.path.join(ignored_class_dir, f))
            stats['ignored'] += 1

    print("\nитоги")
    print(f"Аксиальных: {stats['axial']}  папка '{AXIAL_DIR}'")
    print(f"Саггитальных: {stats['sagittal']}  папка '{SAGITTAL_DIR}'")
    print(f"Отсеяно: {stats['ignored']}  папка '{IGNORED_DIR}'")
    print(f"\nОригиналы остались нетронутыми в папке '{SOURCE_DIR}'")

if __name__ == '__main__':
    process_folders()