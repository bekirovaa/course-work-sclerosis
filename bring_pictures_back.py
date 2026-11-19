import os
import shutil

# Откуда возвращаем
TRASH_DIRS = ["dataset_trash",'dataset_axial_clean', 'dataset_sagittal_clean', 'dataset_for_resnet', 'dataset_mra_trash']
DATASET_DIR = "dataset_sorted"

def restore():
    restored_count = 0
    
    for trash_dir in TRASH_DIRS:
        if not os.path.exists(trash_dir):
            continue
            
        print(f"Восстанавливаю из {trash_dir}...")
        
        for label in ['0', '1']:
            trash_class_dir = os.path.join(trash_dir, label)
            target_dir = os.path.join(DATASET_DIR, label)
            
            if not os.path.exists(trash_class_dir): continue
                
            os.makedirs(target_dir, exist_ok=True)
            
            for f in os.listdir(trash_class_dir):
                if f.endswith('.png'):
                    src = os.path.join(trash_class_dir, f)
                    dst = os.path.join(target_dir, f)
                    
                    # Если файл с таким именем уже есть в датасете, добавляем префикс, чтобы не перезаписать
                    if os.path.exists(dst):
                        f = "restored_" + f
                        dst = os.path.join(target_dir, f)
                        
                    shutil.move(src, dst)
                    restored_count += 1
                    

    print(f"\n Возвращено {restored_count} картинок в папку '{DATASET_DIR}'.")

if __name__ == "__main__":
    restore()
print