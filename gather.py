import os
import shutil

# Куда собираем
TARGET_DIR = "dataset_sorted"


FOLDERS = [ 
    "dataset_for_resnet", 
    "dataset_axial_clean", 
    "dataset_sagittal_clean",
    "dataset_mra_trash",
    "dataset_trash_final",
    "dataset_trash",
    "dataset_trash_strict",
    "dataset_axial_ideal",
    "dataset_sagittal_ideal",
    "dataset_sagittal_final",
    "dataset_axial_final"
]

def gather():
    restored = 0
    os.makedirs(os.path.join(TARGET_DIR, "0"), exist_ok=True)
    os.makedirs(os.path.join(TARGET_DIR, "1"), exist_ok=True)
    
    for folder in FOLDERS:
        if not os.path.exists(folder): continue
            
        for label in ['0', '1']:
            source_dir = os.path.join(folder, label)
            target_dir = os.path.join(TARGET_DIR, label)
            
            if not os.path.exists(source_dir): continue
                
            for f in os.listdir(source_dir):
                if f.endswith('.png'):
                    src = os.path.join(source_dir, f)
                    dst = os.path.join(target_dir, f)
                    if os.path.exists(dst):
                        base, ext = os.path.splitext(f)
                        dst = os.path.join(target_dir, f"{base}_copy{restored}{ext}")
                        
                    shutil.copy(src, dst)
                    restored += 1
                    
    print(f" Все {restored} картинок собраны в папку '{TARGET_DIR}'")

if __name__ == '__main__':
    gather()