import os
import shutil
import random
from tqdm import tqdm
from PIL import Image

# ===========================
# USER CONFIGURABLE SETTINGS
# ===========================
SOURCE_DIR = "data/data/raw"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
SPLIT_RATIO = 0.8  # 80% train, 20% val
SEED = 42
# ===========================

def is_image_file(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except:
        return False

def split_data(source, train_dir, val_dir, split_ratio=0.8):
    random.seed(SEED)
    classes = [d for d in os.listdir(source) if os.path.isdir(os.path.join(source, d))]
    
    for cls in tqdm(classes, desc="Processing classes", colour="green"):
        cls_src_path = os.path.join(source, cls)
        imgs = [f for f in os.listdir(cls_src_path) if is_image_file(os.path.join(cls_src_path, f))]
        random.shuffle(imgs)
        split_idx = int(len(imgs) * split_ratio)
        train_imgs = imgs[:split_idx]
        val_imgs = imgs[split_idx:]

        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

        for img in tqdm(train_imgs, desc=f"Copying train/{cls}", leave=False):
            shutil.copy(os.path.join(cls_src_path, img), os.path.join(train_dir, cls, img))
        for img in tqdm(val_imgs, desc=f"Copying val/{cls}", leave=False):
            shutil.copy(os.path.join(cls_src_path, img), os.path.join(val_dir, cls, img))

    print("\n✅ Data split completed successfully.")

if __name__ == "__main__":
    split_data(SOURCE_DIR, TRAIN_DIR, VAL_DIR, SPLIT_RATIO)
