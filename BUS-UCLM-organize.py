import os, shutil
from PIL import Image

RAW_ROOT = "dataset/BioMedicalDataset/BUS-UCLM/data/BUS-UCLM"
IMG_IN   = os.path.join(RAW_ROOT, "images")
MASK_IN  = os.path.join(RAW_ROOT, "masks")
NEW_ROOT = "dataset/BioMedicalDataset/BUS-UCLM"

def make_dirs():
    for lbl in ("benign","malignant"):
        os.makedirs(os.path.join(NEW_ROOT, lbl, "images"), exist_ok=True)
        os.makedirs(os.path.join(NEW_ROOT, lbl, "masks"),  exist_ok=True)

def classify_mask(mask_path):
    mask = Image.open(mask_path).convert("RGB")
    r,g,_ = mask.split()
    # sum up each channel
    r_sum = sum(r.getdata())
    g_sum = sum(g.getdata())
    if g_sum > r_sum:
        return "benign"
    if r_sum > g_sum:
        return "malignant"
    return None  # skip normals

def run():
    make_dirs()
    for fn in os.listdir(MASK_IN):
        if not fn.lower().endswith(".png"):
            continue
        mask_path = os.path.join(MASK_IN, fn)
        img_path  = os.path.join(IMG_IN, fn)
        if not os.path.exists(img_path):
            print(f"⚠️ image missing for {fn}")
            continue
        lbl = classify_mask(mask_path)
        if lbl is None:
            continue
        dst_img  = os.path.join(NEW_ROOT, lbl, "images", fn)
        dst_mask = os.path.join(NEW_ROOT, lbl, "masks",  fn)
        shutil.copy(img_path,  dst_img)
        shutil.copy(mask_path, dst_mask)
    print("BUS-UCLM reorganized by mask color at", NEW_ROOT)

if __name__ == "__main__":
    run()