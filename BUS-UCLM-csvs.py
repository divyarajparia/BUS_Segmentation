import os
import pandas as pd
import random

DATASET_DIR = "dataset/BioMedicalDataset/BUS-UCLM"
SPLIT_COUNTS = [481, 161, 64]  # train, test, val

def collect_pairs(class_name):
    img_dir = os.path.join(DATASET_DIR, class_name, "images")
    mask_dir = os.path.join(DATASET_DIR, class_name, "masks")
    pairs = []
    for fn in os.listdir(img_dir):
        if fn.lower().endswith(".png") and os.path.exists(os.path.join(mask_dir, fn)):
            pairs.append({
                "image_path": f"{class_name} {fn}",
                "mask_path": f"{class_name} {fn}"
            })
    return pairs

def main():
    all_pairs = collect_pairs("benign") + collect_pairs("malignant")
    random.seed(4321)
    random.shuffle(all_pairs)
    total = sum(SPLIT_COUNTS)
    if len(all_pairs) < total:
        print(f"Warning: Not enough samples ({len(all_pairs)}) for requested split ({total}). Adjusting splits.")
        # Adjust splits to fit available data
        train_n = int(0.7 * len(all_pairs))
        test_n = int(0.23 * len(all_pairs))
        val_n = len(all_pairs) - train_n - test_n
        counts = [train_n, test_n, val_n]
    else:
        counts = SPLIT_COUNTS

    train = all_pairs[:counts[0]]
    test = all_pairs[counts[0]:counts[0]+counts[1]]
    val = all_pairs[counts[0]+counts[1]:counts[0]+counts[1]+counts[2]]

    for split, data in zip(["train", "test", "val"], [train, test, val]):
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(DATASET_DIR, f"{split}_frame.csv"), index=False)

    print("Done! CSVs written to", DATASET_DIR)

if __name__ == "__main__":
    main()