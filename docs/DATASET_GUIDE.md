# 📥 Dataset Download Guide

## Chest X-Ray14 (Pneumonia Detection)

1. Go to: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Sign in with a free Kaggle account
3. Click **Download** (1.2 GB)
4. Extract to: `data/raw/chest_xray/`

Expected structure:
```
data/raw/chest_xray/
    train/
        NORMAL/    ← 1,341 images
        PNEUMONIA/ ← 3,875 images
    test/
        NORMAL/    ← 234 images
        PNEUMONIA/ ← 390 images
    val/
        NORMAL/    ← 8 images
        PNEUMONIA/ ← 8 images
```

---

## HAM10000 (Skin Lesion Classification)

1. Go to: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
2. Click **Download** (~3.5 GB)
3. Extract to: `data/raw/skin_lesion/`

Expected structure:
```
data/raw/skin_lesion/
    HAM10000_images_part_1/   ← ~5,000 images
    HAM10000_images_part_2/   ← ~5,000 images
    HAM10000_metadata.csv     ← labels
```

---

## BraTS 2021 (Brain Tumour Segmentation)

1. Register at: https://www.synapse.org
2. Go to: https://www.synapse.org/#!Synapse:syn27046444
3. Accept data use agreement
4. Download NIfTI files
5. Extract to: `data/raw/brain_mri/`

---

## ⚡ No Downloads? Use Demo Mode!

```bash
python main.py --mode demo
```

Synthetic data is generated automatically — the project runs end-to-end with zero downloads.
