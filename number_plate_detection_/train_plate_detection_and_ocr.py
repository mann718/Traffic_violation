"""
==============================================================
  Vehicle Number Plate — FULL TRAINING PIPELINE
  Part 1 : YOLOv8  — Plate Detection
  Part 2 : CRNN    — Plate OCR / Text Recognition
==============================================================

Requirements
------------
pip install ultralytics torch torchvision torchaudio
pip install easyocr opencv-python pillow matplotlib tqdm
pip install scikit-learn

Dataset
-------
Kaggle: andrewmvd/car-plate-detection
  └── car-plate-detection/
        ├── images/       *.png / *.jpg
        └── annotations/  *.xml  (Pascal VOC format)
"""

# ─────────────────────────────────────────────────────────────
# SHARED IMPORTS
# ─────────────────────────────────────────────────────────────
import os, re, glob, shutil, random, xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
from collections import Counter

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# ─────────────────────────────────────────────────────────────
# GLOBAL PATHS  — change DATASET_PATH to your local folder
# ─────────────────────────────────────────────────────────────
DATASET_PATH = './car-plate-detection'
IMAGES_PATH  = os.path.join(DATASET_PATH, 'images')
ANNOTS_PATH  = os.path.join(DATASET_PATH, 'annotations')

YOLO_DIR     = './yolo_dataset'          # prepared YOLO data
CRNN_CROPS   = './crnn_crops'            # cropped plate images for CRNN
CRNN_LABELS  = './crnn_labels.txt'       # image_path \t plate_text
YOLO_MODEL   = './runs/detect/train/weights/best.pt'
CRNN_MODEL   = './crnn_best.pth'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}')


# ==============================================================
# HELPER — parse Pascal VOC XML
# ==============================================================
def parse_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.findtext('filename', default='unknown')
    width    = int(root.findtext('size/width',  default=0))
    height   = int(root.findtext('size/height', default=0))
    boxes = []
    for obj in root.findall('object'):
        bnd = obj.find('bndbox')
        if bnd is not None:
            xmin = int(float(bnd.findtext('xmin', default=0)))
            ymin = int(float(bnd.findtext('ymin', default=0)))
            xmax = int(float(bnd.findtext('xmax', default=0)))
            ymax = int(float(bnd.findtext('ymax', default=0)))
            boxes.append((xmin, ymin, xmax, ymax))
    return filename, width, height, boxes


# ==============================================================
# ██████╗  █████╗ ██████╗ ████████╗   ██╗
# ██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝  ███║
# ██████╔╝███████║██████╔╝   ██║      ██║
# ██╔═══╝ ██╔══██║██╔══██╗   ██║      ██║
# ██║     ██║  ██║██║  ██║   ██║      ██║
# ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝      ╚═╝
#
#  YOLOv8  —  Plate Detection Training
# ==============================================================

def prepare_yolo_dataset(val_split=0.2, seed=42):
    """
    Convert Pascal VOC XML annotations to YOLO txt format.
    Splits into train / val sets.
    Directory layout created:
        yolo_dataset/
          images/train/   images/val/
          labels/train/   labels/val/
          data.yaml
    """
    image_files = sorted(glob.glob(os.path.join(IMAGES_PATH, '*.png')) +
                         glob.glob(os.path.join(IMAGES_PATH, '*.jpg')) +
                         glob.glob(os.path.join(IMAGES_PATH, '*.jpeg')))
    annot_files = sorted(glob.glob(os.path.join(ANNOTS_PATH, '*.xml')))

    img_dict = {Path(f).stem: f for f in image_files}
    xml_dict = {Path(f).stem: f for f in annot_files}
    common   = sorted(set(img_dict.keys()) & set(xml_dict.keys()))

    # shuffle & split
    random.seed(seed)
    random.shuffle(common)
    n_val   = max(1, int(len(common) * val_split))
    val_set = set(common[:n_val])

    for split in ('train', 'val'):
        os.makedirs(os.path.join(YOLO_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DIR, 'labels', split), exist_ok=True)

    converted = 0
    for key in tqdm(common, desc='Preparing YOLO dataset'):
        split = 'val' if key in val_set else 'train'
        img_src = img_dict[key]
        _, W, H, boxes = parse_annotation(xml_dict[key])

        if W == 0 or H == 0:
            # read actual size from image
            img_tmp = cv2.imread(img_src)
            if img_tmp is None:
                continue
            H, W = img_tmp.shape[:2]

        # copy image
        ext = Path(img_src).suffix
        dst_img = os.path.join(YOLO_DIR, 'images', split, key + ext)
        shutil.copy2(img_src, dst_img)

        # write label (class 0 = licence_plate)
        dst_lbl = os.path.join(YOLO_DIR, 'labels', split, key + '.txt')
        with open(dst_lbl, 'w') as f:
            for (xmin, ymin, xmax, ymax) in boxes:
                cx = ((xmin + xmax) / 2) / W
                cy = ((ymin + ymax) / 2) / H
                bw = (xmax - xmin) / W
                bh = (ymax - ymin) / H
                f.write(f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')
        converted += 1

    # write data.yaml
    yaml_path = os.path.join(YOLO_DIR, 'data.yaml')
    abs_yolo  = os.path.abspath(YOLO_DIR)
    with open(yaml_path, 'w') as f:
        f.write(f"path: {abs_yolo}\n")
        f.write(f"train: images/train\n")
        f.write(f"val:   images/val\n")
        f.write(f"nc: 1\n")
        f.write(f"names: ['licence_plate']\n")

    print(f'\n✅ YOLO dataset ready — {converted} images converted')
    print(f'   Train: {converted - n_val}   Val: {n_val}')
    print(f'   Saved to: {abs_yolo}')
    return yaml_path


def train_yolo(yaml_path,
               model_size='n',
               epochs=50,
               imgsz=640,
               batch=16):
    """
    Fine-tune YOLOv8 on the licence-plate dataset.

    Parameters
    ----------
    yaml_path  : str   — path to data.yaml
    model_size : str   — 'n' | 's' | 'm' | 'l' | 'x'
    epochs     : int
    imgsz      : int   — input image size
    batch      : int
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Run: pip install ultralytics")

    print('\n' + '='*60)
    print('  PART 1 — YOLOv8 PLATE DETECTION TRAINING')
    print('='*60)

    model = YOLO(f'yolov8{model_size}.pt')   # downloads pretrained weights

    results = model.train(
        data    = yaml_path,
        epochs  = epochs,
        imgsz   = imgsz,
        batch   = batch,
        device  = DEVICE,
        project = './runs/detect',
        name    = 'train',
        exist_ok= True,
        patience= 15,           # early stopping
        save    = True,
        plots   = True,
        augment = True,
        # augmentation hyper-params
        hsv_h   = 0.015,
        hsv_s   = 0.7,
        hsv_v   = 0.4,
        degrees = 5.0,
        translate=0.1,
        scale   = 0.5,
        flipud  = 0.0,
        fliplr  = 0.5,
        mosaic  = 1.0,
    )

    best_weights = './runs/detect/train/weights/best.pt'
    print(f'\n✅ YOLOv8 training complete!')
    print(f'   Best weights : {best_weights}')
    print(f'   mAP@0.5      : {results.results_dict.get("metrics/mAP50(B)", "N/A")}')
    return best_weights


def validate_yolo(weights_path, yaml_path):
    """Run validation on the trained YOLOv8 model."""
    from ultralytics import YOLO
    model = YOLO(weights_path)
    metrics = model.val(data=yaml_path, device=DEVICE)
    print(f'\n📊 Validation results:')
    print(f'   mAP@0.5    : {metrics.box.map50:.4f}')
    print(f'   mAP@0.5:95 : {metrics.box.map:.4f}')
    print(f'   Precision  : {metrics.box.mp:.4f}')
    print(f'   Recall     : {metrics.box.mr:.4f}')
    return metrics


def detect_with_yolo(image_path, weights_path, conf=0.25):
    """Run inference using the trained YOLO model (returns crops)."""
    from ultralytics import YOLO
    model  = YOLO(weights_path)
    result = model.predict(image_path, conf=conf, device=DEVICE)[0]
    img    = cv2.imread(image_path)
    crops  = []
    for box in result.boxes.xyxy.cpu().numpy().astype(int):
        x1, y1, x2, y2 = box
        crops.append(img[y1:y2, x1:x2])
    return crops, result


# ==============================================================
# ██████╗  █████╗ ██████╗ ████████╗   ██████╗
# ██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝  ╚════██╗
# ██████╔╝███████║██████╔╝   ██║       █████╔╝
# ██╔═══╝ ██╔══██║██╔══██╗   ██║      ██╔═══╝
# ██║     ██║  ██║██║  ██║   ██║      ███████╗
# ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝      ╚══════╝
#
#  CRNN  —  Plate OCR Training
# ==============================================================

# ── Alphabet (characters the CRNN can predict) ────────────────
ALPHABET  = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
BLANK_IDX = len(ALPHABET)          # CTC blank token index
NUM_CLASSES = len(ALPHABET) + 1    # +1 for CTC blank

# ── Image size fed to CRNN ────────────────────────────────────
IMG_H = 32
IMG_W = 128


# ── 1. Collect plate crops + pseudo-labels via EasyOCR ────────
def collect_crnn_crops(use_easyocr_labels=True):
    """
    Crop licence plates using GT bounding boxes.
    Optionally generate pseudo text labels with EasyOCR for training.
    Saves crops to CRNN_CROPS/ and writes CRNN_LABELS file.
    """
    os.makedirs(CRNN_CROPS, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(IMAGES_PATH, '*.png')) +
                         glob.glob(os.path.join(IMAGES_PATH, '*.jpg')) +
                         glob.glob(os.path.join(IMAGES_PATH, '*.jpeg')))
    annot_files = sorted(glob.glob(os.path.join(ANNOTS_PATH, '*.xml')))
    img_dict = {Path(f).stem: f for f in image_files}
    xml_dict = {Path(f).stem: f for f in annot_files}
    common   = sorted(set(img_dict.keys()) & set(xml_dict.keys()))

    if use_easyocr_labels:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=(DEVICE == 'cuda'), verbose=False)
        print('✅ EasyOCR initialised for label generation')

    records = []
    idx     = 0
    for key in tqdm(common, desc='Collecting CRNN crops'):
        img_bgr = cv2.imread(img_dict[key])
        if img_bgr is None:
            continue
        _, _, _, boxes = parse_annotation(xml_dict[key])
        for (xmin, ymin, xmax, ymax) in boxes:
            crop = img_bgr[ymin:ymax, xmin:xmax]
            if crop.size == 0:
                continue

            crop_name = f'plate_{idx:05d}.jpg'
            crop_path = os.path.join(CRNN_CROPS, crop_name)
            cv2.imwrite(crop_path, crop)

            if use_easyocr_labels:
                rgb     = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                results = reader.readtext(rgb, detail=1, paragraph=False)
                text    = ' '.join(r[1] for r in results)
                text    = re.sub(r'[^A-Za-z0-9]', '', text).upper()
            else:
                text = 'UNKNOWN'    # placeholder when no labels available

            if text:
                records.append(f'{crop_path}\t{text}')
            idx += 1

    with open(CRNN_LABELS, 'w') as f:
        f.write('\n'.join(records))

    print(f'\n✅ Collected {len(records)} labelled plate crops')
    print(f'   Crops : {CRNN_CROPS}')
    print(f'   Labels: {CRNN_LABELS}')
    return CRNN_LABELS


# ── 2. Dataset ────────────────────────────────────────────────
class PlateOCRDataset(Dataset):
    """
    Loads (plate_image, label_text) pairs.
    Returns:
        image  — (1, IMG_H, IMG_W) float tensor
        target — list[int] encoded character indices
        target_length — len(target)
        raw_text — original string
    """
    def __init__(self, label_file, transform=None):
        self.samples   = []
        self.transform = transform
        with open(label_file) as f:
            for line in f:
                line = line.strip()
                if '\t' not in line:
                    continue
                path, text = line.split('\t', 1)
                text = text.upper()
                # keep only chars in ALPHABET
                text = ''.join(c for c in text if c in ALPHABET)
                if text and os.path.exists(path):
                    self.samples.append((path, text))

    def encode(self, text):
        return [ALPHABET.index(c) for c in text]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, text = self.samples[idx]
        img = Image.open(path).convert('L')      # grayscale
        img = img.resize((IMG_W, IMG_H))
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        target = self.encode(text)
        return img, torch.tensor(target, dtype=torch.long), len(target), text


def crnn_collate(batch):
    """Custom collate: pad targets to same length for CTC."""
    images, targets, target_lengths, texts = zip(*batch)
    images  = torch.stack(images, 0)
    t_lens  = torch.tensor(target_lengths, dtype=torch.long)
    targets = torch.cat(targets, 0)
    return images, targets, t_lens, texts


# ── 3. CRNN Model ─────────────────────────────────────────────
class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network for sequence recognition.

    Architecture:
        CNN  : VGG-style feature extractor → (B, C, 1, W')
        RNN  : Bidirectional LSTM × 2
        FC   : Linear → NUM_CLASSES (used with CTC loss)
    """
    def __init__(self, img_h=IMG_H, nc=1, num_classes=NUM_CLASSES,
                 nh=256):
        super().__init__()
        assert img_h % 16 == 0, 'img_h must be divisible by 16'

        # ── CNN backbone ──────────────────────────────────────
        self.cnn = nn.Sequential(
            # block 1
            nn.Conv2d(nc, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),                          # H/2, W/2

            # block 2
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),                          # H/4, W/4

            # block 3
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),                # H/8, W/4

            # block 4
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),                # H/16, W/4

            # block 5 — collapse height to 1
            nn.Conv2d(512, 512, (img_h // 16, 1), 1, 0),
            nn.BatchNorm2d(512), nn.ReLU(True),
        )

        # ── Bidirectional LSTM ────────────────────────────────
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, num_classes),
        )

    def forward(self, x):
        # x : (B, 1, H, W)
        conv = self.cnn(x)           # (B, 512, 1, W')
        conv = conv.squeeze(2)       # (B, 512, W')
        conv = conv.permute(2, 0, 1) # (W', B, 512)  — seq-first
        out  = self.rnn(conv)        # (W', B, num_classes)
        return out


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size,
                           bidirectional=True, batch_first=False)
        self.fc  = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)   # (T, B, 2*H)
        T, B, H = out.shape
        out = out.view(T * B, H)
        out = self.fc(out)
        out = out.view(T, B, -1)
        return out


# ── 4. CTC Decoder ───────────────────────────────────────────
def ctc_decode(log_probs):
    """Greedy CTC decode: collapse repeated chars and remove blanks."""
    indices = log_probs.argmax(dim=2)  # (T, B)
    results = []
    for b in range(indices.shape[1]):
        seq   = indices[:, b].tolist()
        chars = []
        prev  = None
        for idx in seq:
            if idx != prev and idx != BLANK_IDX:
                chars.append(ALPHABET[idx])
            prev = idx
        results.append(''.join(chars))
    return results


# ── 5. Training loop ──────────────────────────────────────────
def train_crnn(label_file=CRNN_LABELS,
               epochs=30,
               batch_size=32,
               lr=1e-3,
               val_split=0.15):
    """
    Train the CRNN OCR model with CTC loss.

    Parameters
    ----------
    label_file : str   — TSV file produced by collect_crnn_crops()
    epochs     : int
    batch_size : int
    lr         : float — initial learning rate
    val_split  : float — fraction used for validation
    """
    print('\n' + '='*60)
    print('  PART 2 — CRNN OCR TRAINING')
    print('='*60)

    # ── Dataset split ─────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    full_ds  = PlateOCRDataset(label_file, transform=transform)
    n_val    = max(1, int(len(full_ds) * val_split))
    n_train  = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  collate_fn=crnn_collate,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, collate_fn=crnn_collate,
                              num_workers=2, pin_memory=True)

    print(f'  Train samples: {n_train}   Val samples: {n_val}')

    # ── Model, loss, optimiser ────────────────────────────────
    model     = CRNN().to(DEVICE)
    criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # ── Training ──────────────────────────────────────────────
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):

        # — Train —
        model.train()
        train_loss = 0.0
        for images, targets, t_lens, _ in train_loader:
            images  = images.to(DEVICE)
            targets = targets.to(DEVICE)

            log_probs   = model(images)              # (T, B, C)
            input_lens  = torch.full(
                (images.size(0),), log_probs.size(0),
                dtype=torch.long
            )

            loss = criterion(
                log_probs.log_softmax(2),
                targets, input_lens, t_lens
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # — Validate —
        model.eval()
        val_loss  = 0.0
        correct   = 0
        total     = 0
        with torch.no_grad():
            for images, targets, t_lens, texts in val_loader:
                images  = images.to(DEVICE)
                targets = targets.to(DEVICE)

                log_probs  = model(images)
                input_lens = torch.full(
                    (images.size(0),), log_probs.size(0),
                    dtype=torch.long
                )
                loss = criterion(
                    log_probs.log_softmax(2),
                    targets, input_lens, t_lens
                )
                val_loss += loss.item()

                preds = ctc_decode(log_probs.cpu())
                for pred, gt in zip(preds, texts):
                    if pred.upper() == gt.upper():
                        correct += 1
                    total += 1

        val_loss /= len(val_loader)
        val_acc   = correct / total if total else 0
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'Epoch {epoch:3d}/{epochs} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'Val Acc: {val_acc*100:.1f}%')

        # — Save best ——
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CRNN_MODEL)
            print(f'             ✅ Best model saved → {CRNN_MODEL}')

    print(f'\n✅ CRNN training complete! Best val loss: {best_val_loss:.4f}')
    _plot_crnn_history(history)
    return model, history


def _plot_crnn_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'],   label='Val Loss')
    ax1.set_title('CRNN — Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2.plot([v * 100 for v in history['val_acc']], color='green')
    ax2.set_title('CRNN — Validation Accuracy (%)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')

    plt.tight_layout()
    plt.savefig('crnn_training_curves.png', dpi=120)
    plt.show()
    print('📊 Training curves saved → crnn_training_curves.png')


# ── 6. Inference with trained CRNN ────────────────────────────
def load_crnn(weights_path=CRNN_MODEL):
    model = CRNN().to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    return model


def crnn_predict(model, plate_crop_bgr):
    """
    Predict plate text from a BGR numpy crop using the trained CRNN.
    Returns predicted string.
    """
    transform = transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    img   = Image.fromarray(cv2.cvtColor(plate_crop_bgr, cv2.COLOR_BGR2GRAY))
    inp   = transform(img).unsqueeze(0).to(DEVICE)   # (1, 1, H, W)
    with torch.no_grad():
        out = model(inp)                              # (T, 1, C)
    return ctc_decode(out.cpu())[0]


# ==============================================================
# ████████╗██╗   ██╗███████╗███████╗
# ╚══██╔══╝██║   ██║██╔════╝██╔════╝
#    ██║   ██║   ██║███████╗█████╗
#    ██║   ██║   ██║╚════██║██╔══╝
#    ██║   ╚██████╔╝███████║███████╗
#    ╚═╝    ╚═════╝ ╚══════╝╚══════╝
#
#  Full combined pipeline (detect → OCR) after training
# ==============================================================

def full_pipeline_trained(image_path,
                           yolo_weights=YOLO_MODEL,
                           crnn_weights=CRNN_MODEL,
                           conf=0.25):
    """
    End-to-end inference using BOTH trained models:
      1. YOLOv8 detects plate bounding boxes
      2. CRNN reads the plate text from each crop
    """
    print(f'\n🚗 Processing: {os.path.basename(image_path)}')

    # ── Step 1: Detection ─────────────────────────────────────
    crops, yolo_result = detect_with_yolo(image_path, yolo_weights, conf)
    print(f'   Detected {len(crops)} plate(s)')

    if not crops:
        print('   ⚠️  No plates detected.')
        return

    # ── Step 2: OCR ───────────────────────────────────────────
    crnn_model = load_crnn(crnn_weights)
    results    = []
    for i, crop in enumerate(crops):
        text = crnn_predict(crnn_model, crop)
        results.append(text)
        print(f'   Plate {i+1}: {text}')

    # ── Visualise ─────────────────────────────────────────────
    img_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    boxes   = yolo_result.boxes.xyxy.cpu().numpy().astype(int)

    fig, axes = plt.subplots(1, len(crops) + 1,
                             figsize=(5 * (len(crops) + 1), 5))
    if len(crops) == 0:
        axes = [axes]

    vis = img_rgb.copy()
    for (x1, y1, x2, y2), text in zip(boxes, results):
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(vis, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    axes[0].imshow(vis)
    axes[0].set_title('Detections')
    axes[0].axis('off')

    for i, (crop, text) in enumerate(zip(crops, results)):
        axes[i + 1].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        axes[i + 1].set_title(f'Plate: {text}')
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.show()
    return results


# ==============================================================
# MAIN — run the entire training pipeline
# ==============================================================
if __name__ == '__main__':

    print('\n' + '█'*60)
    print('  VEHICLE NUMBER PLATE — FULL TRAINING PIPELINE')
    print('█'*60)

    # ──────────────────────────────────────────────────────────
    # PART 1 : YOLOv8 Detection Training
    # ──────────────────────────────────────────────────────────
    yaml_path = prepare_yolo_dataset(val_split=0.2)
    yolo_best = train_yolo(
        yaml_path  = yaml_path,
        model_size = 'n',     # 'n' = nano (fastest). Use 's' or 'm' for better mAP
        epochs     = 50,
        imgsz      = 640,
        batch      = 16,
    )
    validate_yolo(yolo_best, yaml_path)

    # ──────────────────────────────────────────────────────────
    # PART 2 : CRNN OCR Training
    # ──────────────────────────────────────────────────────────
    label_file = collect_crnn_crops(use_easyocr_labels=True)
    crnn_model, history = train_crnn(
        label_file = label_file,
        epochs     = 30,
        batch_size = 32,
        lr         = 1e-3,
    )

    # ──────────────────────────────────────────────────────────
    # OPTIONAL: test the combined pipeline on a single image
    # ──────────────────────────────────────────────────────────
    # full_pipeline_trained('./your_vehicle_image.jpg')

    print('\n' + '='*60)
    print('  ✅ ALL DONE!')
    print(f'  YOLOv8 weights → {YOLO_MODEL}')
    print(f'  CRNN weights   → {CRNN_MODEL}')
    print('='*60)
