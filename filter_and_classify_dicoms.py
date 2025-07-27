import os
import pydicom
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

model_path = r'd:\us\Deep-learning-echocardiogram-view-classification\model.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(model_path, map_location=device)
#model.eval()

input_shape = (224, 224)
labels = {0: 'plax', 1: 'psax-av', 2: 'psax-mv', 3: 'psax-ap', 4: 'a4c', 5: 'a5c', 6: 'a3c', 7: 'a2c'}

preprocess = transforms.Compose([
    transforms.Resize(input_shape),
    transforms.ToTensor(),
    # Add normalization if your model requires it, e.g.:
    # transforms.Normalize([0.485], [0.229])
])

def is_video_dicom(ds):
    # Multi-frame DICOMs have NumberOfFrames > 1
    return hasattr(ds, 'NumberOfFrames') and int(ds.NumberOfFrames) > 1

def classify_view(ds):
    try:
        arr = ds.pixel_array
        # If multi-frame, take the first frame
        if arr.ndim == 3:
            frame = arr[0]
        else:
            frame = arr
        # Convert to uint8
        frame = (frame / frame.max() * 255).astype(np.uint8)
        img = Image.fromarray(frame)
        img = img.convert('RGB')
        img = preprocess(img)
        img = img.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img)
            pred_idx = torch.argmax(pred, dim=1).item()
        return labels.get(pred_idx, 'others')
    except Exception as e:
        return 'others'

def process_dicoms(folder):
    results = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith('.dcm'):
                path = os.path.join(root, file)
                try:
                    ds = pydicom.dcmread(path, stop_before_pixels=False)
                    if is_video_dicom(ds):
                        view = classify_view(ds)
                        results.append({'filepath': path, 'view': view})
                except Exception as e:
                    continue
    return pd.DataFrame(results)

# Usage
dataset_folder = r'C:\Users\oronbarazani\OneDrive - Technion\DS\Cardio-Onco Echo SZMC\161905270433\1.2.840.113619.2.182.10808663255255.1558941617.147830'
df = process_dicoms(dataset_folder)
print(df)