import cv2
import numpy as np
import json
from PIL import Image
from skimage.measure import label, regionprops

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

from paddleocr import PaddleOCR

def preprocess(img_path, size=(512, 512)):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.resize(gray, size)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return img, rgb
resnet = ResNet50(weights="imagenet", include_top=False)
feat_layer = "conv3_block4_out"
model = tf.keras.Model(
    inputs=resnet.input,
    outputs=resnet.get_layer(feat_layer).output
)

def geometry_diff(imgA, imgB):
    xA = preprocess_input(np.expand_dims(imgA, 0))
    xB = preprocess_input(np.expand_dims(imgB, 0))

    fA = model.predict(xA, verbose=0)
    fB = model.predict(xB, verbose=0)

    diff = np.mean(np.abs(fA - fB), axis=-1)[0]
    diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
    return diff
def extract_regions(diff_map, threshold=0.25):
    mask = diff_map > threshold
    labels = label(mask)

    boxes = []
    for r in regionprops(labels):
        if r.area > 50:
            y1, x1, y2, x2 = r.bbox
            boxes.append((x1, y1, x2, y2))
    return boxes, mask
ocr = PaddleOCR(use_angle_cls=True, lang="en")

def extract_text(img):
    result = ocr.ocr(img, cls=True)
    texts = []

    for line in result[0]:
        box = np.array(line[0]).astype(int)
        text = line[1][0]
        texts.append({
            "text": text,
            "bbox": box.tolist()
        })
    return texts
def compare_text(textA, textB):
    mapA = {t["text"]: t for t in textA}
    mapB = {t["text"]: t for t in textB}

    removed = [mapA[k] for k in mapA if k not in mapB]
    added = [mapB[k] for k in mapB if k not in mapA]

    return removed, added
def visualize(img, heatmap, boxes, text_added, text_removed):
    h, w = img.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap_color = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(img, 0.7, heatmap_color, 0.3, 0)

    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)

    for t in text_added:
        pts = np.array(t["bbox"])
        cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)

    for t in text_removed:
        pts = np.array(t["bbox"])
        cv2.polylines(overlay, [pts], True, (255, 0, 0), 2)

    return overlay
imgA_raw, imgA = preprocess("img.png")
imgB_raw, imgB = preprocess("img_1.png")

# Geometry diff
diff_map = geometry_diff(imgA, imgB)
boxes, mask = extract_regions(diff_map)

# Text diff
textA = extract_text(imgA_raw)
textB = extract_text(imgB_raw)
text_removed, text_added = compare_text(textA, textB)

# Visualization
final_vis = visualize(imgB_raw, diff_map, boxes, text_added, text_removed)
cv2.imwrite("final_diff.png", final_vis)

# JSON output
result = {
    "geometry_changes": boxes,
    "text_added": text_added,
    "text_removed": text_removed
}

with open("diff_report.json", "w") as f:
    json.dump(result, f, indent=2)

print("✔ Comparison complete")
print("✔ final_diff.png saved")
print("✔ diff_report.json saved")
