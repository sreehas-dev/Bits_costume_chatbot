import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
import cv2

def load_and_preprocess(img_path, size=(224,224)):
    img = image.load_img(img_path, target_size=size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, 0)
    x = preprocess_input(x)
    return img, x

# 1) load images
img1_pil, img1 = load_and_preprocess(
    "img.png"
)
img2_pil, img2 = load_and_preprocess(
    "img_1.png"
)

# 2) build model that outputs intermediate feature maps
base_model = VGG16(weights="imagenet", include_top=False)
layer_name = "block3_conv3"  # choose layer to compare
model = tf.keras.Model(inputs=base_model.input,
                       outputs=base_model.get_layer(layer_name).output)

# 3) get feature maps
feat1 = model.predict(img1)  # shape: (1, h, w, c)
feat2 = model.predict(img2)

# 4) compute absolute difference
feat_diff = np.abs(feat1 - feat2)

# 5) collapse channels (e.g. mean)
map_diff = np.mean(feat_diff, axis=-1)[0]  # shape: (h,w)

# 6) normalize
heatmap = (map_diff - map_diff.min()) / (map_diff.max() - map_diff.min())

# 7) resize heatmap to original image
heatmap = cv2.resize(heatmap, (img2_pil.size[0], img2_pil.size[1]))

# 8) convert heatmap to color
heatmap_color = cv2.applyColorMap(
    np.uint8(255 * heatmap),
    cv2.COLORMAP_JET
)

# overlay on image
overlay = cv2.addWeighted(
    np.array(img2_pil),
    0.6,
    heatmap_color,
    0.4,
    0
)

# save
cv2.imwrite("deep_diff_map.png", overlay)
print("Saved deep_diff_map.png")
