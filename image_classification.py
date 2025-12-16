import numpy as np
import time
import cv2
import os
import random
import glob

base_path = r"C:\Users\Viman Mario\Desktop\FACULTATE\Master\Sisteme dedicate pentru IoT\Proiect\ImageClassificationModel"

images_folder = base_path + r"\images"

label_path = base_path + r"\classification_classes_ILSVRC2012.txt"
prototxt_path = base_path + r"\bvlc_googlenet.prototxt"
model_path = base_path + r"\bvlc_googlenet.caffemodel"

image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif', '*.tiff']
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(images_folder, ext)))
    image_files.extend(glob.glob(os.path.join(images_folder, ext.upper())))

if not image_files:
    print(f"Error: No images found in {images_folder}")
    print("Please add some images to the folder or update the path.")
    exit()

image_path = random.choice(image_files)
print(f"Selected image: {os.path.basename(image_path)}")
print(f"Full path: {image_path}\n")

image = cv2.imread(image_path)
rows = open(label_path).read().strip().split("\n")
classes = [r.split(",")[0] for r in rows]

blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))

print("Loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

net.setInput(blob)
start = time.time()
preds = net.forward()
end = time.time()

print("Classification time: {:.5f} seconds".format(end - start))

idxs = np.argsort(preds[0])[::-1][:5]

for (i, idx) in enumerate(idxs):
    if i == 0:
        text = "Label: {}, {:.2f}%".format(classes[idx], preds[0][idx] * 100)
        cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

    print("{}. label: {}, probability: {:.5f}".format(
        i + 1, classes[idx], preds[0][idx]))

# Redimensionare imagine pentru vizualizare mai bună
display_width = 600  # Lățime dorită
h, w = image.shape[:2]
aspect_ratio = w / h
display_height = int(display_width / aspect_ratio)

# Redimensionare păstrând proporțiile
display_image = cv2.resize(image, (display_width, display_height))

cv2.imshow("Image Classification - GoogLeNet", display_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
