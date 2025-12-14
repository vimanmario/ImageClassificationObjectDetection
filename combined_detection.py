import numpy as np
import time
import cv2

classification_base = r"C:\Users\Viman Mario\Desktop\FACULTATE\Master\Sisteme dedicate pentru IoT\Proiect\ImageClassificationModel"
classification_prototxt = classification_base + r"\bvlc_googlenet.prototxt"
classification_model = classification_base + r"\bvlc_googlenet.caffemodel"
classification_labels = classification_base + r"\classification_classes_ILSVRC2012.txt"

detection_base = r"C:\Users\Viman Mario\Desktop\FACULTATE\Master\Sisteme dedicate pentru IoT\Proiect\ObjectDetectionModel"
detection_prototxt = detection_base + r"\MobileNetSSD_deploy.prototxt"
detection_model = detection_base + r"\MobileNetSSD_deploy.caffemodel"

image_path = detection_base + r"\car-bicycle.png"

conf_limit = 0.25

DETECTION_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                     "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", 
                     "dog", "horse", "motorbike", "person", "pottedplant", "sheep", 
                     "sofa", "train", "tv/monitor"]

COLORS = np.random.uniform(0, 255, size=(len(DETECTION_CLASSES), 3))

rows = open(classification_labels).read().strip().split("\n")
classification_classes = [r.split(",")[0] for r in rows]

print("=" * 60)
print("Loading models...")
print("=" * 60)

classification_net = cv2.dnn.readNetFromCaffe(classification_prototxt, classification_model)
detection_net = cv2.dnn.readNetFromCaffe(detection_prototxt, detection_model)

print("✓ Classification model loaded (GoogLeNet)")
print("✓ Object detection model loaded (MobileNetSSD)")

image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not load image from {image_path}")
    exit()

(h, w) = image.shape[:2]
result_image = image.copy()

print("\n" + "=" * 60)
print("OBJECT DETECTION")
print("=" * 60)

blob_detection = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 
                                       0.007843, (300, 300), 127.5)

detection_net.setInput(blob_detection)
start = time.time()
detections = detection_net.forward()
detection_time = time.time() - start

detected_objects = []
for i in np.arange(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > conf_limit:
        idx = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        label = "{}: {:.2f}%".format(DETECTION_CLASSES[idx], confidence * 100)
        detected_objects.append((DETECTION_CLASSES[idx], confidence))
        print(f"  • {label}")

        cv2.rectangle(result_image, (startX, startY), (endX, endY),
                     COLORS[idx], 2)

        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(result_image, label, (startX, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

print(f"Detection time: {detection_time:.5f} seconds")
print(f"Objects found: {len(detected_objects)}")

print("\n" + "=" * 60)
print("IMAGE CLASSIFICATION")
print("=" * 60)

# Image Classification
blob_classification = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))

classification_net.setInput(blob_classification)
start = time.time()
preds = classification_net.forward()
classification_time = time.time() - start

# Top 5 predicții
idxs = np.argsort(preds[0])[::-1][:5]

print(f"Top 5 classifications:")
for (i, idx) in enumerate(idxs):
    print(f"  {i + 1}. {classification_classes[idx]}: {preds[0][idx] * 100:.2f}%")

print(f"Classification time: {classification_time:.5f} seconds")

top_class_text = f"Image: {classification_classes[idxs[0]]} ({preds[0][idxs[0]] * 100:.1f}%)"
cv2.putText(result_image, top_class_text, (10, 30),
           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

info_text = f"Objects detected: {len(detected_objects)}"
cv2.putText(result_image, info_text, (10, h - 20),
           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Overall scene: {classification_classes[idxs[0]]}")
print(f"Detected objects: {', '.join([obj[0] for obj in detected_objects]) if detected_objects else 'None'}")
print(f"Total processing time: {detection_time + classification_time:.5f} seconds")
print("=" * 60)

# Afișare rezultat
cv2.imshow("Combined: Classification + Object Detection", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
