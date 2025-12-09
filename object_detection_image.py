import numpy as np
import cv2

base_path = r"C:\Users\Viman Mario\Desktop\FACULTATE\Master\Sisteme dedicate pentru IoT\Proiect\ObjectDetectionModel"

model_path = base_path + r"\MobileNetSSD_deploy.caffemodel"
prototxt_path = base_path + r"\MobileNetSSD_deploy.prototxt"
image_path = base_path + r"\car-bicycle.png"

conf_limit = 0.25

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
"horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
"train", "tv/monitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("Loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

image = cv2.imread(image_path)
(h, w) = image.shape[:2]

blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843,
                             (300, 300), 127.5)

print("Sending image through the network...")
net.setInput(blob)
detections = net.forward()

for i in np.arange(0, detections.shape[2]):

    confidence = detections[0, 0, i, 2]

    if confidence > conf_limit:
        idx = int(detections[0, 0, i, 1])

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
        print(label)

        cv2.rectangle(image, (startX, startY), (endX, endY),
                      COLORS[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

cv2.imshow("MobileNetSSD Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
