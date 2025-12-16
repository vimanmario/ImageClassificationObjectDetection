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

conf_limit = 0.25
CLASSIFICATION_INTERVAL = 30  # Clasifică o dată la 30 de frame-uri (pentru performanță)

DETECTION_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                     "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                     "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                     "sofa", "train", "tv/monitor"]

COLORS = np.random.uniform(0, 255, size=(len(DETECTION_CLASSES), 3))

rows = open(classification_labels).read().strip().split("\n")
classification_classes = [r.split(",")[0] for r in rows]

print("=" * 60)
print("REAL-TIME DETECTION & CLASSIFICATION")
print("=" * 60)
print("Loading models...")

classification_net = cv2.dnn.readNetFromCaffe(classification_prototxt, classification_model)
detection_net = cv2.dnn.readNetFromCaffe(detection_prototxt, detection_model)

print("✓ Classification model loaded (GoogLeNet)")
print("✓ Object detection model loaded (MobileNetSSD)")
print("\nStarting webcam...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("✓ Webcam started successfully")
print("\nControls:")
print("  - Press 'q' to quit")
print("  - Press 's' to save current frame")
print("=" * 60)

frame_count = 0
fps_start_time = time.time()
fps = 0
current_classification = "Analyzing..."
classification_confidence = 0
saved_count = 0

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to grab frame")
            break

        frame_count += 1
        (h, w) = frame.shape[:2]
        result_frame = frame.copy()

        if frame_count % 10 == 0:
            fps_end_time = time.time()
            fps = 10 / (fps_end_time - fps_start_time)
            fps_start_time = fps_end_time

        blob_detection = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 
                                               0.007843, (300, 300), 127.5)

        detection_net.setInput(blob_detection)
        detections = detection_net.forward()

        detected_objects = []
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > conf_limit:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                label = "{}: {:.0f}%".format(DETECTION_CLASSES[idx], confidence * 100)
                detected_objects.append(DETECTION_CLASSES[idx])

                cv2.rectangle(result_frame, (startX, startY), (endX, endY),
                             COLORS[idx], 2)

                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                y = startY - 10 if startY - 10 > 10 else startY + 10

                cv2.rectangle(result_frame, 
                            (startX, y - label_size[1] - 5), 
                            (startX + label_size[0], y + 5),
                            COLORS[idx], -1)

                cv2.putText(result_frame, label, (startX, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        if frame_count % CLASSIFICATION_INTERVAL == 0:
            blob_classification = cv2.dnn.blobFromImage(frame, 1, (224, 224), (104, 117, 123))
            classification_net.setInput(blob_classification)
            preds = classification_net.forward()

            idx = np.argmax(preds[0])
            current_classification = classification_classes[idx]
            classification_confidence = preds[0][idx] * 100

        overlay = result_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, result_frame, 0.4, 0, result_frame)

        cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(result_frame, f"Scene: {current_classification}", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(result_frame, f"Confidence: {classification_confidence:.1f}%", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        objects_text = f"Objects: {len(detected_objects)}"
        if detected_objects:
            unique_objects = list(set(detected_objects))
            objects_text += f" ({', '.join(unique_objects[:3])}{'...' if len(unique_objects) > 3 else ''})"

        cv2.rectangle(overlay, (0, h - 40), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, result_frame, 0.4, 0, result_frame)

        cv2.putText(result_frame, objects_text, (10, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(result_frame, "Press 'q' to quit | 's' to save", (w - 350, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Real-Time Detection & Classification", result_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            saved_count += 1
            filename = f"captured_frame_{saved_count}.png"
            cv2.imwrite(filename, result_frame)
            print(f"Frame saved as: {filename}")

finally:
    print("\nReleasing resources...")
    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("SESSION STATISTICS")
    print("=" * 60)
    print(f"Total frames processed: {frame_count}")
    print(f"Average FPS: {fps:.1f}")
    print(f"Frames saved: {saved_count}")
    print("=" * 60)
