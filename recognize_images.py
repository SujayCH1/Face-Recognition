import cv2
import os
import json
import numpy as np
from keras_facenet import FaceNet
from concurrent.futures import ThreadPoolExecutor
import time

def load_labels_images_mapping(json_file):
    with open(json_file, 'r') as file:
        labels_images_mapping = json.load(file)
    return labels_images_mapping

def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces

def load_embeddings(facenet, labels_images_mapping):
    labels_embeddings_mapping = {}
    for label, image_paths in labels_images_mapping.items():
        embeddings = []
        for image_path in image_paths:
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = detect_faces(image)
            for (x, y, w, h) in faces:
                face_region = image_rgb[y:y+h, x:x+w]
                face_resized = cv2.resize(face_region, (160, 160))
                face_normalized = face_resized / 255.0
                face_batch = np.expand_dims(face_normalized, axis=0)
                face_embedding = facenet.model.predict(face_batch)
                embeddings.append(face_embedding)
        labels_embeddings_mapping[label] = embeddings
    return labels_embeddings_mapping

def recognize_faces(frame, labels_images_mapping, labels_embeddings_mapping, facenet):
    recognized_labels = []
    for label, image_paths in labels_images_mapping.items():
        for image_path in image_paths:
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = detect_faces(frame)
            for (x, y, w, h) in faces:
                face_region = image_rgb[y:y+h, x:x+w]
                face_resized = cv2.resize(face_region, (160, 160))
                face_normalized = face_resized / 255.0
                face_batch = np.expand_dims(face_normalized, axis=0)
                face_embedding = facenet.model.predict(face_batch)
                recognized_label = "Unknown"
                min_distance = float('inf')
                for known_label, known_embeddings in labels_embeddings_mapping.items():
                    for known_embedding in known_embeddings:
                        distance = np.linalg.norm(face_embedding - known_embedding)
                        if distance < min_distance:
                            min_distance = distance
                            recognized_label = known_label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, recognized_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                recognized_labels.append(recognized_label)

    if recognized_labels:
        print("Recognized labels:", recognized_labels)

def process_frame(frame, labels_images_mapping, labels_embeddings_mapping, facenet):
    recognize_faces(frame, labels_images_mapping, labels_embeddings_mapping, facenet)
    return frame

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.join(current_dir, 'labels_images_mapping.json')

    labels_images_mapping = load_labels_images_mapping(json_file)

    facenet = FaceNet()

    labels_embeddings_mapping = load_embeddings(facenet, labels_images_mapping)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to open the camera.")
        return

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=2) as executor:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to capture image.")
                break

            current_time = time.time()
            if current_time - start_time >= 1.0:
                start_time = current_time

            future = executor.submit(process_frame, frame, labels_images_mapping, labels_embeddings_mapping, facenet)
            processed_frame = future.result()

            cv2.imshow('Recognize Faces', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
