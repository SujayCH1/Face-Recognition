import cv2
import os
import face_recognition
import numpy as np

def capture_images(output_dir, num_images_per_person):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_capture = cv2.VideoCapture(0)

    while True:
        person_name = input("Enter the name of the person (or 'q' to quit): ")
        if person_name == 'q':
            break

        person_dir = os.path.join(output_dir, person_name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)

        for i in range(num_images_per_person):
            print(f"Capturing image {i+1}/{num_images_per_person} for {person_name}")

            ret, frame = video_capture.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.imshow('Capture Faces', frame)

            img_name = f"{person_dir}/image_{i}.jpg"
            cv2.imwrite(img_name, frame)

            cv2.waitKey(500)

    video_capture.release()
    cv2.destroyAllWindows()

def encode_faces(input_dir):
    known_face_encodings = []
    known_face_names = []

    for person_name in os.listdir(input_dir):
        person_dir = os.path.join(input_dir, person_name)
        if os.path.isdir(person_dir):
            for filename in os.listdir(person_dir):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image = face_recognition.load_image_file(os.path.join(person_dir, filename))
                    face_encodings = face_recognition.face_encodings(image)
                    
                    if len(face_encodings) > 0:
                        encoding = face_encodings[0] 
                        known_face_encodings.append(encoding)
                        known_face_names.append(person_name)
                    else:
                        print(f"No face detected in {filename}")

    return known_face_encodings, known_face_names

import os

def main():
    output_dir = "captured_images"
    num_images_per_person = int(input("Enter the number of images to capture for each person: "))
    capture_images(output_dir, num_images_per_person)
    print("Image capture process completed.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_dir = os.path.join(script_dir, "captured_images")

    face_encodings_file = os.path.join(script_dir, "face_encodings.npy")
    face_names_file = os.path.join(script_dir, "face_names.txt")

    known_face_encodings, known_face_names = encode_faces(input_dir)

    with open(face_encodings_file, 'wb') as f:
        np.save(f, known_face_encodings)

    with open(face_names_file, 'w') as f:
        for name in known_face_names:
            f.write("%s\n" % name)

    print("Face encodings and names saved.")

if __name__ == "__main__":
    main()
