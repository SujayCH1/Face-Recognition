import cv2
import os
import json

def capture_images(output_dir, num_images_per_person):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current directory of the script
    images_folder = os.path.join(current_dir, output_dir)  # Set the images folder path inside the current directory
    json_file = os.path.join(current_dir, 'labels_images_mapping.json')  # Path to the JSON file

    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to open the camera.")
        return

    labels_images_mapping = {}

    while True:
        label = input("Enter the label for the person (or 'q' to quit): ")
        if label.lower() == 'q':
            break

        if label.strip() == '':
            print('Label cannot be empty. Please try again.')
            continue

        person_folder = os.path.join(images_folder, label)
        if not os.path.exists(person_folder):
            os.makedirs(person_folder)

        images_taken = 0
        while images_taken < num_images_per_person:
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to capture image.")
                break

            cv2.imshow('Capture Image', frame)

            image_path = os.path.join(person_folder, f'{label}_{images_taken + 1}.jpg')
            cv2.imwrite(image_path, frame)
            print(f"Image {images_taken + 1}/{num_images_per_person} captured for {label}")

            images_taken += 1

            if cv2.waitKey(1) == ord('q'):
                break

        image_paths = [os.path.abspath(os.path.join(person_folder, image)) for image in os.listdir(person_folder)]
        labels_images_mapping[label] = [path.replace('\\\\', '\\') for path in image_paths]  # Preprocess file paths

    cap.release()
    cv2.destroyAllWindows()

    with open(json_file, 'w') as file:
        json.dump(labels_images_mapping, file)

def main():
    output_dir = 'captured_images'
    num_images_per_person = 100  # Adjust as needed
    capture_images(output_dir, num_images_per_person)

if __name__ == "__main__":
    main()
