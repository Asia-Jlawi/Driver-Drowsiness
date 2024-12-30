import cv2
import mediapipe as mp
import os

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Define the face and eye landmarks
left_eye_landmarks = [33, 133, 159, 145, 158, 153, 144, 163, 7]
right_eye_landmarks = [362, 382, 380, 374, 381, 373, 385, 386, 263]

# Function to adjust brightness
def adjust_brightness(image, brightness=30):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, brightness)
    v = cv2.min(v, 255)  # Ensure values stay within [0, 255]
    final_hsv = cv2.merge((h, s, v))
    bright_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return bright_image

# Function to process a single image and crop the eye region
def process_and_crop_eye(image_path, output_folder, padding=20, brightness=30):
    img = cv2.imread(image_path)

    # Adjust brightness of the image
    bright_img = adjust_brightness(img, brightness=brightness)
    rgb_img = cv2.cvtColor(bright_img, cv2.COLOR_BGR2RGB)

    # Perform face mesh detection
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(rgb_img)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract eye region coordinates for both eyes
                all_eye_landmarks = [face_landmarks.landmark[i] for i in left_eye_landmarks + right_eye_landmarks]
                x_coords = [int(lm.x * img.shape[1]) for lm in all_eye_landmarks]
                y_coords = [int(lm.y * img.shape[0]) for lm in all_eye_landmarks]
                
                # Calculate bounding box
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                # Add padding to the bounding box, ensuring it stays within image bounds
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(img.shape[1], x_max + padding)
                y_max = min(img.shape[0], y_max + padding)

                # Crop the region containing both eyes with padding
                eye_crop = img[y_min:y_max, x_min:x_max]

                # Ensure the output folder exists
                os.makedirs(output_folder, exist_ok=True)

                # Save the cropped eye region
                save_path = os.path.join(output_folder, os.path.basename(image_path))
                cv2.imwrite(save_path, eye_crop)
                print(f"Cropped eye region saved to: {save_path}")
        else:
            print(f"No eyes detected in {image_path}. Skipping this image.")

# Define the folders
base_folder = './Driver Drowsiness Dataset (DDD) copy'
drowsy_folder = os.path.join(base_folder, 'Drowsy')
non_drowsy_folder = os.path.join(base_folder, 'Non Drowsy')

# Process and save cropped images for Drowsy and Non Drowsy categories
for img_name in os.listdir(drowsy_folder):
    image_path = os.path.join(drowsy_folder, img_name)
    process_and_crop_eye(image_path, "processed_drowsy_images", brightness=30)

for img_name in os.listdir(non_drowsy_folder):
    image_path = os.path.join(non_drowsy_folder, img_name)
    process_and_crop_eye(image_path, "processed_non_drowsy_images", brightness=30)

print("Processing and cropping completed for all images.")
