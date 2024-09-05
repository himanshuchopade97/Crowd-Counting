import os
import cv2
import numpy as np
from deepface import DeepFace
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Define the path to the LFW dataset
dataset_path = 'lfw_dataset/lfw-deepfunneled/lfw-deepfunneled'

def load_lfw_dataset(dataset_path):
    embeddings = []
    labels = []
    label_names = []
    label_map = {}
    
    # Loop through each person directory in LFW dataset
    for person_dir in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_dir)
        if os.path.isdir(person_path):
            label_map[len(label_names)] = person_dir
            label_names.append(person_dir)
            
            # Loop through each image in the person's directory
            for image_file in os.listdir(person_path):
                image_path = os.path.join(person_path, image_file)
                
                # Read and preprocess the image
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Cannot open or read the file: {image_path}")
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Extract embedding using VGG-Face model
                try:
                    embedding = DeepFace.represent(img_rgb, model_name='VGG-Face', enforce_detection=False)[0]['embedding']
                    embeddings.append(embedding)
                    labels.append(len(label_names) - 1)
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")
    
    return np.array(embeddings), np.array(labels), label_names

# Load the LFW dataset
embeddings, labels, label_names = load_lfw_dataset(dataset_path)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# Train a classifier (e.g., SVM) using the embeddings
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Predict on the test set and evaluate the accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')

def detect_faces(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open or read the file: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Use extract_faces to detect faces
    detected_faces = DeepFace.extract_faces(img_rgb, detector_backend='opencv')  # or another backend
    return detected_faces

def predict_faces(image_path):
    detected_faces = detect_faces(image_path)
    predictions = []
    
    for face in detected_faces:
        # Extract the face image
        face_img = face['face']
        
        # Convert face image to uint8 if it's not already
        if face_img.dtype != np.uint8:
            face_img = np.clip(face_img, 0, 255).astype(np.uint8)
        
        # Convert the face image to RGB
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Extract embedding for each detected face using VGG-Face
        embedding = DeepFace.represent(face_img_rgb, model_name='VGG-Face', enforce_detection=False)[0]['embedding']
        
        # Predict the label for the embedding
        pred_label = clf.predict([embedding])[0]
        predictions.append(pred_label)
    
    return detected_faces, predictions
      

    
def display_image_with_predictions(image_path, detected_faces, predictions):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open or read the file: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    
    for i, face in enumerate(detected_faces):
        # Extract the facial area from the face dictionary
        facial_area = face.get('facial_area', {})
        if 'x' in facial_area and 'y' in facial_area and 'w' in facial_area and 'h' in facial_area:
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            
            # Draw label
            label = label_names[predictions[i]]
            plt.text(x, y - 10, label, color='red', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
    
    plt.axis('off')
    plt.show()

# Path to your group image
group_image_path = 'checkingimage3.jpg'

# Predict and display results
detected_faces, predictions = predict_faces(group_image_path)
display_image_with_predictions(group_image_path, detected_faces, predictions)
