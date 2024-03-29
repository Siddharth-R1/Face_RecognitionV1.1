import tkinter as tk
from tkinter import simpledialog
import cv2
import os
import pandas as pd
from datetime import datetime
import numpy as np
import face_recognition

# Initialize an empty DataFrame to store face records
face_records = pd.DataFrame(columns=['Name', 'Action', 'Timestamp', 'Images Folder'])
known_face_encodings = []
known_face_names = []

def save_to_excel():
    """Saves the face records DataFrame to an Excel file."""
    face_records.to_excel('face_records.xlsx', index=False, engine='openpyxl')

def update_known_faces():
    """Loads face images, encodes them, and updates known faces."""
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    base_directory = "faces"
    for name in os.listdir(base_directory):
        person_directory = os.path.join(base_directory, name)
        if os.path.isdir(person_directory):
            for image_file in os.listdir(person_directory):
                image_path = os.path.join(person_directory, image_file)
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)
                if face_encodings:
                    known_face_encodings.append(face_encodings[0])
                    known_face_names.append(name)
    print("Known faces updated.")

def capture_and_save_images(name):
    """Captures 500 images of the user's face, shows a live feed with user info, and saves them."""
    base_directory = "faces"
    person_directory = os.path.join(base_directory, name)
    os.makedirs(person_directory, exist_ok=True)
    
    video_capture = cv2.VideoCapture(0)
    
    count = 0
    while count < 500:  # Ensures 500 images are captured
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame.")
            break
        
        face_locations = face_recognition.face_locations(frame)
        for top, right, bottom, left in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"{name}: {count+1}/500", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow('Capture Faces', frame)
        
        img_path = os.path.join(person_directory, f"{name}_{count}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
        cv2.imwrite(img_path, frame)
        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    update_known_faces()
    print(f"Completed capturing 500 images for {name}.")

def set_add_face_mode():
    """Initiates the process to add a face."""
    name = simpledialog.askstring("Input", "Enter your name", parent=root)
    if name:
        capture_and_save_images(name)

def recognize_face():
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Face recognition logic here
        cv2.imshow('Recognize Face', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# GUI setup
root = tk.Tk()
root.title("Face Recognition Database")
root.geometry("300x150")

add_face_button = tk.Button(root, text="Add Face", command=set_add_face_mode)
add_face_button.pack(pady=5)

recognize_face_button = tk.Button(root, text="Recognize Face", command=recognize_face)
recognize_face_button.pack(pady=5)

root.mainloop()
