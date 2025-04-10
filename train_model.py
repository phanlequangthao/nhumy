import cv2
import os
import pickle
import mediapipe as mp
from student import Student

# Khởi tạo MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Load hoặc tạo mới danh sách học sinh
def load_students():
    if os.path.exists("students.pkl"):
        with open("students.pkl", "rb") as f:
            return pickle.load(f)
    return {}

# Lưu danh sách học sinh
def save_students(students):
    with open("students.pkl", "wb") as f:
        pickle.dump(students, f)

# Huấn luyện mô hình cho một học sinh
def train_student(name, video_paths):
    students = load_students()
    if name not in students:
        students[name] = Student(name)
    
    student = students[name]
    
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)
            
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    student.face_encodings.append(bbox)
        
        cap.release()
    
    save_students(students)
    return student

if __name__ == "__main__":
    # Ví dụ huấn luyện
    video_paths = [
        "dataset/thao/v1.mp4",
        "dataset/thao/v2.mp4",
        "dataset/thao/v3.mp4",
        "dataset/thao/v4.mp4"
    ]
    train_student("thao", video_paths)