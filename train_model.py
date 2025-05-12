import cv2
import os
import pickle
import mediapipe as mp
import face_recognition
from student import Student

# Khởi tạo MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

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

def get_face_encoding(face_image):
    # Chuyển đổi ảnh sang RGB
    rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    # Tìm khuôn mặt và tính encoding
    face_locations = face_recognition.face_locations(rgb_image)
    if face_locations:
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        if face_encodings:
            return face_encodings[0]
    return None

# Huấn luyện mô hình cho một học sinh
def train_student(name, video_paths):
    students = load_students()
    if name not in students:
        students[name] = Student(name)
    
    student = students[name]
    print(f"Đang huấn luyện cho học sinh: {name}")
    
    for video_path in video_paths:
        print(f"Đang xử lý video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Chỉ xử lý mỗi 5 frame để tăng tốc độ và đa dạng dữ liệu
            frame_count += 1
            if frame_count % 5 != 0:
                continue
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)
            
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x1 = int(bbox.xmin * w)
                    y1 = int(bbox.ymin * h)
                    x2 = int((bbox.xmin + bbox.width) * w)
                    y2 = int((bbox.ymin + bbox.height) * h)
                    
                    # Đảm bảo tọa độ không âm
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue
                    
                    # Lấy face encoding
                    face_encoding = get_face_encoding(face_crop)
                    if face_encoding is not None:
                        student.face_encodings.append(face_encoding)
                        print(f"Đã thêm encoding cho {name}")
        
        cap.release()
    
    print(f"Hoàn thành huấn luyện cho {name} với {len(student.face_encodings)} mẫu")
    save_students(students)
    return student

if __name__ == "__main__":
    nam_video_paths = [
        "dataset/nam/nam.mp4"
    ]
    train_student("nam", nam_video_paths)

    linh_video_paths = [
        "dataset/gialinh/glinh.mp4"
    ]
    train_student("linh", linh_video_paths)

    # Huấn luyện cho my
    my_video_paths = [
        "dataset/my/v1.mp4"
    ]
    train_student("my", my_video_paths)

    # Huấn luyện cho thao
    thao_video_paths = [
        "dataset/thao/v1.mp4",
        "dataset/thao/v2.mp4",
        "dataset/thao/v3.mp4",
        "dataset/thao/v4.mp4"
    ]
    train_student("thao", thao_video_paths)