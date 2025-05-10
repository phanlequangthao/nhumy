import cv2
import os
import numpy as np
from datetime import datetime
import pandas as pd
import mediapipe as mp
import pickle
from student import Student
import math
import face_recognition

# Khởi tạo MediaPipe face detection và face mesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=10, refine_landmarks=True)

# Các tham số cho việc đánh giá trạng thái
LOOK_DOWN_THRESHOLD = 0.22  # Ngưỡng cho việc cúi xuống (khoảng cách dọc mặt)
LOOK_SIDE_THRESHOLD = 10.0  # Ngưỡng cho việc nhìn sang ngang (độ)
LOOK_UP_THRESHOLD = 0.03    # Ngưỡng cho việc ngửa mặt lên
EYE_CLOSED_THRESHOLD = 0.03  # Ngưỡng cho việc nhắm mắt
ROTATION_THRESHOLD = 15.0    # Ngưỡng góc quay (độ)
FACE_DISTANCE_THRESHOLD = 0.45  # Ngưỡng nhận diện khuôn mặt (55% độ giống)

# Hàm tính góc quay của khuôn mặt
def calculate_rotation_angle(face_landmarks):
    # Lấy các điểm mốc quan trọng
    nose_tip = face_landmarks.landmark[1]
    left_eye = face_landmarks.landmark[33]
    right_eye = face_landmarks.landmark[263]
    
    # Tính vector từ mắt trái đến mắt phải (vector ngang)
    eye_vector = np.array([right_eye.x - left_eye.x, right_eye.y - left_eye.y])
    
    # Tính vector từ điểm giữa hai mắt đến mũi (vector dọc)
    eye_center = np.array([(left_eye.x + right_eye.x) / 2, (left_eye.y + right_eye.y) / 2])
    nose_vector = np.array([nose_tip.x - eye_center[0], nose_tip.y - eye_center[1]])
    
    # Tính góc giữa vector mũi và vector ngang
    dot_product = np.dot(eye_vector, nose_vector)
    eye_magnitude = np.linalg.norm(eye_vector)
    nose_magnitude = np.linalg.norm(nose_vector)
    
    if eye_magnitude == 0 or nose_magnitude == 0:
        return 0.0
    
    cos_angle = dot_product / (eye_magnitude * nose_magnitude)
    angle = math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))
    
    # Xác định hướng quay (trái/phải) dựa vào vị trí tương đối của mũi
    if nose_tip.x < eye_center[0]:
        angle = -angle  # Quay sang trái
    
    return angle

# Load danh sách học sinh đã huấn luyện
def load_students():
    if os.path.exists("students.pkl"):
        with open("students.pkl", "rb") as f:
            return pickle.load(f)
    return {}

# Add head pose estimation functions
def get_head_pose(landmarks, image_shape):
    image_points = np.array([
        landmarks[1],    # Nose tip
        landmarks[152],  # Chin
        landmarks[263],  # Right eye right corner
        landmarks[33],   # Left eye left corner
        landmarks[287],  # Right mouth corner
        landmarks[57],   # Left mouth corner
    ], dtype="double")
    
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -63.6, -12.5),         # Chin
        (43.3, 32.7, -26.0),         # Right eye right corner
        (-43.3, 32.7, -26.0),        # Left eye left corner
        (28.9, -28.9, -24.1),        # Right mouth corner
        (-28.9, -28.9, -24.1)        # Left mouth corner
    ])
    
    size = image_shape
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    
    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    pitch = np.arctan2(-rotation_matrix[2, 0], sy)
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    
    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)
    roll = np.degrees(roll)
    
    return pitch, yaw, roll

def classify_head_pose(pitch, yaw):
    if pitch < -20:
        return "CÚI MẶT"
    elif pitch > 20:
        return "NGẢ RA SAU"
    elif yaw < -25:
        return "NHÌN SANG TRÁI"
    elif yaw > 25:
        return "NHÌN SANG PHẢI"
    else:
        return "NHÌN THẲNG"

# Hàm đánh giá trạng thái chú ý
def check_attention_status(face_landmarks, frame_shape):
    # Convert landmarks to list of points
    h, w = frame_shape[:2]
    landmarks = [(int(p.x * w), int(p.y * h)) for p in face_landmarks.landmark]
    
    # Get head pose angles
    pitch, yaw, roll = get_head_pose(landmarks, frame_shape)
    pose_status = classify_head_pose(pitch, yaw)
    
    # Existing eye closure check
    left_eye_top = face_landmarks.landmark[159]
    left_eye_bottom = face_landmarks.landmark[145]
    right_eye_top = face_landmarks.landmark[386]
    right_eye_bottom = face_landmarks.landmark[374]
    
    left_eye_height = abs(left_eye_top.y - left_eye_bottom.y)
    right_eye_height = abs(right_eye_top.y - right_eye_bottom.y)
    avg_eye_height = (left_eye_height + right_eye_height) / 2
    
    if avg_eye_height < EYE_CLOSED_THRESHOLD:
        return "Nham mat"
    
    # Map head pose status to existing status categories
    if pose_status == "CÚI MẶT":
        return "Cui xuong"
    elif pose_status in ["NHÌN SANG TRÁI", "NHÌN SANG PHẢI"]:
        return "Nhin ra ngoai"
    else:
        return "Tap trung"

# Điểm danh vào file CSV
def mark_attendance(name):
    df = pd.read_csv("diemdanh.csv") if os.path.exists("diemdanh.csv") else pd.DataFrame(columns=["Name", "Time"])
    if name not in df["Name"].values:
        now = datetime.now()
        time_str = now.strftime('%H:%M:%S')
        df.loc[len(df)] = [name, time_str]
        df.to_csv("diemdanh.csv", index=False)

# Hàm tính IoU (Intersection over Union) giữa hai bounding box
def calculate_iou(box1, box2):
    x1 = max(box1.xmin, box2.xmin)
    y1 = max(box1.ymin, box2.ymin)
    x2 = min(box1.xmin + box1.width, box2.xmin + box2.width)
    y2 = min(box1.ymin + box1.height, box2.ymin + box2.height)
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = box1.width * box1.height
    box2_area = box2.width * box2.height
    
    return intersection / float(box1_area + box2_area - intersection)

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

def compare_faces(face_encoding1, face_encoding2):
    if face_encoding1 is None or face_encoding2 is None:
        return float('inf')
    return face_recognition.face_distance([face_encoding1], face_encoding2)[0]

# Xử lý 1 frame hình ảnh
def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    status_dict = {}
    students = load_students()

    if results.detections:
        # Sắp xếp các detection theo vị trí từ trái sang phải
        detections = sorted(results.detections, 
                          key=lambda x: x.location_data.relative_bounding_box.xmin)
        
        for detection in detections:
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
                
            # Lấy face encoding của khuôn mặt hiện tại
            current_face_encoding = get_face_encoding(face_crop)
            if current_face_encoding is None:
                continue

            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            mesh_results = face_mesh.process(face_rgb)
            
            # Tìm học sinh phù hợp nhất
            best_match = None
            best_distance = FACE_DISTANCE_THRESHOLD
            
            for student in students.values():
                for encoding in student.face_encodings:
                    distance = compare_faces(current_face_encoding, encoding)
                    if distance < best_distance:
                        best_distance = distance
                        best_match = student
            
            if best_match:
                status = "Unknown"
                if mesh_results.multi_face_landmarks:
                    for lm in mesh_results.multi_face_landmarks:
                        raw_status = check_attention_status(lm, frame.shape)
                        status = best_match.update_status(raw_status)
                
                # Chỉ hiển thị màu đỏ cho các trạng thái không tập trung
                color = (0, 255, 0) if status == "Tap trung" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{best_match.name} - {status}", (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                mark_attendance(best_match.name)
                status_dict[best_match.name] = status
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "NGUOI LA", (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                if not os.path.exists("canhbao"):
                    os.mkdir("canhbao")
                filename = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"canhbao/nguoi_la_{filename}.jpg", face_crop)

    return frame, status_dict
