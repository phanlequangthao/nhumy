import streamlit as st
import av
import cv2
import numpy as np
import pandas as pd
import os
import pickle
import face_recognition
from datetime import datetime
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Union

# Cấu hình trang
st.set_page_config(page_title="SmartClass AI", layout="wide")
st.title("🎓 SmartClass AI – Trợ Lý Lớp Học Thông Minh v2.0")

# Các hằng số và thiết lập
FACE_DISTANCE_THRESHOLD = 0.55  # Ngưỡng nhận diện khuôn mặt (giá trị càng thấp càng chặt chẽ)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Khởi tạo session state
if 'running' not in st.session_state:
    st.session_state.running = False
if 'detection_threshold' not in st.session_state:
    st.session_state.detection_threshold = FACE_DISTANCE_THRESHOLD
if 'analysis_started' not in st.session_state:
    st.session_state.analysis_started = datetime.now()
if 'statistics' not in st.session_state:
    st.session_state.statistics = {}

# Load danh sách học sinh đã huấn luyện
def load_students():
    if os.path.exists("students.pkl"):
        with open("students.pkl", "rb") as f:
            try:
                return pickle.load(f)
            except Exception as e:
                st.error(f"Lỗi khi tải dữ liệu học sinh: {e}")
                return {}
    return {}

# Lưu thống kê đi kèm vào file CSV
def mark_attendance(name):
    df = pd.read_csv("diemdanh.csv") if os.path.exists("diemdanh.csv") else pd.DataFrame(columns=["Name", "Time"])
    if name not in df["Name"].values:
        now = datetime.now()
        time_str = now.strftime('%H:%M:%S')
        df.loc[len(df)] = [name, time_str]
        df.to_csv("diemdanh.csv", index=False)

# Điều chỉnh độ sáng của hình ảnh
def adjust_brightness(image, factor=1.0):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    hsv = np.array(hsv, dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Hàm so sánh khuôn mặt với danh sách học sinh
def compare_face_with_students(face_encoding, students, threshold=FACE_DISTANCE_THRESHOLD):
    if not face_encoding or not students:
        return None, float('inf')
    
    best_match = None
    best_distance = threshold
    
    for student in students.values():
        # Chỉ kiểm tra tối đa 3 encoding đầu tiên của học sinh để tăng tốc
        for encoding in student.face_encodings[:3]:
            if encoding is not None:
                # Tính khoảng cách Euclide cho nhanh
                face_distance = np.linalg.norm(np.array(face_encoding) - np.array(encoding))
                face_distance = face_distance / 128.0  # Chuẩn hóa
                if face_distance < best_distance:
                    best_distance = face_distance
                    best_match = student
    
    return best_match, best_distance

# Lớp xử lý WebRTC cho video
class VideoProcessor(VideoProcessorBase):
    def __init__(self, students: Dict, threshold: float = FACE_DISTANCE_THRESHOLD):
        self.students = students
        self.threshold = threshold
        self.status_dict = {}
        self.last_processed_time = time.time()
        self.process_every_n_frames = 5  # Tăng từ 2 lên 5 để tăng tốc
        self.frame_counter = 0
        # Pre-allocate face locations để tái sử dụng
        self.face_locations = []
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self.frame_counter += 1
        
        # Chỉ xử lý mỗi n frame
        if self.frame_counter % self.process_every_n_frames != 0:
            # Trả về frame trực tiếp không xử lý để tiết kiệm thời gian
            return frame
            
        # Chuyển đổi frame sang định dạng hình ảnh OpenCV
        img = frame.to_ndarray(format="bgr24")
        
        # Thời gian hiện tại
        current_time = time.time()
        
        # Resize ảnh nhỏ hơn để xử lý nhanh hơn
        h, w = img.shape[:2]
        # Giảm kích thước nhiều hơn, chỉ còn 320 pixel theo chiều rộng
        scale = 320 / w
        img_small = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            
        # Phát hiện khuôn mặt và tính encoding
        rgb_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        # Sử dụng HOG model nhanh hơn CNN (mặc định)
        face_locations = face_recognition.face_locations(rgb_small, model="hog")
        
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations, num_jitters=1)
            
            # Đảm bảo chỉ xử lý khi có face_encodings
            if len(face_encodings) > 0:
                small_to_orig_ratio = img.shape[0] / img_small.shape[0]
                
                # Xử lý từng khuôn mặt
                for i, (top, right, bottom, left) in enumerate(face_locations):
                    if i < len(face_encodings):
                        # Chuyển tọa độ về khung hình gốc
                        top = int(top * small_to_orig_ratio)
                        right = int(right * small_to_orig_ratio)
                        bottom = int(bottom * small_to_orig_ratio)
                        left = int(left * small_to_orig_ratio)
                        
                        # Tìm học sinh phù hợp nhất
                        best_match, distance = compare_face_with_students(
                            face_encodings[i], self.students, self.threshold
                        )
                        
                        # Vẽ bounding box
                        if best_match:
                            # Cập nhật trạng thái học sinh
                            status = best_match.update_status("Tap trung")  # Default là tập trung
                            
                            # Màu sắc dựa trên trạng thái
                            color = (0, 255, 0) if status == "Tap trung" else (0, 0, 255)
                            
                            # Vẽ hộp và tên (đơn giản hóa)
                            cv2.rectangle(img, (left, top), (right, bottom), color, 2)
                            cv2.putText(img, best_match.name, 
                                      (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.7, color, 2)
                            
                            # Ghi nhận điểm danh
                            mark_attendance(best_match.name)
                            
                            # Lưu thông tin trạng thái
                            self.status_dict[best_match.name] = status
                        else:
                            # Vẽ hộp đỏ cho người lạ
                            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        
        # Cập nhật trạng thái toàn cục để hiển thị
        st.session_state.statistics = self.status_dict
        
        # Thêm FPS hiển thị
        fps = 1.0 / (time.time() - self.last_processed_time) * self.process_every_n_frames
        cv2.putText(img, f"FPS: {fps:.1f}", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        self.last_processed_time = time.time()
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Sidebar cho cài đặt và thông tin hệ thống
st.sidebar.title("⚙️ Cài đặt")
detection_threshold = st.sidebar.slider(
    "Ngưỡng nhận diện khuôn mặt", 
    min_value=0.3, 
    max_value=0.9, 
    value=st.session_state.detection_threshold,
    step=0.05,
    help="Giá trị thấp hơn cho phép nhận diện dễ dàng hơn nhưng có thể gây nhầm lẫn. Giá trị cao hơn yêu cầu độ chính xác cao hơn."
)

# Cập nhật ngưỡng phát hiện
if detection_threshold != st.session_state.detection_threshold:
    st.session_state.detection_threshold = detection_threshold
    FACE_DISTANCE_THRESHOLD = detection_threshold

# Tab cho các tính năng khác nhau
tab1, tab2, tab3 = st.tabs(["📹 Giám sát lớp học", "📊 Thống kê", "⚡ Huấn luyện"])

# Tab giám sát lớp học
with tab1:
    st.subheader("📹 Giám sát lớp học - WebRTC")
    students = load_students()
    
    # Hiển thị danh sách học sinh đã huấn luyện
    st.sidebar.subheader("👨‍🎓 Danh sách học sinh đã huấn luyện:")
    if students:
        for student_name, student in students.items():
            num_encodings = len(student.face_encodings)
            st.sidebar.write(f"- **{student_name}**: {num_encodings} mẫu khuôn mặt")
    else:
        st.sidebar.warning("⚠️ Chưa có học sinh nào được huấn luyện!")
    
    # Chia giao diện thành 2 cột
    col_video, col_status = st.columns([3, 1])  # Thay đổi tỉ lệ để video lớn hơn
    
    # Vùng hiển thị WebRTC video
    with col_video:
        ctx = webrtc_streamer(
            key="smartclass",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=lambda: VideoProcessor(students, FACE_DISTANCE_THRESHOLD),
            media_stream_constraints={
                "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},  # Giảm kích thước video
                "audio": False
            },
            async_processing=True,
        )
    
    # Vùng hiển thị trạng thái học sinh
    with col_status:
        status_placeholder = st.empty()
        attention_placeholder = st.empty()
        
        # Hiển thị bảng trạng thái
        if st.session_state.statistics:
            df = pd.DataFrame(list(st.session_state.statistics.items()), columns=["Tên", "Trạng thái"])
            status_placeholder.dataframe(df, hide_index=True)
            
            # Hiển thị đánh giá mức độ tập trung
            attention_stats = {}
            for name, student_obj in students.items():
                if name in st.session_state.statistics:
                    attention_stats[name] = {
                        "Điểm tập trung": int(student_obj.attention_score),
                        "% Thời gian tập trung": f"{student_obj.get_attention_percentage():.1f}%"
                    }
            
            if attention_stats:
                attention_df = pd.DataFrame.from_dict(attention_stats, orient='index')
                attention_placeholder.dataframe(attention_df, hide_index=False)
        else:
            status_placeholder.info("👆 Bấm 'START' để bắt đầu phát hiện khuôn mặt")

# Tab thống kê
with tab2:
    st.subheader("📊 Thống kê đơn giản")
    
    # Hiển thị thời gian phân tích
    analysis_duration = datetime.now() - st.session_state.analysis_started
    st.write(f"⏱️ Thời gian phân tích: {analysis_duration.total_seconds():.0f} giây")
    
    if students:
        # Hiển thị bảng thống kê đơn giản
        st.write("### Bảng thống kê tập trung")
        
        # Tạo bảng thống kê đơn giản
        stats_data = []
        for name, student in students.items():
            if not hasattr(student, 'attention_score'):
                student.attention_score = 100
            
            stats_data.append({
                "Học sinh": name, 
                "Trạng thái": student.last_status,
                "Điểm tập trung": f"{student.attention_score:.0f}/100"
            })
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, hide_index=True)
        else:
            st.info("Chưa có dữ liệu thống kê")
        
        # Nút reset thống kê
        if st.button("🔄 Đặt lại thống kê"):
            for student in students.values():
                student.reset_statistics()
            st.success("✅ Đã đặt lại thống kê cho tất cả học sinh!")
            st.session_state.analysis_started = datetime.now()
    else:
        st.warning("⚠️ Chưa có học sinh nào được huấn luyện để hiển thị thống kê")

# Tab huấn luyện
with tab3:
    st.subheader("⚡ Huấn luyện mô hình")
    st.write("""
    #### Hướng dẫn huấn luyện:
    1. Đặt video của học sinh vào thư mục `dataset/[tên học sinh]/`
    2. Chạy lệnh `python train_model.py` để huấn luyện
    3. Hoặc sử dụng form dưới đây để huấn luyện nhanh
    """)
    
    # Form huấn luyện từ file có sẵn
    with st.form("train_form"):
        student_name = st.text_input("Tên học sinh:")
        video_path = st.text_input("Đường dẫn đến video/ảnh:")
        
        submitted = st.form_submit_button("🚀 Huấn luyện")
        if submitted:
            if student_name and video_path:
                if os.path.exists(video_path):
                    st.info(f"⏳ Đang huấn luyện cho học sinh {student_name}...")
                    
                    # Import hàm train_student từ train_model.py
                    from train_model import train_student
                    try:
                        student = train_student(student_name, [video_path])
                        st.success(f"✅ Đã huấn luyện thành công cho {student_name} với {len(student.face_encodings)} mẫu khuôn mặt")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"❌ Lỗi khi huấn luyện: {str(e)}")
                else:
                    st.error(f"❌ Không tìm thấy file video/ảnh: {video_path}")
            else:
                st.error("❌ Vui lòng điền đầy đủ thông tin")
    
    # Hiển thị mẫu khuôn mặt đã huấn luyện
    st.subheader("👁️ Mẫu khuôn mặt đã huấn luyện")
    
    for name in students.keys():
        sample_dir = Path(f"dataset/{name}")
        if sample_dir.exists():
            st.write(f"#### 🧑‍🎓 {name}")
            
            # Lấy tối đa 5 mẫu khuôn mặt
            samples = list(sample_dir.glob("*.jpg"))[:5]
            if not samples:
                samples = list(sample_dir.glob("*.jpeg"))[:5]
                
            if samples:
                cols = st.columns(min(5, len(samples)))
                for i, sample_path in enumerate(samples):
                    with cols[i]:
                        sample_img = cv2.imread(str(sample_path))
                        if sample_img is not None:
                            st.image(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB), 
                                    caption=f"Mẫu {i+1}",
                                    width=100)
            else:
                st.info(f"Không tìm thấy mẫu khuôn mặt cho {name}")

st.markdown("---")
st.caption("© 2023 SmartClass AI - Trợ lý lớp học thông minh | Phiên bản 2.0 (WebRTC)")

# Thêm thông tin FPS
st.sidebar.write("#### ⚡ Thông tin hiệu suất")
st.sidebar.info("""
WebRTC cho phép xử lý video trực tiếp trong trình duyệt, giúp tăng FPS đáng kể so với phương pháp xử lý trên server.
- Xử lý mỗi frame thứ 2
- Giảm kích thước hình ảnh phân tích để tăng tốc
- Giảm thiểu việc truyền dữ liệu lên server
""")