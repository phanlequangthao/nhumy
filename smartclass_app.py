import streamlit as st
import cv2
from model_utils import process_frame, load_students
import pandas as pd

st.set_page_config(page_title="SmartClass AI", layout="wide")
st.title("🎓 SmartClass AI – Trợ Lý Lớp Học Thông Minh")

# Khởi tạo session state
if 'running' not in st.session_state:
    st.session_state.running = False

# Hiển thị danh sách học sinh đã huấn luyện
students = load_students()
if students:
    st.sidebar.subheader("Danh sách học sinh đã huấn luyện:")
    for student_name in students.keys():
        st.sidebar.write(f"- {student_name}")
else:
    st.sidebar.warning("Chưa có học sinh nào được huấn luyện!")

# Phần giám sát lớp học
col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Bắt đầu giám sát lớp học") and not st.session_state.running:
        st.session_state.running = True
with col2:
    if st.button("⏹ Dừng") and st.session_state.running:
        st.session_state.running = False

frame_placeholder = st.empty()
table_placeholder = st.empty()

if st.session_state.running:
    cap = cv2.VideoCapture(0)
    
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            break

        processed, status = process_frame(frame)
        frame_placeholder.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), channels="RGB")

        if status:
            df = pd.DataFrame(list(status.items()), columns=["Tên", "Trạng thái"])
            table_placeholder.dataframe(df)

    cap.release()
