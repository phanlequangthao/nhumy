import os
from train_model import train_student

# Xóa file students.pkl nếu tồn tại
if os.path.exists("students.pkl"):
    os.remove("students.pkl")
    print("Đã xóa file students.pkl cũ")

# Huấn luyện lại mô hình
video_paths = [
    "dataset/thao/v1.mp4",
    "dataset/thao/v2.mp4",
    "dataset/thao/v3.mp4",
    "dataset/thao/v4.mp4"
]
train_student("thao", video_paths)
print("Đã huấn luyện lại mô hình thành công") 