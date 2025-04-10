from datetime import datetime
from collections import deque

class Student:
    def __init__(self, name):
        self.name = name
        self.face_encodings = []
        self.look_down_frames = deque(maxlen=30)  # 30 frames = 1 giây với 30fps
        self.look_side_frames = deque(maxlen=15)  # 15 frames = 0.5 giây
        self.last_status = "Tap trung"
        self.status_change_time = datetime.now()

    def update_status(self, new_status):
        current_time = datetime.now()
        
        # Xử lý trạng thái cúi xuống
        if new_status == "Cui xuong":
            self.look_down_frames.append(current_time)
            if len(self.look_down_frames) == 30:  # Đủ 30 frames
                duration = (current_time - self.look_down_frames[0]).total_seconds()
                if duration >= 3.0:  # Cúi xuống ít nhất 3 giây
                    self.last_status = "Cui xuong"
                    self.status_change_time = current_time
        # Xử lý trạng thái nhìn ra ngoài
        elif new_status == "Nhin ra ngoai":
            self.look_side_frames.append(current_time)
            if len(self.look_side_frames) == 15:  # Đủ 15 frames
                duration = (current_time - self.look_side_frames[0]).total_seconds()
                if duration >= 1.5:  # Nhìn ra ngoài ít nhất 1.5 giây
                    self.last_status = "Nhin ra ngoai"
                    self.status_change_time = current_time
        # Xử lý các trạng thái khác
        else:
            self.look_down_frames.clear()
            self.look_side_frames.clear()
            self.last_status = new_status
            self.status_change_time = current_time

        return self.last_status 