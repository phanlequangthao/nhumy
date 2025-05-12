from datetime import datetime
from collections import deque

class Student:
    def __init__(self, name):
        self.name = name
        self.face_encodings = []
        self.look_down_frames = deque(maxlen=10)  # Giảm xuống 10 frames
        self.look_side_frames = deque(maxlen=5)  # Giảm xuống 5 frames
        self.last_status = "Tap trung"
        self.status_change_time = datetime.now()
        
        # Đảm bảo luôn có thuộc tính này khi tạo mới đối tượng
        self.attention_score = 100
        
        # Thống kê tối giản
        self.status_statistics = {
            "Tap trung": 0,
            "Cui xuong": 0,
            "Nhin ra ngoai": 0,
            "Nham mat": 0
        }
        self.last_status_update = datetime.now()

    def update_status(self, new_status):
        current_time = datetime.now()
        
        # Cập nhật thời gian với tính toán đơn giản hơn
        time_diff = (current_time - self.last_status_update).total_seconds()
        if time_diff < 5:
            self.status_statistics[self.last_status] += time_diff
            
            # Đảm bảo thuộc tính attention_score tồn tại
            if not hasattr(self, 'attention_score'):
                self.attention_score = 100
                
            # Cập nhật điểm tập trung với logic đơn giản
            if self.last_status == "Tap trung":
                self.attention_score = min(100, self.attention_score + 1)
            else:
                self.attention_score = max(0, self.attention_score - 1)
            
        self.last_status_update = current_time
        
        # Xử lý trạng thái đơn giản hơn
        if new_status != "Tap trung":
            self.last_status = new_status
        else:
            self.last_status = "Tap trung"

        return self.last_status
    
    def get_attention_percentage(self):
        """Trả về phần trăm tập trung của học sinh."""
        if not self.status_statistics:
            return 100
        
        total_time = sum(self.status_statistics.values())
        if total_time == 0:
            return 100
        
        focus_time = self.status_statistics.get("Tap trung", 0)
        return (focus_time / total_time) * 100
    
    def get_status_summary(self):
        
        """Trả về bảng tóm tắt trạng thái của học sinh."""
        # Đảm bảo thuộc tính attention_score tồn tại
        if not hasattr(self, 'attention_score'):
            self.attention_score = 100
            
        return {
            "name": self.name,
            "current_status": self.last_status,
            "attention_score": self.attention_score,
            "attention_percentage": self.get_attention_percentage(),
            "statistics": self.status_statistics
        }
    
    def reset_statistics(self):
        """Đặt lại thống kê."""
        self.status_statistics = {key: 0 for key in self.status_statistics}
        self.attention_score = 100
        self.last_status_update = datetime.now() 