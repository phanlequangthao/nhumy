# SmartClass AI - Trợ Lý Lớp Học Thông Minh v3.0

Ứng dụng theo dõi trạng thái học sinh trong lớp học sử dụng xử lý video trực tiếp trên trình duyệt với MediaPipe và TensorFlow.js. Giải pháp này tối ưu hiệu suất để đạt 30+ FPS.

## Tính năng chính

- **Xử lý video trực tiếp trên trình duyệt:** Sử dụng MediaPipe Face Mesh để phát hiện và phân tích khuôn mặt
- **Theo dõi trạng thái học sinh:** Phát hiện khi học sinh đang tập trung, mất tập trung, cúi xuống, hoặc nhìn ra ngoài
- **Hiệu suất cao:** Đạt 30+ FPS nhờ xử lý trực tiếp trên trình duyệt
- **Trực quan hóa:** Hiển thị trạng thái học sinh theo thời gian thực

## Cài đặt

1. Clone repository:
```
git clone <repository-url>
cd smartclass-ai
```

2. Cài đặt dependencies:
```
npm install
```

3. Khởi động server:
```
npm start
```

4. Mở trình duyệt và truy cập:
```
http://localhost:3000
```

## Cách sử dụng

1. Truy cập ứng dụng qua trình duyệt
2. Nhấn nút "Bắt đầu" để kích hoạt camera
3. Cho phép trình duyệt truy cập camera khi được hỏi
4. Ứng dụng sẽ tự động phát hiện khuôn mặt và theo dõi trạng thái học sinh
5. Thông tin trạng thái học sinh sẽ được hiển thị ở bảng bên phải

## Yêu cầu hệ thống

- Trình duyệt hiện đại hỗ trợ WebRTC (Chrome, Firefox, Edge)
- Camera web
- Node.js 12.0 trở lên

## Giải pháp kỹ thuật

Ứng dụng sử dụng:
- **MediaPipe Face Mesh**: Phát hiện và theo dõi 468 điểm trên khuôn mặt
- **TensorFlow.js**: Xử lý mô hình máy học trên trình duyệt
- **Express**: Server web đơn giản
- **JavaScript hiện đại**: Xử lý giao diện và logic ứng dụng

## Hiệu suất

- **FPS**: 30-60 FPS tùy thuộc vào phần cứng
- **CPU**: Tối ưu sử dụng CPU nhờ xử lý trực tiếp trên GPU trình duyệt
- **Độ chính xác**: Cao nhờ sử dụng mô hình Face Mesh của Google 