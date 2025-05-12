const express = require('express');
const path = require('path');
const app = express();
const port = 3000;

// Phục vụ tất cả file tĩnh từ thư mục hiện tại
app.use(express.static(__dirname));

// Route chính trả về file index.html
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// API endpoint để lấy danh sách học sinh (mock data)
app.get('/api/students', (req, res) => {
  const students = [
    { id: 1, name: "Nam", attentionScore: 100, status: "Tap trung" },
    { id: 2, name: "Linh", attentionScore: 100, status: "Tap trung" },
    { id: 3, name: "My", attentionScore: 100, status: "Tap trung" },
    { id: 4, name: "Thao", attentionScore: 100, status: "Tap trung" }
  ];
  
  res.json(students);
});

// Khởi động server
app.listen(port, () => {
  console.log(`SmartClass AI server đang chạy tại http://localhost:${port}`);
}); 