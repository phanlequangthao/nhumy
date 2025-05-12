// Các biến toàn cục
let videoElement = document.getElementById('input_video');
let canvasElement = document.getElementById('output_canvas');
let canvasCtx = canvasElement.getContext('2d');
let startButton = document.getElementById('startButton');
let stopButton = document.getElementById('stopButton');
let fpsElement = document.getElementById('fps');
let thresholdSlider = document.getElementById('detectionThreshold');
let thresholdValue = document.getElementById('thresholdValue');

// Biến theo dõi FPS
let lastFrameTime = 0;
let frameCount = 0;
let fps = 0;

// Ngưỡng nhận diện khuôn mặt
let FACE_DETECTION_THRESHOLD = 0.5;

// Các điểm landmark quan trọng trên khuôn mặt
const EYE_LEFT_OUTER = 33;
const EYE_LEFT_INNER = 133;
const EYE_RIGHT_OUTER = 263;
const EYE_RIGHT_INNER = 362;
const NOSE_TIP = 1;
const MOUTH_TOP = 13;
const MOUTH_BOTTOM = 14;
const FACE_TOP = 10;
const FACE_BOTTOM = 152;

// Danh sách học sinh đã biết (sẽ được thay thế bằng dữ liệu từ server)
const knownStudents = [
    { id: 1, name: "Nam", faceEmbedding: null, attentionScore: 100, status: "Tap trung" },
    { id: 2, name: "Linh", faceEmbedding: null, attentionScore: 100, status: "Tap trung" },
    { id: 3, name: "My", faceEmbedding: null, attentionScore: 100, status: "Tap trung" },
    { id: 4, name: "Thao", faceEmbedding: null, attentionScore: 100, status: "Tap trung" }
];

// Khởi tạo Face Mesh
const faceMesh = new FaceMesh({
    locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4/${file}`;
    }
});

// Cấu hình Face Mesh
faceMesh.setOptions({
    maxNumFaces: 4,
    refineLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});

// Khởi tạo Face Detection
const faceDetection = new FaceDetection({
    locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection@0.4/${file}`;
    }
});

// Cấu hình Face Detection
faceDetection.setOptions({
    modelSelection: 0,
    minDetectionConfidence: 0.5
});

// Hàm tính góc ngẩng/cúi đầu dựa trên landmarks
function calculateHeadPose(landmarks) {
    const nose = landmarks[NOSE_TIP];
    const topFace = landmarks[FACE_TOP];
    const bottomFace = landmarks[FACE_BOTTOM];
    
    // Tính góc dọc (pitch) - góc ngẩng/cúi đầu
    const pitch = Math.atan2(nose.y - topFace.y, nose.z - topFace.z);
    const pitchDegrees = pitch * (180 / Math.PI);
    
    return {
        pitch: pitchDegrees
    };
}

// Hàm tính trạng thái mắt mở/nhắm
function calculateEyeState(landmarks) {
    const leftEyeTop = landmarks[159]; // Mí trên mắt trái
    const leftEyeBottom = landmarks[145]; // Mí dưới mắt trái
    const rightEyeTop = landmarks[386]; // Mí trên mắt phải
    const rightEyeBottom = landmarks[374]; // Mí dưới mắt phải
    
    // Tính khoảng cách giữa mí trên và mí dưới
    const leftEyeDistance = Math.sqrt(
        Math.pow(leftEyeTop.x - leftEyeBottom.x, 2) +
        Math.pow(leftEyeTop.y - leftEyeBottom.y, 2) +
        Math.pow(leftEyeTop.z - leftEyeBottom.z, 2)
    );
    
    const rightEyeDistance = Math.sqrt(
        Math.pow(rightEyeTop.x - rightEyeBottom.x, 2) +
        Math.pow(rightEyeTop.y - rightEyeBottom.y, 2) +
        Math.pow(rightEyeTop.z - rightEyeBottom.z, 2)
    );
    
    // Định nghĩa ngưỡng để xác định mắt mở hay nhắm
    const EYE_OPEN_THRESHOLD = 0.018;
    
    return {
        leftEyeOpen: leftEyeDistance > EYE_OPEN_THRESHOLD,
        rightEyeOpen: rightEyeDistance > EYE_OPEN_THRESHOLD,
        eyesOpen: leftEyeDistance > EYE_OPEN_THRESHOLD && rightEyeDistance > EYE_OPEN_THRESHOLD
    };
}

// Hàm tính hướng nhìn của khuôn mặt
function calculateGazeDirection(landmarks) {
    const leftEyeOuter = landmarks[EYE_LEFT_OUTER];
    const leftEyeInner = landmarks[EYE_LEFT_INNER];
    const rightEyeOuter = landmarks[EYE_RIGHT_OUTER];
    const rightEyeInner = landmarks[EYE_RIGHT_INNER];
    const noseTip = landmarks[NOSE_TIP];
    
    // Vector từ sống mũi đến giữa hai mắt
    const midEyesX = (leftEyeInner.x + rightEyeInner.x) / 2;
    const midEyesY = (leftEyeInner.y + rightEyeInner.y) / 2;
    const midEyesZ = (leftEyeInner.z + rightEyeInner.z) / 2;
    
    // Tính vector hướng nhìn
    const gazeVectorX = midEyesX - noseTip.x;
    const gazeVectorY = midEyesY - noseTip.y;
    const gazeVectorZ = midEyesZ - noseTip.z;
    
    // Tính góc nhìn theo chiều ngang (yaw)
    const yaw = Math.atan2(gazeVectorX, gazeVectorZ);
    const yawDegrees = yaw * (180 / Math.PI);
    
    // Xác định hướng nhìn
    if (Math.abs(yawDegrees) > 20) {
        return yawDegrees > 0 ? "Nhin sang phai" : "Nhin sang trai";
    } else {
        return "Nhin thang";
    }
}

// Hàm xác định trạng thái học sinh dựa trên các thông số của khuôn mặt
function determineStudentStatus(headPose, eyeState, gazeDirection) {
    // Nếu cúi đầu quá 20 độ
    if (headPose.pitch < -20) {
        return "Cui xuong";
    }
    
    // Nếu mắt nhắm
    if (!eyeState.eyesOpen) {
        return "Nham mat";
    }
    
    // Nếu đang nhìn ra ngoài
    if (gazeDirection !== "Nhin thang") {
        return "Nhin ra ngoai";
    }
    
    // Mặc định là tập trung
    return "Tap trung";
}

// Xử lý kết quả từ Face Mesh
faceMesh.onResults((results) => {
    // Đo FPS
    const now = performance.now();
    frameCount++;
    
    if (now - lastFrameTime >= 1000) {
        fps = frameCount;
        frameCount = 0;
        lastFrameTime = now;
        fpsElement.textContent = `FPS: ${fps}`;
    }
    
    // Vẽ hình ảnh lên canvas
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
    
    // Xử lý từng khuôn mặt phát hiện được
    if (results.multiFaceLandmarks) {
        for (let i = 0; i < results.multiFaceLandmarks.length; i++) {
            const landmarks = results.multiFaceLandmarks[i];
            
            // Tính các thông số của khuôn mặt
            const headPose = calculateHeadPose(landmarks);
            const eyeState = calculateEyeState(landmarks);
            const gazeDirection = calculateGazeDirection(landmarks);
            
            // Xác định trạng thái học sinh
            const status = determineStudentStatus(headPose, eyeState, gazeDirection);
            
            // Giả lập cập nhật trạng thái học sinh (trong thực tế sẽ kết nối với backend)
            if (i < knownStudents.length) {
                const student = knownStudents[i];
                student.status = status;
                
                // Cập nhật điểm tập trung
                if (status === "Tap trung") {
                    student.attentionScore = Math.min(100, student.attentionScore + 0.1);
                } else {
                    student.attentionScore = Math.max(0, student.attentionScore - 0.3);
                }
            }
            
            // Vẽ các đường nối landmarks
            canvasCtx.strokeStyle = status === "Tap trung" ? 'green' : 'red';
            canvasCtx.lineWidth = 1;
            
            drawConnectors(canvasCtx, landmarks, FACEMESH_TESSELATION, {color: '#C0C0C070', lineWidth: 0.5});
            drawConnectors(canvasCtx, landmarks, FACEMESH_RIGHT_EYE, {color: '#30FF30'});
            drawConnectors(canvasCtx, landmarks, FACEMESH_LEFT_EYE, {color: '#30FF30'});
            drawConnectors(canvasCtx, landmarks, FACEMESH_FACE_OVAL, {color: '#E0E0E0'});
            
            // Hiển thị trạng thái
            if (i < knownStudents.length) {
                const student = knownStudents[i];
                // Tính vị trí để hiển thị tên và trạng thái
                const nose = landmarks[NOSE_TIP];
                const textX = nose.x * canvasElement.width;
                const textY = (nose.y * canvasElement.height) - 30;
                
                canvasCtx.fillStyle = status === "Tap trung" ? 'green' : 'red';
                canvasCtx.font = '16px Arial';
                canvasCtx.fillText(`${student.name} - ${status}`, textX, textY);
            }
        }
    }
    
    canvasCtx.restore();
    
    // Cập nhật bảng trạng thái học sinh
    updateStudentStatusTable();
});

// Cập nhật bảng trạng thái học sinh
function updateStudentStatusTable() {
    const tableBody = document.getElementById('student-status');
    tableBody.innerHTML = '';
    
    knownStudents.forEach(student => {
        const row = document.createElement('tr');
        row.className = student.status === "Tap trung" ? 'student-row focused' : 'student-row distracted';
        
        const nameCell = document.createElement('td');
        nameCell.textContent = student.name;
        
        const statusCell = document.createElement('td');
        statusCell.textContent = student.status;
        
        const scoreCell = document.createElement('td');
        scoreCell.textContent = `${Math.round(student.attentionScore)}/100`;
        
        row.appendChild(nameCell);
        row.appendChild(statusCell);
        row.appendChild(scoreCell);
        
        tableBody.appendChild(row);
    });
}

// Khởi tạo camera
const camera = new Camera(videoElement, {
    onFrame: async () => {
        await faceMesh.send({image: videoElement});
    },
    width: 640,
    height: 480
});

// Sự kiện nút bắt đầu
startButton.addEventListener('click', () => {
    camera.start();
    startButton.disabled = true;
    stopButton.disabled = false;
});

// Sự kiện nút dừng lại
stopButton.addEventListener('click', () => {
    camera.stop();
    startButton.disabled = false;
    stopButton.disabled = true;
});

// Sự kiện thay đổi ngưỡng nhận diện
thresholdSlider.addEventListener('input', () => {
    FACE_DETECTION_THRESHOLD = parseFloat(thresholdSlider.value);
    thresholdValue.textContent = thresholdSlider.value;
    
    // Cập nhật cấu hình
    faceMesh.setOptions({
        minDetectionConfidence: FACE_DETECTION_THRESHOLD
    });
    
    faceDetection.setOptions({
        minDetectionConfidence: FACE_DETECTION_THRESHOLD
    });
});

// Khởi tạo giao diện
window.addEventListener('load', () => {
    // Cập nhật bảng trạng thái ban đầu
    updateStudentStatusTable();
    
    // Thông báo sẵn sàng
    console.log("Face tracker initialized and ready to start.");
}); 