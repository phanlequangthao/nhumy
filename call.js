'use strict';

const baseURL = "https://b6fd-14-164-114-227.ngrok-free.app/"
let remoteVideo = document.querySelector('#remoteVideo');
// let localVideo = document.querySelector('#localVideo');  // Thêm dòng này
let mediaPipeHands;
let mediaPipeCamera;
let otherUser;
let remoteRTCMessage;
const NUM_OF_TIMESTEPS = 12;
const NUM_HAND_LANDMARKS = 21;
let frameBuffer = [];
let label = "Unknown";
let confidence = 0;
let iceCandidatesFromCaller = [];
let peerConnection;
let remoteStream;
let localStream;
let mediaRecorder = null;
const startRecordingButton = document.getElementById('startRecording');
const stopRecordingButton = document.getElementById('stopRecording');
let callInProgress = false;
let isProcessing = false;
let lstlabel = "";
let labelcnt = 0;
const LABEL_THRESHOLD = 3;


function call() {
    let userToCall = document.getElementById("callName").value;
    otherUser = userToCall;
    createPeerConnection();
    beReady()
        .then(bool => {
            processCall(userToCall)
        })
}

//event from html
function answer() {
    //do the event firing

    beReady()
        .then(bool => {
            processAccept();
        })

    document.getElementById("answer").style.display = "none";
}

let pcConfig = {
    "iceServers": [
        // STUN Servers
        // { urls: 'stun:stun.l.google.com:19302' },
        // { urls: 'stun:stun1.l.google.com:19302' },
        // { urls: 'stun:stun2.l.google.com:19302' },
        // { urls: 'stun:stun3.l.google.com:19302' },
        // { urls: 'stun:stun4.l.google.com:19302' },
        // { urls: 'stun:stun.services.mozilla.com' },
        // { urls: 'stun:stun.stunprotocol.org:3478' },

        // TURN Servers
        {
            urls: ["stun:hk-turn1.xirsys.com"]
        },
        // TURN server từ Xirsys
        {
            username: "KPuma8XJuifPAr3utWXEuSDa5d4w6nvDelWF05X0jvmkPWBjTNnDHPmBen7xKxArAAAAAGdeSup0aGFvcGhhbg==",
            credential: "7e025f0e-ba93-11ef-9570-0242ac120004",
            urls: [
                "turn:hk-turn1.xirsys.com:80?transport=udp",
                "turn:hk-turn1.xirsys.com:3478?transport=udp",
                "turn:hk-turn1.xirsys.com:80?transport=tcp",
                "turn:hk-turn1.xirsys.com:3478?transport=tcp",
                "turns:hk-turn1.xirsys.com:443?transport=tcp",
                "turns:hk-turn1.xirsys.com:5349?transport=tcp"
            ]
        },
        // {
        //     url: "turn:relay1.expressturn.com:3478",
        //     username: "tkavd5I74u5S2iE9",
        //     credential: "efRU01A9B3T19DPHNE"
        // },
        // {
        //     url: 'turn:turn.bistri.com:80',
        //     credential: 'homeo',
        //     username: 'homeo'
        // },
        // {
        //     url: 'turn:turn.anyfirewall.com:443?transport=tcp',
        //     credential: 'webrtc',
        //     username: 'webrtc'
        // },
        // {
        //     url: "TURNS:freeturn.net:5349",
        //     username: "free",
        //     credential: "free"
        // }
    ]
};


// Set up audio and video regardless of what devices are present.
let sdpConstraints = {
    offerToReceiveAudio: true,
    offerToReceiveVideo: true
};

/////////////////////////////////////////////

let socket;
let callSocket;
function connectSocket() {
    let ws_scheme = window.location.protocol == "https:" ? "wss://" : "ws://";
    

    callSocket = new WebSocket(
        ws_scheme
        + window.location.host
        + '/ws/call/'
    );
    console.log(callSocket);
    callSocket.onopen = event =>{
    //let's send myName to the socket
        callSocket.send(JSON.stringify({
            type: 'login',
            data: {
                name: myName
            }
        }));
    }
    
    callSocket.onmessage = (e) =>{
        let response = JSON.parse(e.data);
        console.log("Message received from server:", response);
        // console.log(response);

        let type = response.type;

        if(type == 'connection') {
            console.log(response.data.message)
        }

        if(type == 'call_received') {
            // console.log(response);
            onNewCall(response.data)
        }

        if(type == 'call_answered') {
            onCallAnswered(response.data);
        }

        if(type == 'ICEcandidate') {
            onICECandidate(response.data);
        }
        if (type == 'chat_message') {
            displayMessage(response.data.message, 'received');
            console.log("Received message: ", response.data.message);
        }
    };

    const onNewCall = (data) => {
        otherUser = data.caller;
        remoteRTCMessage = data.rtcMessage;

        document.getElementById("callerName").innerHTML = otherUser;
        document.getElementById("call").style.display = "none";
        document.getElementById("answer").style.display = "block";
    };

    const onCallAnswered = (data) => {
        remoteRTCMessage = data.rtcMessage;
        peerConnection.setRemoteDescription(new RTCSessionDescription(remoteRTCMessage));

        document.getElementById("calling").style.display = "none";

        console.log("Call Started. They Answered");

        callProgress();
    };

    const onICECandidate = (data) => {
        console.log("GOT ICE candidate");

        let message = data.rtcMessage;

        let candidate = new RTCIceCandidate({
            sdpMLineIndex: message.label,
            candidate: message.candidate
        });

        if (peerConnection) {
            console.log("ICE candidate Added");
            peerConnection.addIceCandidate(candidate);
        } else {
            console.log("ICE candidate Pushed");
            iceCandidatesFromCaller.push(candidate);
        }
    };
}

/**
 * 
 * @param {Object} data 
 * @param {number} data.name - the name of the user to call
 * @param {Object} data.rtcMessage - the rtc create offer object
 */
// var btnSendMsg = document.querySelector('#btn-send-msg');
// var messagelist = document.querySelector('#message-list');
// var messageInput = document.querySelector ('#msg');
// btnSendMsg.addEventListener('click', SendMsgOnClick);

function sendMessage() {
    const chatInput = document.getElementById('chat-input');
    const message = chatInput.value.trim();
    if (message && peerConnection.dataChannel.readyState === "open") {
        peerConnection.dataChannel.send(message);
        console.log("Data Channel State:", peerConnection.dataChannel.readyState);
        displayMessage(message, 'sent');
        console.log("done connect");
        chatInput.value = '';
    } else {
        console.log("Data Channel is not open or message is empty.");
    }
}
function displayMessage(message, type) {
    const li = document.createElement("li");
    li.textContent = type === 'sent' ? `Me: ${message}` : `Stranger: ${message}`;
    console.log(li.textContent)
    document.querySelector("#message-list").appendChild(li);
    console.log("done display");
}

function sendCall(data) {
    //to send a call
    console.log("Send Call");

    // socket.emit("call", data);
    callSocket.send(JSON.stringify({
        type: 'call',
        data
    }));

    document.getElementById("call").style.display = "none";
    // document.getElementById("profileImageCA").src = baseURL + otherUserProfile.image;
    document.getElementById("otherUserNameCA").innerHTML = otherUser;
    document.getElementById("calling").style.display = "block";
}

/**
 * 
 * @param {Object} data 
 * @param {number} data.caller - the caller name
 * @param {Object} data.rtcMessage - answer rtc sessionDescription object
 */
function answerCall(data) {
    callSocket.send(JSON.stringify({
        type: 'answer_call',
        data
    }));
    callProgress();
}

/**
 * 
 * @param {Object} data 
 * @param {number} data.user - the other user //either callee or caller 
 * @param {Object} data.rtcMessage - iceCandidate data 
 */
function sendICEcandidate(data) {
    console.log("Send ICE candidate");
    callSocket.send(JSON.stringify({
        type: 'ICEcandidate',
        data
    }));
}



let lastres = '';


let camera;
let localVideo;

document.addEventListener('DOMContentLoaded', async () => {
    localVideo = document.querySelector('#localVideo');
    
    try {
        camera = new Camera(localVideo, {
            onFrame: async () => {
                await captureAndSendFrame();
            },
            width: 720,
            height: 720
        });
        await camera.start();
        console.log("Camera started successfully");
    } catch (error) {
        console.error("Error starting camera:", error);
    }
});

async function captureAndSendFrame() {
    if (!localVideo || !localVideo.videoWidth) {
        console.log('Video not ready yet');
        return;
    }

    // console.log('Capturing frame');

    const canvas = document.createElement('canvas');
    canvas.width = localVideo.videoWidth;
    canvas.height = localVideo.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(localVideo, 0, 0, canvas.width, canvas.height);

    try {
        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg'));
        frameBuffer.push(blob);

        if (frameBuffer.length === NUM_OF_TIMESTEPS && !isProcessing) {
            isProcessing = true;
            await sendframe();
            isProcessing = false;
        }
    } catch (error) {
        console.error('Failed to capture frame:', error);
    }
}

async function sendframe() {
    const formData = new FormData(); 
    frameBuffer.forEach((frame, index) => {
        formData.append(`frame${index}`, frame, `frame${index}.jpg`);
    }); // thêm các ảnh đã cắt từ camera vào 1 buffer

    // console.log(`send ${frameBuffer.length} frame to sv`);

    try {
        const response = await fetch(`${baseURL}api/analyze_frames`, {
            method: 'POST',
            body: formData
        }); // gửi các ảnh lên server

        if (!response.ok) {
            // throw new Error(`sv er: ${response.status}`);
        }

        const res = await response.json(); // nhận về kết quả từ server
        console.log('sv res:', res);

        if (res.label && res.confidence) {
            updateLabelDisplay(res.label, res.confidence); // in ra label
            
            if (res.label === lstlabel) { //kiểm tra xem kết quả đợt 2 có giống đợt 1 ko
                labelcnt++; // tăng số lần giống nhau
                if (labelcnt >= LABEL_THRESHOLD && res.confidence >= 0.85) { 
                    // nếu đủ 5 lần giống nhau và có độ tin cậy lớn hơn 0.85 thì tiếp tục
                    if (res.label !== lastres) {
                        if( res.label == "space"){
                            res.label = " ";
                        }
                        appendres(res.label); // thêm class vào ô trò truyện
                        lastres = res.label; 
                        // cập nhật label cuối là label hiện tại, tránh thêm liên tục label lên ô chat
                    }
                    labelcnt = 0;
                    lstlabel = "";
                }
            } else {
                labelcnt = 1;
                lstlabel = res.label;
            }
        } else {
            console.error('er sv:', res.error);
        }
    } catch (error) {
        // console.error('failed to send to sv: ', error);
    } finally {
        frameBuffer = [];
    }
    // in ra các lỗi của server nếu có
}


function beReady() {
    return new Promise((resolve, reject) => {
        if (camera && camera.video && camera.video.srcObject) {
            localVideo.srcObject = camera.video.srcObject;
            localStream = camera.video.srcObject;
            navigator.mediaDevices.getUserMedia({
                audio: true
            })
            .then(audioStream => {
                localStream = new MediaStream([...camera.video.srcObject.getTracks(), ...audioStream.getTracks()]);
                
                localVideo.srcObject = localStream;
                createConnectionAndAddStream().then(resolve).catch(reject);
            })
            .catch(error => {
                reject(error);
            });
        } else {
            reject(new Error("Camera is not ready or srcObject is not available"));
        }
    });
}





function appendres(newres) {
    const chatInput = document.getElementById("chat-input");
    const currentText = chatInput.value;
    if (newres !== lastres) {
        chatInput.value = currentText ? `${currentText}${newres}` : newres;
        lastres = newres;
    }
}

function checkLastres() {
    lastres = '';
}





function createConnectionAndAddStream() {
    return new Promise((resolve, reject) => {
        try {
            createPeerConnection();
            peerConnection.addStream(localStream);
            resolve(true);
        } catch (error) {
            reject(error);
        }
    });
}


function processCall(userName) {
    peerConnection.createOffer((sessionDescription) => {
        peerConnection.setLocalDescription(sessionDescription);
        sendCall({
            name: userName,
            rtcMessage: sessionDescription
        })
    }, (error) => {
        console.log("Error");
    });
}

function processAccept() {

    peerConnection.setRemoteDescription(new RTCSessionDescription(remoteRTCMessage));
    peerConnection.createAnswer((sessionDescription) => {
        peerConnection.setLocalDescription(sessionDescription);

        if (iceCandidatesFromCaller.length > 0) {
            //I am having issues with call not being processed in real world (internet, not local)
            //so I will push iceCandidates I received after the call arrived, push it and, once we accept
            //add it as ice candidate
            //if the offer rtc message contains all thes ICE candidates we can ingore this.
            for (let i = 0; i < iceCandidatesFromCaller.length; i++) {
                //
                let candidate = iceCandidatesFromCaller[i];
                console.log("ICE candidate Added From queue");
                try {
                    peerConnection.addIceCandidate(candidate).then(done => {
                        console.log(done);
                    }).catch(error => {
                        console.log(error);
                    })
                } catch (error) {
                    console.log(error);
                }
            }
            iceCandidatesFromCaller = [];
            console.log("ICE candidate queue cleared");
        } else {
            console.log("NO Ice candidate in queue");
        }

        answerCall({
            caller: otherUser,
            rtcMessage: sessionDescription
        })

    }, (error) => {
        console.log("Error");
    })
}

/////////////////////////////////////////////////////////

function createPeerConnection() {
    try {
        peerConnection = new RTCPeerConnection(pcConfig);
        peerConnection.dataChannel = peerConnection.createDataChannel("chat");

        setupDataChannelEvents(peerConnection.dataChannel);

        peerConnection.onicecandidate = handleIceCandidate;
        peerConnection.onaddstream = handleRemoteStreamAdded;
        peerConnection.onremovestream = handleRemoteStreamRemoved;
        console.log('RTCPeerConnnection and Data Channel created');
    } catch (e) {
        console.error('Failed to create PeerConnection or Data Channel, exception:', e);
    }
}

function setupDataChannelEvents(dataChannel) {
    dataChannel.onopen = function() {
        console.log("Data Channel is open");
    };

    dataChannel.onclose = function() {
        console.log("Data Channel is closed");
    };

    dataChannel.onerror = function(error) {
        console.error("Data Channel Error:", error);
    };

    dataChannel.onmessage = function(event) {
        console.log("Received Data Channel message:", event.data);
        displayMessage(event.data, 'received'); 
    };
    peerConnection.ondatachannel = function(event) {
        peerConnection.dataChannel = event.channel;
        setupDataChannelEvents(peerConnection.dataChannel);
    };
}




function handleIceCandidate(event) {
    // console.log('icecandidate event: ', event);
    if (event.candidate) {
        console.log("Local ICE candidate");
        // console.log(event.candidate.candidate);

        sendICEcandidate({
            user: otherUser,
            rtcMessage: {
                label: event.candidate.sdpMLineIndex,
                id: event.candidate.sdpMid,
                candidate: event.candidate.candidate
            }
        })

    } else {
        console.log('End of candidates.');
    }
}

function handleRemoteStreamRemoved(event) {
    console.log('Remote stream removed. Event: ', event);
    remoteVideo.srcObject = null;
    localVideo.srcObject = null;
}

function handleRemoteStreamAdded(event) {
    console.log('Remote stream added.');
    remoteStream = event.stream;
    remoteVideo.srcObject = remoteStream;
    remoteVideo.onloadedmetadata = function(e) {
        remoteVideo.play();
    };
    remoteVideo.width = 1280;
    remoteVideo.height = 720;
}
function showElement(elementId) {
    document.getElementById(elementId).classList.remove('hidden');
    document.getElementById(elementId).classList.add('visible');
}

function hideElement(elementId) {
    document.getElementById(elementId).classList.remove('visible');
    document.getElementById(elementId).classList.add('hidden');
}

function login() {
    let userName = document.getElementById('userNameInput').value;
    myName = userName;
    hideElement('userName');
    showElement('call');

    document.getElementById('nameHere').innerHTML = userName;
    showElement('userInfo');

    connectSocket();
}

window.onbeforeunload = function () {
    if (callInProgress) {
        stop();
    }
};


function stop() {
    localStream.getTracks().forEach(track => track.stop());
    callInProgress = false;
    peerConnection.close();
    peerConnection = null;
    document.getElementById("call").style.display = "block";
    document.getElementById("answer").style.display = "none";
    document.getElementById("inCall").style.display = "none";
    document.getElementById("calling").style.display = "none";
    document.getElementById("endVideoButton").style.display = "none"
    otherUser = null;
}

function callProgress() {

    document.getElementById("videos").style.display = "block";
    document.getElementById("otherUserNameC").innerHTML = otherUser;
    document.getElementById("inCall").style.display = "block";

    callInProgress = true;
}






function updateLabelDisplay(predictedLabel, confidenceValue) {
    const labelElement = document.getElementById('labelDisplay');
    const confidenceElement = document.getElementById('confidenceDisplay');

    if (labelElement) labelElement.textContent = `Predicted Sign: ${predictedLabel}`;
    if (confidenceElement) confidenceElement.textContent = `Confidence: ${confidenceValue.toFixed(4)}`;
}

function startRecording() {
    console.log('Start recording clicked');
    let recordedChunks = [];
    const canvas = document.querySelector('.output_video');
    if (!canvas) {
        console.error('Canvas not found');
        return;
    }
    const stream = canvas.captureStream(30); // 30 FPS

    // Try to use H.264 codec for MP4
    const options = {mimeType: 'video/mp4;codecs=avc1.42E01E,mp4a.40.2'};
    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
        console.error(`${options.mimeType} is not supported`);
        return;
    }

    mediaRecorder = new MediaRecorder(stream, options);
    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            recordedChunks.push(event.data);
        }
    };
    mediaRecorder.onstop = () => {
        console.log('Recording stopped');
        const blob = new Blob(recordedChunks, {
            type: 'video/mp4'
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        document.body.appendChild(a);
        a.style = 'display: none';
        a.href = url;
        a.download = 'recorded-video.mp4';
        a.click();
        window.URL.revokeObjectURL(url);
    };
    mediaRecorder.start();
    startRecordingButton.style.display = 'none';
    stopRecordingButton.style.display = 'inline-block';
}

function stopRecording() {
    console.log('Stop recording clicked');
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        startRecordingButton.style.display = 'inline-block';
        stopRecordingButton.style.display = 'none';
    }
}

// Đảm bảo rằng các nút đã được tải trước khi gán sự kiện
document.addEventListener('DOMContentLoaded', (event) => {
    const startRecordingButton = document.getElementById('startRecording');
    const stopRecordingButton = document.getElementById('stopRecording');

    if (startRecordingButton) {
        startRecordingButton.addEventListener('click', startRecording);
    }

    if (stopRecordingButton) {
        stopRecordingButton.addEventListener('click', stopRecording);
    }
});
