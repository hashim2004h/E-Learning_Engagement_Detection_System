// static/js/script.js
let isMonitoring = false;
let mediaStream = null;
let audioContext = null;
let analyser = null;
let sessionId = Date.now().toString();
let mouseActivity = 0;

document.getElementById('startBtn').addEventListener('click', startMonitoring);
document.getElementById('stopBtn').addEventListener('click', stopMonitoring);

async function startMonitoring() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        mediaStream = stream;
        
        // Setup webcam
        const video = document.getElementById('webcam');
        video.srcObject = stream;
        
        // Setup audio analysis
        setupAudioAnalyser(stream);
        
        // Start face detection
        setInterval(captureFrame, 1000);
        
        // Start mouse tracking
        document.addEventListener('mousemove', handleMouseMove);
        
        isMonitoring = true;
    } catch (err) {
        console.error('Error accessing media devices:', err);
    }
}

function setupAudioAnalyser(stream) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    const source = audioContext.createMediaStreamSource(stream);
    source.connect(analyser);
    analyser.fftSize = 32;
    
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    function updateAudioLevel() {
        analyser.getByteFrequencyData(dataArray);
        const level = Math.max(...dataArray) / 255;
        document.getElementById('voiceLevel').style.width = `${level * 100}%`;
        if(isMonitoring) requestAnimationFrame(updateAudioLevel);
    }
    updateAudioLevel();
}

function handleMouseMove() {
    mouseActivity = Math.min(mouseActivity + 2, 100);
    document.getElementById('mouseActivity').style.width = `${mouseActivity}%`;
    setTimeout(() => {
        mouseActivity = Math.max(mouseActivity - 1, 0);
        document.getElementById('mouseActivity').style.width = `${mouseActivity}%`;
    }, 100);
}

async function captureFrame() {
    if(!isMonitoring) return;
    
    const video = document.getElementById('webcam');
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    
    const image = canvas.toDataURL('image/jpeg');
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image, session_id: sessionId })
        });
        
        const data = await response.json();
        document.getElementById('emotionDisplay').textContent = data.emotion;
        document.getElementById('confidenceDisplay').textContent = 
            `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
    } catch (err) {
        console.error('Prediction error:', err);
    }
}

function stopMonitoring() {
    isMonitoring = false;
    if(mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
    }
    document.removeEventListener('mousemove', handleMouseMove);
}