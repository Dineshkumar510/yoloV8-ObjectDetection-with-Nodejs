const ort = require("onnxruntime-node");
const express = require('express');
const multer = require("multer");
const sharp = require("sharp");
const fs = require("fs");
const ffmpeg = require("fluent-ffmpeg");
const { Readable } = require("stream");
const path = require('path');

// Configure multer for video upload
const storage = multer.memoryStorage();
const upload = multer({ 
    storage: storage,
    limits: {
        fileSize: 50 * 1024 * 1024 // 50MB limit
    }
});

const app = express();
app.use(express.static('public'));

// Define YOLOv8 class labels
const yolo_classes = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
];

// Verify ONNX model exists
const modelPath = path.join(__dirname, 'yolov8m.onnx');
if (!fs.existsSync(modelPath)) {
    console.error('ERROR: YOLOv8 model file not found! Please ensure yolov8m.onnx is in the root directory.');
    process.exit(1);
}

app.post('/detect', upload.single('video'), async (req, res) => {
    if (!req.file) {
        return res.status(400).send('No video file uploaded');
    }
    console.log('Processing video:', {
        mimetype: req.file.mimetype,
        size: req.file.size,
        originalName: req.file.originalname
    });
    
    try {
        const results = await detectObjectsInVideo(req.file.buffer);
        console.log(`Processed ${results.length} frames with detections`);
        res.json(results);
    } catch (error) {
        console.error('Error processing video:', error);
        res.status(500).send(`Error processing video: ${error.message}`);
    }
});

function cleanupTempDir() {
    const tempDir = path.join(__dirname, 'temp_frames');
    if (fs.existsSync(tempDir)) {
        fs.rmSync(tempDir, { recursive: true });
        console.log('Temporary directory cleaned up');
    }
}

async function detectObjectsInVideo(videoBuffer) {
    return new Promise((resolve, reject) => {
        const frames = [];
        const tempDir = path.join(__dirname, 'temp_frames');
        
        // Ensure temp directory exists and is empty
        if (fs.existsSync(tempDir)) {
            fs.rmSync(tempDir, { recursive: true });
        }
        fs.mkdirSync(tempDir);
        
        console.log('Starting video processing...');
        
        ffmpeg(Readable.from(videoBuffer))
            .inputFormat('mp4')
            .outputOptions([
                '-vf', 'fps=1',
                '-vsync', '0',
                '-frame_pts', '1'  // Add presentation timestamp
            ])
            .output(path.join(tempDir, 'frame-%d.jpg'))
            .on('start', () => console.log('FFmpeg started processing'))
            .on('progress', (progress) => {
                console.log(`FFmpeg Progress: ${progress.percent}%`);
            })
            .on('error', (err) => {
                console.error('FFmpeg error:', err);
                cleanupTempDir();
                reject(err);
            })
            .on('end', async () => {
                try {
                    console.log('FFmpeg finished extracting frames');
                    const frameFiles = fs.readdirSync(tempDir)
                        .filter(file => file.endsWith('.jpg'))
                        .sort((a, b) => {
                            const numA = parseInt(a.match(/\d+/)[0]);
                            const numB = parseInt(b.match(/\d+/)[0]);
                            return numA - numB;
                        });
                    
                    console.log(`Found ${frameFiles.length} frames to process`);
                    
                    for (const file of frameFiles) {
                        const frameBuffer = fs.readFileSync(path.join(tempDir, file));
                        const detections = await detectObjectsOnImage(frameBuffer);
                        frames.push(detections);
                        console.log(`Processed frame ${file}: found ${detections.length} objects`);
                    }
                    
                    cleanupTempDir();
                    resolve(frames);
                } catch (err) {
                    console.error('Frame processing error:', err);
                    cleanupTempDir();
                    reject(err);
                }
            })
            .run();
    });
}

async function detectObjectsOnImage(imageBuffer) {
    try {
        // Enhanced error logging
        console.log('Starting object detection on frame');
        
        const img = sharp(imageBuffer);
        const metadata = await img.metadata();
        
        console.log('Image metadata:', metadata);
        
        if (!metadata || !metadata.width || !metadata.height) {
            throw new Error('Invalid image metadata');
        }

        const processedBuffer = await img
            .jpeg()
            .toBuffer();

        const [input, imgWidth, imgHeight] = await prepareInput(processedBuffer);
        console.log(`Prepared input tensor for ${imgWidth}x${imgHeight} image`);
        
        const output = await runModel(input);
        const detections = processOutput(output, imgWidth, imgHeight);
        console.log(`Found ${detections.length} objects in frame`);
        
        return detections;
    } catch (error) {
        console.error('Error in object detection:', error);
        return [];
    }
}

async function runModel(input) {
    try {
        console.log('Loading ONNX model...');
        const model = await ort.InferenceSession.create(modelPath);
        
        console.log('Running inference...');
        const tensor = new ort.Tensor('float32', input, [1, 3, 640, 640]);
        const results = await model.run({ images: tensor });
        
        if (!results['output0'] || !results['output0'].data) {
            throw new Error('Model output is invalid');
        }
        
        console.log('Inference completed successfully');
        return results['output0'].data;
    } catch (error) {
        console.error('Error running model:', error);
        throw error;
    }
}



async function prepareInput(imageBuffer) {
    try {
        const img = sharp(imageBuffer);
        const metadata = await img.metadata();
        
        if (!metadata || !metadata.width || !metadata.height) {
            throw new Error('Invalid image metadata');
        }
        
        const [imgWidth, imgHeight] = [metadata.width, metadata.height];
        
        // Enhanced image processing pipeline
        const resizedImage = await img
            .removeAlpha()
            .resize(640, 640, {
                fit: 'fill',
                withoutEnlargement: false
            })
            .raw()
            .toBuffer({ resolveWithObject: true });

        if (resizedImage.data.length !== 640 * 640 * 3) {
            throw new Error(`Invalid image dimensions: ${resizedImage.data.length} bytes`);
        }

        const channels = {
            r: new Float32Array(640 * 640),
            g: new Float32Array(640 * 640),
            b: new Float32Array(640 * 640)
        };

        for (let i = 0; i < resizedImage.data.length; i += 3) {
            const pixel = i / 3;
            channels.r[pixel] = resizedImage.data[i] / 255.0;
            channels.g[pixel] = resizedImage.data[i + 1] / 255.0;
            channels.b[pixel] = resizedImage.data[i + 2] / 255.0;
        }

        const input = new Float32Array([...channels.r, ...channels.g, ...channels.b]);
        return [input, imgWidth, imgHeight];
    } catch (error) {
        console.error('Error in prepareInput:', error);
        throw error;
    }
}


function processOutput(output, imgWidth, imgHeight) {
    let processedBoxes = []; // Changed from const to let
    const confidenceThreshold = 0.5;
    const numClasses = 80;
    const numBoxes = 8400;

    // Process each detection
    for (let i = 0; i < numBoxes; i++) {
        // Find class with highest confidence for this detection
        let maxConfidence = 0;
        let detectedClass = -1;

        for (let j = 0; j < numClasses; j++) {
            const confidence = output[numBoxes * (j + 4) + i];
            if (confidence > maxConfidence) {
                maxConfidence = confidence;
                detectedClass = j;
            }
        }

        // Skip if confidence is below threshold
        if (maxConfidence < confidenceThreshold) continue;

        // Get bounding box coordinates
        const x = output[i];
        const y = output[numBoxes + i];
        const w = output[2 * numBoxes + i];
        const h = output[3 * numBoxes + i];

        // Convert normalized coordinates to actual image coordinates
        const x1 = Math.max(0, ((x - w/2) / 640) * imgWidth);
        const y1 = Math.max(0, ((y - h/2) / 640) * imgHeight);
        const x2 = Math.min(imgWidth, ((x + w/2) / 640) * imgWidth);
        const y2 = Math.min(imgHeight, ((y + h/2) / 640) * imgHeight);

        processedBoxes.push([x1, y1, x2, y2, yolo_classes[detectedClass], maxConfidence]);
    }

    // Sort boxes by confidence
    processedBoxes.sort((a, b) => b[5] - a[5]);

    // Apply Non-Maximum Suppression (NMS)
    const selectedBoxes = [];
    const iouThreshold = 0.7;

    while (processedBoxes.length > 0) {
        selectedBoxes.push(processedBoxes[0]);
        const remaining = processedBoxes.filter(box => 
            calculateIOU(processedBoxes[0], box) < iouThreshold || box === processedBoxes[0]
        );
        processedBoxes = remaining.slice(1);
    }

    return selectedBoxes;
}


function calculateIOU(box1, box2) {
    // Calculate intersection
    const [x1_1, y1_1, x2_1, y2_1] = box1;
    const [x1_2, y1_2, x2_2, y2_2] = box2;

    const intersectionX1 = Math.max(x1_1, x1_2);
    const intersectionY1 = Math.max(y1_1, y1_2);
    const intersectionX2 = Math.min(x2_1, x2_2);
    const intersectionY2 = Math.min(y2_1, y2_2);

    const intersectionArea = Math.max(0, intersectionX2 - intersectionX1) * 
                           Math.max(0, intersectionY2 - intersectionY1);

    // Calculate union
    const box1Area = (x2_1 - x1_1) * (y2_1 - y1_1);
    const box2Area = (x2_2 - x1_2) * (y2_2 - y1_2);
    const unionArea = box1Area + box2Area - intersectionArea;

    return intersectionArea / unionArea;
}

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});