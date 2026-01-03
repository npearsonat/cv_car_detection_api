from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import io
from PIL import Image
import base64

app = FastAPI(title="Car Detection API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor - load lazily
predictor = None

def get_predictor():
    """Lazy load the model on first request"""
    global predictor
    if predictor is None:
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = "./model_final.pth"
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.DEVICE = "cpu"
        predictor = DefaultPredictor(cfg)
    return predictor

@app.get("/")
def read_root():
    return {"message": "Car Detection API", "status": "running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        pred = get_predictor()  # Load model on first call
        
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        outputs = pred(img)
        instances = outputs["instances"].to("cpu")
        
        boxes = instances.pred_boxes.tensor.numpy().tolist()
        scores = instances.scores.numpy().tolist()
        
        detections = []
        for box, score in zip(boxes, scores):
            detections.append({
                "bbox": {
                    "xmin": float(box[0]),
                    "ymin": float(box[1]),
                    "xmax": float(box[2]),
                    "ymax": float(box[3])
                },
                "confidence": float(score)
            })
        
        return JSONResponse({
            "num_cars": len(detections),
            "detections": detections,
            "image_size": {
                "width": img.shape[1],
                "height": img.shape[0]
            }
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/visualize")
async def predict_visualize(file: UploadFile = File(...)):
    try:
        pred = get_predictor()  # Load model on first call
        
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        outputs = pred(img)
        instances = outputs["instances"].to("cpu")
        
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Car: {score:.2f}"
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return JSONResponse({
            "num_cars": len(boxes),
            "image": img_base64
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)