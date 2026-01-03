# Car Detection API

A custom-trained object detection system that identifies vehicles in street scene images. Built with Faster R-CNN and deployed as a REST API on Google Cloud Run with a web interface for real-time inference. Trained on ~350 labeled street scene images with bounding box annotations.

![Detection Demo](screenshots/detection_visual_example.jpg)

## Model Performance

The performance metric IoU stands for intersection over union and measures how much the predicted box overlaps with the true box from the sample csv file. The model achieved a 59.1% mAP (average precisision across IoU), 94.8% AP50 (precision at 50% IoU threshold), 69.3% AP75 (precision at 75% IoU).This means that 94.8% of the test image cars had atleast 50% label box overlap.

## API Endpoints

**`GET /`** - Health check endpoint

**`POST /predict`** - Returns JSON with car count, bounding boxes, and confidence scores

**`POST /predict/visualize`** - Returns base64-encoded image with bounding boxes drawn

Example response from `/predict`:
```json
{
  "num_cars": 2,
  "detections": [
    {
      "bbox": {"xmin": 100.5, "ymin": 200.3, "xmax": 300.2, "ymax": 400.8},
      "confidence": 0.95
    }
  ],
  "image_size": {"width": 640, "height": 480}
}
```
## Web Interface

The web interface provides a user-friendly way to test the model. Users can upload PNG or JPG images through a simple drag-and-drop interface. The HTML file connects directly to the deployed API on Google Cloud Run and returns an annotated image with bounding boxes drawn around detected vehicles.

![Web Interface Demo](screenshots/web_interface_screenshot.png)

## Installation & Local Setup
```bash
git clone https://github.com/yourusername/car-detection-api.git
cd car-detection-api

pip install -r requirements.txt

# Trained model file (model_final.pth) not included due to size
# Download from [link] or train your own following the training notebook
```

## Deployment

The API is containerized with Docker and deployed on Google Cloud Run.
```bash
# Build and deploy to Google Cloud Run
gcloud run deploy car-detection-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 300
```

**Live API:** `https://car-detection-api-251509948200.us-central1.run.app`

## Testing with Postman
Postman provides an alternative method for testing the API programmatically. Users create a POST request to the `/predict` endpoint with form-data containing an image file. The API returns a JSON response with the number of detected cars, bounding box coordinates for each detection, and confidence scores. This method is useful for developers integrating the API into their own applications or for automated testing workflows.

![Postman Demo](screenshots/postman_screenshot.png)

**Model Training:**
- PyTorch and Detectron2 (Faster R-CNN with ResNet-50 FPN backbone)
- OpenCV for image processing
- Trained in Google Colab

**Backend:**
- FastAPI web framework
- Uvicorn server
- Containerized with Docker
- Deployed on Google Cloud Run

**Frontend:**
- HTML/CSS/JavaScript
- Fetch API for HTTP requests

## Project Structure
```
car-detection-api/
├── Dockerfile
├── requirements.txt
├── main.py
├── index.html
├── screenshots/
└── README.md
```

## Model Training

Used transfer learning to adapt a pre-trained Faster R-CNN model:
1. Loaded weights from COCO dataset pre-training
2. Fine-tuned on street car images (~350 annotated images)
3. Converted CSV bounding box annotations to COCO JSON format
4. Ran 1000 training iterations using 80/20 train/validation split

## License

MIT License - see LICENSE file for details

