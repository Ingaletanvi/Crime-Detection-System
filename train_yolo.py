from ultralytics import YOLO

def train_yolov8():
    # Load a pre-trained YOLOv8 model (e.g. yolov8n.pt for nano)
    model = YOLO("yolov8n.pt")  # or yolov8s.pt, yolov8m.pt, depending on your compute

    # Train on your dataset
    model.train(
        data="data/data.yaml",
    epochs=5,
    batch=8,
    imgsz=320,
    cache=True,
    device='cpu'
    )

if __name__ == "__main__":
    train_yolov8()
