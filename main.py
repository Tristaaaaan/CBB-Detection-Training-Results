from ultralytics import YOLO
 
# Load the model.
model = YOLO('yolov8m.pt')
 
# Training.
results = model.train(
   data='data.yaml',
   imgsz=640,
   batch=5,
   epochs=8,
   name='cbbdata')