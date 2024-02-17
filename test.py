from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
model = YOLO('best.pt')

# Perform object detection on an image using the model
results = model('test.jpg')

# Display the results
results.show()
