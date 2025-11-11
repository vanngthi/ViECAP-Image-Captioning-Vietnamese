from src.detector.detection import ObjectDetector

detector = ObjectDetector()
results = detector.detect("./dataset/Flick_sportball/images/111796099.jpg")

for obj in results:
    print(obj["label"], obj["confidence"])