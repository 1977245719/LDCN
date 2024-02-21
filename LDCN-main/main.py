
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("ultralytics/cfg/models/v8/yolov8LDCN.yaml")
    results = model.train(data="datasets/datavoc.yaml", epochs=200)
    results = model.train(resume=True)



