import torch, cv2, pickle, os
import numpy as np

from ultralytics import YOLO

def model_fn(model_dir):
    print("Executing model_fn from inference.py ...")

    env = os.environ
    model = YOLO('/opt/ml/model/' + env["MODEL"])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('read model!')
    return model

def input_fn(request_body, request_content_type):
    print("Executing input_fn from inference.py ...")
    data = pickle.loads(request_body)

    images = []
    for i in data:
        encoded_img = np.frombuffer(i, dtype=np.uint8)
        img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)

        images.append(img)
    return images
    
def predict_fn(input_data, model):
    with torch.no_grad():
        result = model.track(source=input_data, persist=True)
    print("predicted!")
    return result
        
def output_fn(prediction_output, content_type):
    print("Executing output_fn from inference.py ...")

    return [e.cpu().tojson() for e in prediction_output]


