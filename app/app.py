import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import gradio as gr
import json


number_plate_model = YOLO("best.pt", task='detection')
reader = easyocr.Reader(['en'])

def detect_number_plate(image):
    results = number_plate_model(image)
    for result in results:
        result = json.loads(result.tojson()) 
        
        x1, y1, x2, y2 = map(int, result[0]['box'].values())
        # print(x1, y1, x2, y2)
        
        number_plate = image[y1:y2, x1:x2]
        number_plate = cv2.cvtColor(number_plate, cv2.COLOR_BGR2RGB)
        
        number_plate_text = reader.readtext(number_plate)
        if len(number_plate_text) > 0:
            number_plate_text = number_plate_text[0][-2]
        else:
            number_plate_text = "No text detected"
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, number_plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    
    return image

# Define Gradio interface
image = gr.Image(height=640, width=640, type="numpy")
label = gr.Image(height=640, width=640, type="numpy")

gr.Interface(fn=detect_number_plate, inputs=image, outputs=label).launch(debug=True, share=False)
