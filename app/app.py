from ultralytics import YOLO
import gradio as gr
import cv2
import numpy as np


number_plate_model = YOLO(r"runs\exp\weights\best.pt")

def detect_number_plate(image):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = number_plate_model(image)
    
    return results[0].plot( )


image = gr.Image(height=640, width=640, type="numpy")
label = gr.Image(height=640, width=640, type="numpy")

gr.Interface(fn=detect_number_plate, inputs=image, outputs=label).launch(debug=True, share=False)

