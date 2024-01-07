import torch
# Load model directly
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import cv2 
##load model configuration through transformers api
processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-101") # -101 for resnet 101
model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-101")

#img=cv2.imread("full.PNG")


def detect_balls(img):
    inputs = processor(images=img, return_tensors="pt")
    outputs = model(**inputs)
    height, width, _ = img.shape
    print(img.shape)
    target_sizes = torch.tensor([[height,width]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]
    boxes=results["boxes"]
    labels=results["labels"]
    boxes_new=[]
    for box, label  in zip(boxes,labels):
         box = [round(i, 2) for i in box.tolist()]
         box=[int(i) for i in box]
         if model.config.id2label[label.item()] == "sports ball":
               boxes_new.append(box)
             
    #print(boxes_new)
    return boxes_new

    
"""
     for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
         box = [round(i, 2) for i in box.tolist()]
         
         print(
             f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
          )
          """   
#sample_failed =cv2.imread("sample_1.jpg")
#detect_balls(sample_failed)




