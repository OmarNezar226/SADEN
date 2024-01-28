from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "../04-06-segment-anything/weights/groundingdino_swint_ogc.pth") #loading the detection model (model structure, model weights)
IMAGE_PATH = "/usr/image to be detected"
TEXT_PROMPT = "car" #promt object to be detected
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH) #pre-loading the image to be detected 

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
) #passing the image along with params to the detection model

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases) #overwriting the detected object, bounding box and probability on the original image
cv2.imwrite("annotated_image.jpg", annotated_frame) #saving teh overwritten image to a file