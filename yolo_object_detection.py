import cv2
import numpy as np

# Loading the  YOLO model
def load_yolo_model():
    net = cv2.dnn.readNet('weights/yolov3.weights', 'cfg/yolov3.cfg')
    with open('data/coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return net, classes

# Pre-processing the  image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    
    # Resizeing and normalize image
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    
    return image, blob, height, width

# Perform detection using YOLO
def detect_objects(net, blob):
    net.setInput(blob)
    
    # Getting output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    
    # Running forward pass
    detections = net.forward(output_layers)
    
    return detections

# Extracting bounding boxes from detections
def get_bounding_boxes(detections, width, height, confidence_threshold=0.5):
    boxes = []
    confidences = []
    class_ids = []

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > confidence_threshold:
                # Getting object coordinates
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)

                # Calculating coordinates for the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences, class_ids

# Draw bounding boxes on the image
def draw_bounding_boxes(image, boxes, confidences, class_ids, classes):
    # Checking if any bounding boxes were detected
    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                x, y, w, h = box
                label = str(classes[class_ids[i]])
                confidence = confidences[i]

                # Drawing the rectangle and then labeling it 
                color = (0, 255, 0)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            print("No bounding boxes passed the non-max suppression filter.")
    else:
        print("No objects detected.")

    return image


# Saving  and displaying the  image with bounding boxes
def save_and_display_image(image, output_path='output_image.jpg'):
    cv2.imwrite(output_path, image)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main function to run the detection
def main(image_path):
    net, classes = load_yolo_model()
    image, blob, height, width = preprocess_image(image_path)
    detections = detect_objects(net, blob)
    boxes, confidences, class_ids = get_bounding_boxes(detections, width, height, confidence_threshold=0.3)
    image_with_boxes = draw_bounding_boxes(image, boxes, confidences, class_ids, classes)
    save_and_display_image(image_with_boxes)

if __name__ == '__main__':
    main('images/shelf.jpg')
