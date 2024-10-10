
# YOLO Object Detection Project

This project demonstrates how to use OpenCV and a pre-trained YOLOv3 model for object detection on images. The code processes an image, detects objects, and draws bounding boxes around them.

## Project Overview

- **Frameworks used**: OpenCV, YOLOv3
- **Languages used**: Python
- **Dependencies**: Listed in `requirements.txt`

This project implements object detection using YOLOv3 and draws bounding boxes around detected objects. The YOLOv3 model is pre-trained on the COCO dataset, which contains 80 object classes.

## Getting Started

### Prerequisites

Before running this project, ensure you have the following installed on your local machine:

- Python 3.x

### Installation Steps

1. **Clone this repository**:

    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    ```

2. **Navigate into the project directory**:

    ```bash
    cd your-repository-name
    ```

3. **Set up a virtual environment**:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

4. **Install the required Python packages**:

    ```bash
    pip install -r requirements.txt
    ```

### Download YOLOv3 Weights

Due to file size limitations, the `yolov3.weights` file is not included in the repository. You will need to manually download it.

1. **Download the pre-trained YOLOv3 weights** from the following link:
   - [Download YOLOv3 Weights](https://pjreddie.com/media/files/yolov3.weights)

2. **Place the `yolov3.weights` file in the `weights/` directory** of the project. If the `weights/` directory does not exist, create it.

   Your folder structure should look like this:

   ```
   ├── weights/
       └── yolov3.weights
   ```

### Running the Project

1. Ensure that you have an image file in the `images/` folder. You can replace `shelf.jpg` with your own image if desired.
   
2. Open a terminal and run the Python script for object detection:

    ```bash
    python yolo_object_detection.py
    ```

3. After running the script, the processed image with bounding boxes will be saved as `output_image.jpg` in the project directory and displayed in a window.

### Project Structure

The project is organized into the following structure:

```plaintext
├── cfg/
│   └── yolov3.cfg            # YOLO configuration file
├── data/
│   └── coco.names            # COCO dataset object names
├── images/
│   └── shelf.jpg             # Input images for object detection
├── weights/
│   └── yolov3.weights        # YOLOv3 pre-trained weights (not included, download manually)
├── requirements.txt          # Python dependencies
├── yolo_object_detection.py  # Main Python script for object detection
└── README.md                 # Project documentation
```

### Example Output

Here’s an example of the output you can expect from running the project:

![output_image](https://github.com/user-attachments/assets/818bb888-bfec-49a7-b27d-ce5e5bb54b05)


### Known Issues

- **YOLOv3 Weights**: The `yolov3.weights` file is too large to be included in the repository. Follow the instructions above to manually download and add it.
- **Object Detection Limitations**: YOLOv3 is trained on 80 common object categories, so it may not detect objects outside of this scope without retraining the model.
