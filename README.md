# Smart security system

This Flask application streams video from a webcam and performs real-time object detection using the YOLO model from Ultralytics. It also captures images of intruders at regular intervals and applies various image processing techniques.

---

## **Installation Steps**
1. Clone this repository to your local machine.
2. Install the required dependencies by running:  
   ```bash
   pip install -r requirements.txt
   ```

---

## **Note Before Run**
1. Train the model with images of your choice in google colab , to get proper weights which is used to classify the intruder in further steps
2. Save the weights as best (3).pt and replace the existing file 
3. Run the application from the terminal only.
4. **Before every run:**
   - Delete the `runs` folder (if it exists) from the directory where the project files are located.
   - Delete all folders ending with `_images` in the same directory to avoid browser caching.
5. Ensure the directory contains only the following files and folders:
   - `app.py`
   - `iptechniques.py`
   - `best (3).pt`
   - `requirements.txt`
   - `Readme.md`
   - `templates` (folder)
   - `logo` (folder)

---

## **Classes Detected**
This trained model can identify only 2 classes:  
- **Sameer**
- **Purna**  

If any person other than Sameer or Purna appears in front of the camera, the model will identify them as **intruder**.

---

## **Usage**
1. Run the Flask application from the terminal:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to the localhost IP address visible in the terminal to:
   - View the real-time video stream.
   - Access image processing functionalities.

---

## **Features**
1. **Real-time video streaming:** Streams video from your webcam.
2. **Object detection:** Utilizes the YOLOv8 model for identifying specific classes and intruders.
3. **Image capture:** Automatically captures images of the detected objects at regular intervals.
4. **Image processing techniques:**
   - Histogram Equalization
   - Edge Detection
   - Median filter
   - Highpass filter
   - Sobel filter
   - Unsharp masking

---

## **Authors**
- Keerthi Reddy Rajamuri  
- Ashwitha Reddy Nimmala  
- Sameer Shaik
- PurnaChander Rao Thoom
- Dakshika Palanisamy  

---
