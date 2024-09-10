import os
import cv2
import time
import requests
import paddlehub as hub
from ultralytics import YOLO

yolo_model = YOLO('weights/hull.pt')
ocr = hub.Module(name="ch_pp-ocrv3")
external_url = 'https://dashboard-kpp.kecilin.id/api/v1/hull_number/store'
rtsp_url = "rtsp://localhost:8554/PITSTOP"

def detect_hull(image):
    results = yolo_model.predict(image)
    bboxes = []
    for result in results:
        for box in result.boxes:
            if int(box.cls) == 0:  
                bboxes.append(box.xywh.cpu().numpy()[0])
    return bboxes

def draw_bbox(image, bboxes):
    for bbox in bboxes:
        x, y, w, h = bbox
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image

def crop_image(image, bbox):
    x, y, w, h = bbox
    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)
    return image[y1:y2, x1:x2]

def perform_ocr(image):
    results = ocr.recognize_text(images=[image])
    return results

def encode_image_to_bytes(image):
    _, buffer = cv2.imencode('.jpg', image)
    return buffer.tobytes()

def process_frame(frame):
    hull_data = []
    detected_text = ""

    bboxes = detect_hull(frame)
    if len(bboxes) > 0:
        frame_with_bboxes = draw_bbox(frame.copy(), bboxes)

        first_bbox = bboxes[0]
        cropped_image = crop_image(frame, first_bbox)
        ocr_results = perform_ocr(cropped_image)

        if ocr_results:
            for result in ocr_results:
                for item in result['data']:
                    detected_text += item['text'] + " "

        if not detected_text.strip():
            detected_text = "None"

        return detected_text.strip(), frame_with_bboxes
    return None, None

def send_data_to_url(detected_text, image, external_url):
    if detected_text != "None":
        image_bytes = encode_image_to_bytes(image)  
        files = {
            'detected_text': (None, detected_text), 
            'image': ('frame.jpg', image_bytes, 'image/jpeg') 
        }
        try:
            response = requests.post(external_url, files=files, timeout=10)
            if response.status_code not in [200, 201]:
                print(f"Failed to send data: {response.status_code}, {response.text}")
            else:
                print(f"Data sent successfully: {response.status_code}, {response.text}")
        except requests.Timeout:
            print(f"Error: Request to {external_url} timed out")
        except requests.RequestException as e:
            print(f"Request error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    else:
        print(f"No valid text detected, skipping sending data for this frame.")

def process_rtsp_stream(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print('Error: Unable to open RTSP stream.')
        return

    frame_count = 0
    last_detection_frame = -101  

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count - last_detection_frame > 100:
            detected_text, processed_frame = process_frame(frame)

            if processed_frame is not None:  
                print(f"Truck detected in frame {frame_count}. Processing...")
                send_data_to_url(detected_text, processed_frame, external_url)
                last_detection_frame = frame_count  

    cap.release()

if __name__ == "__main__":
    process_rtsp_stream(rtsp_url)
