import os
import cv2
import time
import requests
import paddlehub as hub
from ultralytics import YOLO

yolo_model = YOLO('hull.pt')
ocr = hub.Module(name="ch_pp-ocrv3")
external_url = 'https://dashboard-kpp.kecilin.id/api/v1/hull_number/store'
rtsp_url = "rtsp://localhost:8554/PITSTOP"


UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

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

def save_image(image, filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    cv2.imwrite(file_path, image)
    return file_path

def process_frame(frame, frame_count):
    hull_data = []
    detected_text = ""

    if frame_count % 25 == 0:
        bboxes = detect_hull(frame)
        if len(bboxes) > 0:
            frame_with_bboxes = draw_bbox(frame.copy(), bboxes)

            for bbox in bboxes:
                x, y, w, h = bbox
                hull_data.append({
                    "xmin": int(x - w / 2),
                    "ymax": int(y + h / 2),
                    "width": int(w),
                    "height": int(h)
                })

            first_bbox = bboxes[0]
            cropped_image = crop_image(frame, first_bbox)
            ocr_results = perform_ocr(cropped_image)

            if ocr_results:
                for result in ocr_results:
                    for item in result['data']:
                        detected_text += item['text'] + " "

            if not detected_text.strip():
                detected_text = "None"

            timestamp = int(time.time())
            filename = f'frame_{frame_count}_{timestamp}.jpg'
            save_image(frame_with_bboxes, filename)

    return hull_data, detected_text.strip()


def send_data_to_url(hull_data, detected_text, external_url):
    data = {
        "data": {
            "hull_num": hull_data,
            "detected_text": detected_text
        }
    }

    try:
        response = requests.post(url=external_url, json=data, timeout=10)
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

def process_rtsp_stream(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print('Error: Unable to open RTSP stream.')
        return

    hull_data = []
    detected_text = ""
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        hulls, text = process_frame(frame, frame_count)

        if hulls:
            hull_data.extend(hulls)
        if text:
            detected_text += text

        # if len(hull_data) >= 15:
        #     break

    cap.release()
    send_data_to_url(hull_data, detected_text.strip(), external_url)

if __name__ == "__main__":
    process_rtsp_stream(rtsp_url)
