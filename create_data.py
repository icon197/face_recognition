import sys
import os
import cv2

# Khai bao bien mac dinh
KEY_ESCAPE = 27
KEY_SPACE = 32
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
MAX_IMAGES = 700


# Phat hien khuon mat
def detect_face_open_cv_dnn(net, frame, conf_threshold=0.5):
    frame_opencv_dnn = frame.copy()
    frame_height = frame_opencv_dnn.shape[0]
    frame_width = frame_opencv_dnn.shape[1]
    blob = cv2.dnn.blobFromImage(frame_opencv_dnn, 1.0, (300, 300), [104, 117, 123], False, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    ajust = 0.17
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            ajust_x = int((x2 - x1) * ajust)
            ajust_y = int((y2 - y1) * ajust)
            x1 -= ajust_x
            x2 += ajust_x
            y1 -= int(ajust_y * 1.5)
            y2 += int(ajust_y * 0.5)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame_opencv_dnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height / 350)), 8)
    return frame_opencv_dnn, bboxes


if __name__ == "__main__":

    # Khoi tao
    user_name = input("Nhap ten: ")
    images_counter = 300

    # Khoi tao moduls phat hien khuon mat
    model_file = "models/opencv_face_detector_uint8.pb"
    config_file = "models/opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(model_file, config_file)

    # Nguon du lieu mac dinh la webcam
    source = 0
    if len(sys.argv) > 1:
        source = sys.argv[1]

    # Khoi tao stream video
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_HEIGHT)

    while True:
        has_frame, frame = cap.read()
        if (not has_frame) or (images_counter >= MAX_IMAGES):
            break

        out_opencv_dnn, bboxes = detect_face_open_cv_dnn(net, frame)

        cv2.imshow("Face Detection", out_opencv_dnn)

        # Bat su kien phim
        key = cv2.waitKey(10)
        if key == KEY_ESCAPE:
            break
        elif key == KEY_SPACE:
            images_counter += 1

            x1, y1, x2, y2 = bboxes[0]
            path_file = "data/{}/{}.jpg".format(user_name, images_counter)
            directory = os.path.dirname(path_file)

            if not os.path.exists(directory):
                os.makedirs(directory)

            cv2.imwrite(path_file, frame[y1:y2, x1:x2])
            print(bboxes[0])
            print("Da chup {}/{}".format(images_counter, MAX_IMAGES))
    cv2.destroyAllWindows()
    cap.release()
