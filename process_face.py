import os
import cv2


def detect_face_open_cv_dnn(net, frame, conf_threshold=0.5):
    frame_opencv_dnn = frame.copy()
    frame_height = frame_opencv_dnn.shape[0]
    frame_width = frame_opencv_dnn.shape[1]
    blob = cv2.dnn.blobFromImage(frame_opencv_dnn, 1.0, (300, 300), [104, 117, 123], False, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame_opencv_dnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height / 350)), 8)
    return frame_opencv_dnn, bboxes


# Khoi tao moduls phat hien khuon mat
model_file = "models/opencv_face_detector_uint8.pb"
config_file = "models/opencv_face_detector.pbtxt"
net = cv2.dnn.readNetFromTensorflow(model_file, config_file)

path = "/home/thanh/mayhocnangcao/face/"
path_result = "/home/thanh/mayhocnangcao/result"

couter = 0
for file_name in os.listdir(path):
    path_file = "{}/{}".format(path, file_name)
    if os.path.isfile(path_file):
        img = cv2.imread(path_file)
        res_image, lst_str = detect_face_open_cv_dnn(net, img)
        for str in lst_str:
            couter += 1
            x1, y1, x2, y2 = str
            if ((x2 - x1) < 70) or ((y2 - y1) < 70):
                continue
            cv2.imwrite("{}/{}.jpg".format(path_result, couter), img[y1:y2, x1:x2])
