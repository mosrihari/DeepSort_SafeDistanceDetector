import tensorflow as tf
import numpy as np
import cv2
from deep_sort.detection import Detection
from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from deep_sort.preprocessing import non_max_suppression
from tools import generate_detections as gdet
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import load_darknet_weights, convert_boxes
from yolov3_tf2.models import YoloV3
from absl import flags
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import sys
FLAGS = flags.FLAGS
FLAGS(sys.argv)
from _collections import deque


# Refer this https://github.com/basileroth75/covid-social-distancing-detection

#################################################################################################
## STEP 1: Define YOLO with weights from weights folder (FOR DETECTION)                         #
## STEP 2: Define feature extractor mars pb file which provides us feature from detection box   #
## STEP 3: Define a metric which compares between time steps to detect whether its a same ID    #
## STEP 4: Define Tracker which tracks the next time step and compares it                       #
#################################################################################################

# STEP 1
with open("data/labels/coco.names","rb") as f:
    class_names = f.read().decode()
    class_names = class_names.split("\r\n")
    class_names = class_names[:-1]
yolo = YoloV3(classes=len(class_names))
yolo.load_weights("weights/yolov3.tf")

# STEP 2
model_file_name = "model_data/mars-small128.pb"
encoder = gdet.create_box_encoder(model_file_name)

# STEP 3
cosine_threshold = 0.5
nn_budget = None
metric = nn_matching.NearestNeighborDistanceMetric("cosine",matching_threshold=cosine_threshold,
                                                   budget=nn_budget)
# STEP 4
tracker = Tracker(metric)

# Read the video and create a video writer
vid = cv2.VideoCapture('data/video/vid_short.mp4')

codec = cv2.VideoWriter_fourcc(*'XVID')
vid_fps =int(vid.get(cv2.CAP_PROP_FPS))
vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./data/video/results.avi', codec, vid_fps, (vid_width, vid_height))
pts = [deque(maxlen=30) for _ in range(1000)]
counter = []

while True:
    violate = set()
    bounding_box_list = []
    _, img = vid.read()
    if img is None:
        print('Completed')
        break
    centroids = []
    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in = tf.expand_dims(img_in, 0)
    img_in = transform_images(img_in, 416)


    boxes, scores, classes, nums = yolo.predict(img_in)

    classes = classes[0]
    names = []
    for i in range(len(classes)):
        names.append(class_names[int(classes[i])])
    names = np.array(names)
    converted_boxes = convert_boxes(img, boxes[0])
    features = encoder(img, converted_boxes)

    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                  zip(converted_boxes, scores[0], names, features)]

    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    nms_max_overlap = 0.8
    indices = non_max_suppression(boxs, classes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    tracker.predict()
    tracker.update(detections)

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bounding_box = track.to_tlbr()
        classified_names = track.get_class()
        if(classified_names == "person"):
            x1 = int(bounding_box[0])
            x2 = int(bounding_box[2])
            y1 = int(bounding_box[1])
            y2 = int(bounding_box[3])
            centerCoord = (int((x1+x2) / 2), int((y1+y2)/2))
            centroids.append(centerCoord)
            bounding_box_list.append(bounding_box)
    if(len(centroids) != 0):
        D = dist.cdist(centroids, centroids, metric="euclidean")
        print(D)
        for i in range(0, D.shape[0]):
            for j in range(i+1, D.shape[1]):
                if(D[i,j] < 130):
                    violate.add(i)
                    violate.add(j)
    for i, box in enumerate(bounding_box_list):
        (startX, startY, endX, endY) = box
        color = (0,255,0)
        if i in violate:
            color = (0,0,255)
        cv2.rectangle(img, (int(startX), int(startY)), (int(endX), int(endY)), color, 2)

    cv2.putText(img, f"VIOLATION : {len(violate)}", (200, 500), 0, 0.75, (255,0,0), 2)

    cv2.imshow('output', img)
    out.write(img)
    if cv2.waitKey(1) == ord('q'):
        break
vid.release()
out.release()
cv2.destroyAllWindows()
