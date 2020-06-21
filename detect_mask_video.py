# USAGE
# python detect_mask_video.py
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
def mask_detect(output, detected_faceNet, maskNet):
	(h, w) = output.shape[:2]
	blob = cv2.dnn.blobFromImage(output, 1.0, (300, 300),(104.0, 177.0, 123.0))
	detected_faceNet.setInput(blob)
	detections = detected_faceNet.forward()
	detected_faces = []
	locations = []
	predicts = []
	for i in range(0, detections.shape[2]):
		conf = detections[0, 0, i, 2]
		if conf > args["conf"]:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(start_x, start_y, end_x, end_y) = box.astype("int")
			(start_x, start_y) = (max(0, start_x), max(0, start_y))
			(end_x, end_y) = (min(w - 1, end_x), min(h - 1, end_y))
			detected_face = output[start_y:end_y, start_x:end_x]
			detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2RGB)
			detected_face = cv2.resize(detected_face, (224, 224))
			detected_face = img_to_array(detected_face)
			detected_face = preprocess_input(detected_face)
			detected_face = np.expand_dims(detected_face, axis=0)
			detected_faces.append(detected_face)
			locations.append((start_x, start_y, end_x, end_y))

	if len(detected_faces)>0:
		predicts = maskNet.predict(detected_faces)
	return (locations, predicts)

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--detected_face", type=str,default="face_detector")
parser.add_argument("-m", "--model", type=str,default="mask.model")
parser.add_argument("-c", "--conf", type=float, default=0.5)
args = vars(parser.parse_args())
prototxtPath = os.path.sep.join([args["detected_face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["detected_face"],"res10_300x300_ssd_iter_140000.caffemodel"])
detected_faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model(args["model"])

video = VideoStream(src=0).start()
time.sleep(2.0)
while True:
	output = video.read()
	output = imutils.resize(output, width=600)
	(locations, predicts) = mask_detect(output, detected_faceNet, maskNet)
	for (box, pred) in zip(locations, predicts):
		(start_x, start_y, end_x, end_y) = box
		(mask, no_mask) = pred
		accuracy = "Accuracy"
		label = "mask" if mask > no_mask else "no Mask"
		color = (255, 0,0) if label == "mask" else (0,0,255)
		label = "{}: {}:{:.2f}%".format(label,accuracy, max(mask, no_mask) * 100)
		cv2.putText(output, label, (start_x, start_y -20),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
		cv2.rectangle(output, (start_x, start_y), (end_x, end_y), color, 2)
	cv2.imshow("Output", output)
	key = cv2.waitKey(1) & 0xFF 
	if key == ord("x"):
		break
cv2.destroyAllWindows()
video.stop()
