#
# Lincoln Hack 2019 Raspberry Pi Open CV Face Recognition.
#
# Based on the code by PyImageSearch here...
# 
# https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/
#

# USAGE
# python recognize_video.py --detector face_detection_model \
#	--embedding-model openface_nn4.small2.v1.t7 \
#	--recognizer output/recognizer.pickle \
#	--le output/le.pickle

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from multiprocessing import Process
from multiprocessing import Queue
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

import serial
import time

import iotc
from iotc import IOTConnectType, IOTLogLevel
from random import randint
import serial
import time

def classify_frame(inputQueue, outputQueue):
	# keep looping
	while True:
		# check to see if there is a frame in our input queue
		if not inputQueue.empty():
			# grab the frame from the input queue, resize it, and
			# construct a blob from it
			frame = inputQueue.get()
			# construct a blob from the image
			imageBlob = cv2.dnn.blobFromImage(
				cv2.resize(frame, (300, 300)), 1.0, (300, 300),
				(104.0, 177.0, 123.0), swapRB=False, crop=False)
			# apply OpenCV's deep learning-based face detector to localize
			# faces in the input image
			detector.setInput(imageBlob)
			detections = detector.forward()
			# write the detections to the output queue
			outputQueue.put(detections)
		
		key = cv2.waitKey(1) & 0xFF

		if key == ord("q"):
			break

# Set up the Serial Port
# ser = serial.Serial('/dev/ttyUSB0', 9600)

# Air Quality and Zombie Readings
airQuality = 0
zombieDetected = 0

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()

# Setup the IoT Central Connection
deviceId = "[device id]"
scopeId = "[scope id]"
deviceKey = "[device key]"

iotc = iotc.Device(scopeId, deviceKey, deviceId, IOTConnectType.IOTC_CONNECT_SYMM_KEY)
iotc.setLogLevel(IOTLogLevel.IOTC_LOGGING_API_ONLY)

# Serial Comms Settings
gCanSend = False
gCounter = 0

data = ""

# initialize the input queue (frames), output queue (detections),
# and the list of actual detections returned by the child process
inputQueue = Queue(maxsize=1)
outputQueue = Queue(maxsize=1)
detections = None

# construct a child process *indepedent* from our main process of
# execution
print("[INFO] starting process...")
p = Process(target=classify_frame, args=(inputQueue,
	outputQueue,))
p.daemon = True
p.start()

# IoT Central is Connected
def onconnect(info):
	global gCanSend
	print("- [onconnect] => status:" + str(info.getStatusCode()))
	if info.getStatusCode() == 0:
		if iotc.isConnected():
			gCanSend = True

# IoT Central Message Sent
def onmessagesent(info):
	print("\t- [onmessagesent] => " + str(info.getPayload()))

# IoT Central Command Recieved
def oncommand(info):
	print("- [oncommand] => " + info.getTag() + " => " + str(info.getPayload()))

# IoT Central Settings Updated
def onsettingsupdated(info):
	print("- [onsettingsupdated] => " + info.getTag() + " => " + info.getPayload())

# Setup IoT Central Event Handlers
iotc.on("ConnectionStatus", onconnect)
iotc.on("MessageSent", onmessagesent)
iotc.on("Command", oncommand)
iotc.on("SettingsUpdated", onsettingsupdated)

# Connect to IoT Central
#iotc.connect()

# loop over frames from the video file stream
while True:

	# data = ser.readline()[:-2] #the last bit gets rid of the new-line chars
		
	if data:
		values = data.decode().split(',')

		airQuality = values[1].split('=')[1]

		print(values)
	
	# If IoT Central is Connected, then send our Telemetry.
	if iotc.isConnected():
		iotc.doNext() # do the async work needed to be done for MQTT

		if gCanSend == True:
			if gCounter % 5 == 0:
				gCounter = 0
				print("Sending telemetry..")
				iotc.sendTelemetry("{ \
\"temp\": " + str(randint(20, 45)) + ", \
\"airQuality\": " + str(airQuality) + ", \
\"zombieDetected\": " + str(zombieDetected) + ", \
\"accelerometerX\": " + str(randint(2, 15)) + ", \
\"accelerometerY\": " + str(randint(3, 9)) + ", \
\"accelerometerZ\": " + str(randint(1, 4)) + "}")

			gCounter += 1

	frame = vs.read()

	# resize the frame to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
	frame = imutils.resize(frame, width=600)
	(h, w) = frame.shape[:2]
		
	# if the input queue *is* empty, give the current frame to
	# classify
	if inputQueue.empty():
		inputQueue.put(frame)

	# if the output queue *is not* empty, grab the detections
	if not outputQueue.empty():
		detections = outputQueue.get()

	# check to see if our detectios are not None (and if so, we'll
	# draw the detections on the frame)
	if detections is not None:

		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections
			if confidence > args["confidence"]:
				# compute the (x, y)-coordinates of the bounding box for
				# the face
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# extract the face ROI
				face = frame[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]

				# ensure the face width and height are sufficiently large
				if fW < 20 or fH < 20:
					continue

				# construct a blob for the face ROI, then pass the blob
				# through our face embedding model to obtain the 128-d
				# quantification of the face
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
					(96, 96), (0, 0, 0), swapRB=True, crop=False)
				embedder.setInput(faceBlob)
				vec = embedder.forward()

				# perform classification to recognize the face
				preds = recognizer.predict_proba(vec)[0]
				j = np.argmax(preds)
				proba = preds[j]
				name = le.classes_[j]

				# If We detect a zombie, then record it and send it to the Arduino
				if name == "zombie":
					zombieDetected = 1
					#ser.write(b'1')
				else:
					zombieDetected = 0
					#ser.write(b'0')

				# draw the bounding box of the face along with the
				# associated probability
				text = "{}: {:.2f}%".format(name, proba * 100)
				y = startY - 10 if startY - 10 > 10 else startY + 10
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 0, 255), 2)
				cv2.putText(frame, text, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	# update the FPS counter
	fps.update()

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()