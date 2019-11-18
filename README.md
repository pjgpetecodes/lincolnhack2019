This is the Ctrl+Alt+Zombie entry for Lincoln Hack 2019.

You can find the Arduino section online here at TinkerCad;

https://www.tinkercad.com/things/7cLzzE9TwKu

The Raspberry Pi Face Recognition Code is based on the PyImageSearch code which can be found here;

https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/

Add your IoT Central DeviceID, Scope ID and Primary Key to the recognize_video.py file to connect to IoT Central.

Adjust the Serial port settings on line 36 to connect the Pi to the Arduino;

ser = serial.Serial('/dev/ttyUSB0', 9600)

Start the Video Recognition using;

python recognize_video.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle

There's a second version which makes use of threads in order to speed up the framerate;

python recognize_video_thread.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle