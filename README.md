This is the Ctrl+Alt+Zombie entry for Lincoln Hack 2019.

You can find the Arduino section online here at TinkerCad;

https://www.tinkercad.com/things/7cLzzE9TwKu

The Raspberry Pi Face Recognition Code is based on the PyImageSearch code which can be found here;

https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/

Add your IoT Central DeviceID, Scope ID and Primary Key to the recognize_video.py file to connect to IoT Central.

Adjust the Serial port settings on line 36 to connect the Pi to the Arduino;

ser = serial.Serial('/dev/ttyUSB0', 9600)

