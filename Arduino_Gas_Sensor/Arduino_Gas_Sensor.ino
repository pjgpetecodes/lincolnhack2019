#include <Servo.h>

Servo servo_6;

int ledRed = 7;
int ledGreen = 8;
int gasSensorA = A5; // Analog pin 0 will be called 'sensor'
int sensorValue = 0; // Set the initial sensorValue to 0

int pushButton = 5;
int sensorOk = 1;
int noZombies = 1;
String serial_string = "";
char recieved_char;
int recieved_int;

int sendCounter = 0;

// The setup routine runs once when you press reset
void setup() {
    Serial.begin(9600); // Initialize serial communication at 9600 bits per second
    pinMode(gasSensorA, INPUT);

    pinMode(ledRed, OUTPUT);
    pinMode(ledGreen, OUTPUT);

    pinMode(pushButton, INPUT);

    servo_6.attach(6);

}

// The loop routine runs over and over again forever
void loop() {

    int buttonState = digitalRead(pushButton);

    //check  air quality
    sensorValue = analogRead(gasSensorA); // Read the input on analog pin 0 (named 'sensor') 
    serial_string = (String)
    "button=" + buttonState + ",sensorValue=" + sensorValue;

    if (Serial.available()) { //From RPi to Arduino
        recieved_int = Serial.parseInt(); //conveting the value of chars to integer

        if (recieved_int > 0) {
            noZombies = 0;
        } else {
            noZombies = 1;
        }

    }

    sendCounter++;

    if (sendCounter == 5000) {
        Serial.println(serial_string);
        sendCounter = 0;
    }

    if (sensorValue > 180 || noZombies == 0) // If sensorValue is greater 180 then it's not ok to go outside.
    {
        sensorOk = 0;
        digitalWrite(ledRed, HIGH);
        digitalWrite(ledGreen, LOW);
    } else {
        sensorOk = 1;
        digitalWrite(ledRed, LOW);
        digitalWrite(ledGreen, HIGH);
    }

    if (buttonState == 1 && sensorOk == 1 && noZombies == 1) {
        // If the button is pressed, air quality ok and no zombies, open the door. 

        servo_6.write(0);

        //give them 5 seconds to leave then go outside.  
        delay(5000);
        servo_6.write(90);

    } else {

        //if any of the above are not ok, don't go outside!!!
        servo_6.write(90);

    }

}
