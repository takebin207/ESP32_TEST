#include <Arduino.h>
#include <ESP32Servo.h>

Servo myservo;

void setup() {
  myservo.setPeriodHertz(50);       // chuẩn servo SG90
  myservo.attach(4, 500, 2400);     // dây signal nối GPIO4
}

void loop() {
  myservo.write(0);     // quay về 0°
  delay(1000);

  myservo.write(90);    // quay tới 90°
  delay(1000);

  myservo.write(180);   // quay tới 180°
  delay(1000);
}