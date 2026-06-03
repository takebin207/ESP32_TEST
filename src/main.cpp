#include <Arduino.h>
#include <ESP32Servo.h>
#include <ESP32PWM.h>

namespace {
Servo sorterServo;

constexpr int kServoPin = 40;
constexpr int kServoMinPulseUs = 500;
constexpr int kServoMaxPulseUs = 2400;

constexpr int kCenterAngle = 90;
constexpr int kLeftAngle = 180;
constexpr int kRightAngle = 0;

String serialBuffer;

void moveServoTo(int angle, const char *label) {
  sorterServo.write(angle);
  Serial.printf("Servo -> %s (%d deg)\n", label, angle);
}

void sweepServo() {
  Serial.println("Servo sweep test start");
  moveServoTo(kRightAngle, "RIGHT");
  delay(900);
  moveServoTo(kCenterAngle, "CENTER");
  delay(900);
  moveServoTo(kLeftAngle, "LEFT");
  delay(900);
  moveServoTo(kCenterAngle, "CENTER");
  Serial.println("Servo sweep test done");
}

void handleCommand(String command) {
  command.trim();
  command.toUpperCase();

  if (command.length() == 0) {
    return;
  }

  if (command == "ORGANIC" || command == "LEFT") {
    moveServoTo(kLeftAngle, "ORGANIC / LEFT");
  } else if (command == "INORGANIC" || command == "RIGHT") {
    moveServoTo(kRightAngle, "INORGANIC / RIGHT");
  } else if (command == "CENTER" || command == "RESET") {
    moveServoTo(kCenterAngle, "CENTER");
  } else if (command == "SWEEP" || command == "TEST") {
    sweepServo();
  } else {
    Serial.printf("Unknown command: %s\n", command.c_str());
  }
}
}  // namespace

void setup() {
  Serial.begin(115200);

  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);

  sorterServo.setPeriodHertz(50);
  sorterServo.attach(kServoPin, kServoMinPulseUs, kServoMaxPulseUs);
  moveServoTo(kCenterAngle, "CENTER");

  Serial.println("Waste sorter ready.");
  Serial.printf("Servo signal pin: GPIO%d\n", kServoPin);
  Serial.println("Commands: ORGANIC, INORGANIC, CENTER, SWEEP");
}

void loop() {
  while (Serial.available() > 0) {
    char incoming = static_cast<char>(Serial.read());

    if (incoming == '\n' || incoming == '\r') {
      if (serialBuffer.length() > 0) {
        handleCommand(serialBuffer);
        serialBuffer = "";
      }
      continue;
    }

    serialBuffer += incoming;
  }
}
