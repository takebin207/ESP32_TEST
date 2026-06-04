#include <Arduino.h>

namespace {
#ifndef SERVO_SIGNAL_PIN
#define SERVO_SIGNAL_PIN 40
#endif

constexpr int kServoPin = SERVO_SIGNAL_PIN;
constexpr int kServoFrequencyHz = 50;
constexpr int kServoResolutionBits = 14;
constexpr int kServoMinPulseUs = 500;
constexpr int kServoMaxPulseUs = 2500;
constexpr int kServoPeriodUs = 1000000 / kServoFrequencyHz;
constexpr int kServoPins[] = {kServoPin, 4, 5, 6};
constexpr int kServoPinCount = sizeof(kServoPins) / sizeof(kServoPins[0]);
constexpr int kRgbPin = 48;

constexpr int kCenterAngle = 90;
constexpr int kLeftAngle = 180;
constexpr int kRightAngle = 0;

String serialBuffer;

uint32_t pulseToDuty(int pulseUs) {
  constexpr uint32_t maxDuty = (1UL << kServoResolutionBits) - 1;
  return (static_cast<uint32_t>(pulseUs) * maxDuty) / kServoPeriodUs;
}

int angleToPulseUs(int angle) {
  angle = constrain(angle, 0, 180);
  return map(angle, 0, 180, kServoMinPulseUs, kServoMaxPulseUs);
}

void moveServoTo(int angle, const char *label) {
  const int pulseUs = angleToPulseUs(angle);
  const uint32_t duty = pulseToDuty(pulseUs);

  for (int channel = 0; channel < kServoPinCount; channel++) {
    ledcWrite(channel, duty);
  }

  Serial.printf(
      "Servo -> %s (%d deg, %d us, signal GPIO%d/GPIO4/GPIO5/GPIO6)\n",
      label,
      angle,
      pulseUs,
      kServoPin);
}

void sweepServo() {
  Serial.println("Servo sweep test start");
  moveServoTo(kRightAngle, "RIGHT");
  delay(1000);
  moveServoTo(kCenterAngle, "CENTER");
  delay(1000);
  moveServoTo(kLeftAngle, "LEFT");
  delay(1000);
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
  } else if (command == "PIN") {
    Serial.printf(
        "Servo signal pins: GPIO%d, GPIO4, GPIO5, GPIO6; %d Hz; pulse %d-%d us\n",
        kServoPin,
        kServoFrequencyHz,
        kServoMinPulseUs,
        kServoMaxPulseUs);
  } else {
    Serial.printf("Unknown command: %s\n", command.c_str());
  }
}
}  // namespace

void setup() {
  Serial.begin(115200);
  delay(500);

  pinMode(kRgbPin, OUTPUT);
  neopixelWrite(kRgbPin, 0, 0, 0);

  for (int channel = 0; channel < kServoPinCount; channel++) {
    pinMode(kServoPins[channel], OUTPUT);
    ledcSetup(channel, kServoFrequencyHz, kServoResolutionBits);
    ledcAttachPin(kServoPins[channel], channel);
  }

  moveServoTo(kCenterAngle, "CENTER");

  Serial.println("Waste sorter ready.");
  Serial.printf("Servo primary signal pin: GPIO%d\n", kServoPin);
  Serial.println("Servo diagnostic pins also active: GPIO4, GPIO5, GPIO6");
  Serial.println("Servo driver: LEDC direct PWM");
  Serial.println("Commands: ORGANIC, INORGANIC, CENTER, SWEEP, PIN");
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
