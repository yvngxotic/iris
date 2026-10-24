#define DRIVE_MODE 0
#define MAX_PWM 200
#define SLEW_STEP 45
#define WATCHDOG_MS 600
#define LOOP_MS 20

#if DRIVE_MODE == 1
#include <Servo.h>
Servo steerServo;
#endif

#if DRIVE_MODE == 0
const uint8_t PIN_ENA = 9, PIN_IN1 = 7, PIN_IN2 = 8;
const uint8_t PIN_ENB = 10, PIN_IN3 = 5, PIN_IN4 = 6;
const bool INVERT_LEFT = false;
const bool INVERT_RIGHT = false;
#else
const uint8_t PIN_SERVO = 3, PIN_ESC = 6;
const int SERVO_CENTER = 90, SERVO_RANGE = 28;
#endif

int targetLeft = 0, targetRight = 0;
int currentLeft = 0, currentRight = 0;
unsigned long lastCommandMs = 0;
unsigned long lastLoopMs = 0;
unsigned long watchdogTrips = 0;
unsigned long commandCount = 0;
char line[48];
uint8_t lineLen = 0;

int clampPwm(int value) {
  if (value > MAX_PWM) return MAX_PWM;
  if (value < -MAX_PWM) return -MAX_PWM;
  return value;
}

int slew(int current, int target) {
  int delta = target - current;
  if (delta > SLEW_STEP) return current + SLEW_STEP;
  if (delta < -SLEW_STEP) return current - SLEW_STEP;
  return target;
}

void applyMotors(int left, int right) {
#if DRIVE_MODE == 0
  int l = INVERT_LEFT ? -left : left;
  int r = INVERT_RIGHT ? -right : right;
  digitalWrite(PIN_IN1, l >= 0 ? HIGH : LOW);
  digitalWrite(PIN_IN2, l >= 0 ? LOW : HIGH);
  digitalWrite(PIN_IN3, r >= 0 ? HIGH : LOW);
  digitalWrite(PIN_IN4, r >= 0 ? LOW : HIGH);
  analogWrite(PIN_ENA, abs(l));
  analogWrite(PIN_ENB, abs(r));
#else
  int steer = SERVO_CENTER + (int)constrain((right - left) * SERVO_RANGE / MAX_PWM, -SERVO_RANGE, SERVO_RANGE);
  int throttle = constrain((left + right) / 2, 0, MAX_PWM);
  steerServo.write(steer);
  analogWrite(PIN_ESC, throttle);
#endif
}

void stopMotors() {
  targetLeft = 0;
  targetRight = 0;
  currentLeft = 0;
  currentRight = 0;
  applyMotors(0, 0);
}

void reply(const __FlashStringHelper *text) {
  Serial.println(text);
}

void handleLine(char *text) {
  if (text[0] == 'D' || text[0] == 'd') {
    int left = 0, right = 0;
    if (sscanf(text + 1, "%d %d", &left, &right) == 2) {
      targetLeft = clampPwm(left);
      targetRight = clampPwm(right);
      lastCommandMs = millis();
      commandCount++;
      if (commandCount % 50 == 0) {
        Serial.print(F("OK "));
        Serial.println(millis());
      }
    }
  } else if (text[0] == 'S' || text[0] == 's') {
    stopMotors();
    lastCommandMs = millis();
  } else if (strncmp(text, "PING", 4) == 0) {
    reply(F("PONG"));
  }
}

void readSerial() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (lineLen > 0) {
        line[lineLen] = '\0';
        handleLine(line);
        lineLen = 0;
      }
    } else if (lineLen < sizeof(line) - 1) {
      line[lineLen++] = c;
    }
  }
}

void setup() {
  Serial.begin(115200);
#if DRIVE_MODE == 0
  pinMode(PIN_ENA, OUTPUT);
  pinMode(PIN_ENB, OUTPUT);
  pinMode(PIN_IN1, OUTPUT);
  pinMode(PIN_IN2, OUTPUT);
  pinMode(PIN_IN3, OUTPUT);
  pinMode(PIN_IN4, OUTPUT);
#else
  steerServo.attach(PIN_SERVO);
  steerServo.write(SERVO_CENTER);
  pinMode(PIN_ESC, OUTPUT);
  analogWrite(PIN_ESC, 0);
#endif
  stopMotors();
  lastCommandMs = millis();
  delay(200);
  reply(F("READY"));
}

void loop() {
  readSerial();
  unsigned long now = millis();

  if (now - lastCommandMs > WATCHDOG_MS) {
    if (targetLeft != 0 || targetRight != 0) {
      watchdogTrips++;
      Serial.print(F("T wd "));
      Serial.println(watchdogTrips);
    }
    targetLeft = 0;
    targetRight = 0;
  }

  if (now - lastLoopMs >= LOOP_MS) {
    lastLoopMs = now;
    currentLeft = slew(currentLeft, targetLeft);
    currentRight = slew(currentRight, targetRight);
    applyMotors(currentLeft, currentRight);
  }
}
