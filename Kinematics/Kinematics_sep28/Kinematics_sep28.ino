#include <Arduino.h>
#include <Servo.h>

//Servos
Servo body;
Servo headPan;
Servo headTilt;
Servo shoulder;
Servo elbow;
Servo gripper;

//Init position of all servos
const int servo_pins[] = {3, 5, 6, 9, 10, 11};

const int pos_init[] = {1700, 1500, 2000, 2200, 1650, 1600};
int curr_pos[6];
int new_servo_val[6];

const int pos_min[] = {560, 550, 950, 750, 550, 550};
const int pos_max[] = {2330, 2340, 2400, 2200, 2400, 2150};

const int pos_move[] = {2200, 1500, 2000, 1100, 2300, 1600};

// Variables for FSM and emotions
enum Emotion {HAPPY, DANCY, TALKING, BORED};  // States for FSM
Emotion currentEmotion = BORED;              // Default state

//Servo update function
void servo_body_ex(const int new_pos) {

  int diff, steps, now, CurrPwm, NewPwm, delta = 6;

  //current servo value
  now = curr_pos[0];
  CurrPwm = now;
  NewPwm = new_pos;

  /* determine interation "diff" from old to new position */
  diff = (NewPwm - CurrPwm)/abs(NewPwm - CurrPwm); // Should return +1 if NewPwm is bigger than CurrPwm, -1 otherwise.
  steps = abs(NewPwm - CurrPwm);
  delay(10);

  for (int i = 0; i < steps; i += delta) {
    now = now + delta*diff;
    body.writeMicroseconds(now);
    delay(20);
  }
  curr_pos[0] = now;
  delay(10);
}

//Servo update function
void servo_neck_pan(const int new_pos) {

  int diff, steps, now, CurrPwm, NewPwm, delta = 6;

  //current servo value
  now = curr_pos[1];
  CurrPwm = now;
  NewPwm = new_pos;

  /* determine interation "diff" from old to new position */
  diff = (NewPwm - CurrPwm)/abs(NewPwm - CurrPwm); // Should return +1 if NewPwm is bigger than CurrPwm, -1 otherwise.
  steps = abs(NewPwm - CurrPwm);
  delay(10);

  for (int i = 0; i < steps; i += delta) {
    now = now + delta*diff;
    headPan.writeMicroseconds(now);
    delay(20);
  }
  curr_pos[1] = now;
  delay(10);
}

//Servo update function
void servo_neck_tilt(const int new_pos) {

  int diff, steps, now, CurrPwm, NewPwm, delta = 6;

  //current servo value
  now = curr_pos[2];
  CurrPwm = now;
  NewPwm = new_pos;

  /* determine interation "diff" from old to new position */
  diff = (NewPwm - CurrPwm)/abs(NewPwm - CurrPwm); // Should return +1 if NewPwm is bigger than CurrPwm, -1 otherwise.
  steps = abs(NewPwm - CurrPwm);
  delay(10);

  for (int i = 0; i < steps; i += delta) {
    now = now + delta*diff;
    headTilt.writeMicroseconds(now);
    delay(20);
  }
  curr_pos[2] = now;
  delay(10);
}

//Servo update function
void servo_shoulder(const int new_pos) {

  int diff, steps, now, CurrPwm, NewPwm, delta = 10;

  //current servo value
  now = curr_pos[3];
  CurrPwm = now;
  NewPwm = new_pos;

  /* determine interation "diff" from old to new position */
  diff = (NewPwm - CurrPwm)/abs(NewPwm - CurrPwm); // Should return +1 if NewPwm is bigger than CurrPwm, -1 otherwise.
  steps = abs(NewPwm - CurrPwm);
  delay(10);

  for (int i = 0; i < steps; i += delta) {
    now = now + delta*diff;
    shoulder.writeMicroseconds(now);
    delay(20);
  }
  curr_pos[3] = now;
  delay(10);
}

//Servo update function
void servo_elbow(const int new_pos) {

  int diff, steps, now, CurrPwm, NewPwm, delta = 10;

  //current servo value
  now = curr_pos[4];
  CurrPwm = now;
  NewPwm = new_pos;

  /* determine interation "diff" from old to new position */
  diff = (NewPwm - CurrPwm)/abs(NewPwm - CurrPwm); // Should return +1 if NewPwm is bigger than CurrPwm, -1 otherwise.
  steps = abs(NewPwm - CurrPwm);
  delay(10);

  for (int i = 0; i < steps; i += delta) {
    now = now + delta*diff;
    elbow.writeMicroseconds(now);
    delay(20);
  }
  curr_pos[4] = now;
  delay(10);
}

//Servo update function
void servo_gripper_ex(const int new_pos) {

  int diff, steps, now, CurrPwm, NewPwm, delta = 6;

  //current servo value
  now = curr_pos[5];
  CurrPwm = now;
  NewPwm = new_pos;

  /* determine interation "diff" from old to new position */
  diff = (NewPwm - CurrPwm)/abs(NewPwm - CurrPwm); // Should return +1 if NewPwm is bigger than CurrPwm, -1 otherwise.
  steps = abs(NewPwm - CurrPwm);
  delay(10);

  for (int i = 0; i < steps; i += delta) {
    now = now + delta*diff;
    gripper.writeMicroseconds(now);
    delay(20);
  }
  curr_pos[5] = now;
  delay(10);
}

// Happy state: wave end-effector and do fake button press
void happyStateAction() {
  int bodyPos = 2200;
  Serial.println("Switching to Happy playlist");
  pressButton(bodyPos);  // Perform fake button press
  waveEndEffector();  // Perform wave motion
}

// Dancy state: fake button press, more dynamic motion
void dancyStateAction() {
  int bodyPos = 2250;
  Serial.println("Switching to Dancy playlist");
  pressButton(bodyPos);
}

// Talking state
void talkingStateAction() {
  int bodyPos = 2150;
  Serial.println("Switching to Talking playlist");
  pressButton(bodyPos);
}

// Bored state
void boredStateAction() {
  int bodyPos = 2100;
  Serial.println("Switching to Bored playlist");
  pressButton(bodyPos);
}

// Fake button press motion
void pressButton(int posButton) {
  servo_shoulder(2100);
  delay(100);
  servo_elbow(2350);
  delay(100);
  servo_body_ex(posButton);
  servo_elbow(2150);
  delay(100);
  servo_elbow(2350);
  delay(100);
  servo_body_ex(pos_init[0]);
  delay(100);
  servo_elbow(pos_init[4]);
  delay(100);
}

// Wave end-effector motion
void waveEndEffector() {
  for(int i = 0; i < 3; i++) {
    servo_shoulder(1000);
    delay(100);
    servo_shoulder(1650);
    delay(100);
  }
  servo_shoulder(pos_init[3]);
}

void dance(){
  servo_shoulder(1900);
  delay(100);
  for (int j = 0; j < 3; j++){
    servo_elbow(2350);
    delay(100);
    servo_body_ex(750);
    delay(100);
    servo_elbow(pos_init[4]);
    delay(100);
    servo_body_ex(1200);
    delay(100);
    servo_elbow(2350);
    delay(100);
    servo_body_ex(pos_init[0]);
    delay(100);
    servo_elbow(pos_init[4]);
    delay(100);
  }
  servo_shoulder(pos_init[3]);
  delay(100);
}

void boredMovement(){
  servo_neck_pan(1200);  
  delay(500);  
  servo_neck_pan(pos_init[1]);  
  delay(500);
  servo_neck_pan(1800);  
  delay(500);
  servo_neck_pan(pos_init[1]);  
  delay(500);
  servo_elbow(2350);
  delay(100);
  servo_shoulder(1850);
  delay(100);
  servo_body_ex(1850);
  delay(100);
  servo_gripper_ex(600);
  delay(100);
  servo_elbow(2150);
  delay(100);
  servo_gripper_ex(800);
  delay(100);
  servo_body_ex(700);
  delay(500);
  servo_body_ex(1850);
  delay(100);
  servo_gripper_ex(600);
  delay(100);
  servo_elbow(2350);
  delay(100);
  servo_gripper_ex(pos_init[5]);
  delay(100);
  servo_body_ex(pos_init[0]);
  delay(100);
  servo_elbow(pos_init[4]);
  delay(100);
  servo_shoulder(pos_init[3]);
  delay(100);
}

// Function to handle state changes
void updateEmotion(String emotionStr) {
  Emotion newEmotion;

  // Map the string to corresponding emotion and handle change
  if (emotionStr == "happy") {
    newEmotion = HAPPY;
  } else if (emotionStr == "dancy") {
    newEmotion = DANCY;
  } else if (emotionStr == "talking") {
    newEmotion = TALKING;
  } else if (emotionStr == "bored") {
    newEmotion = BORED;
  }

  if (currentEmotion != newEmotion){
    currentEmotion = newEmotion;
    // Perform actions based on the new emotion
    switch (currentEmotion) {
      case HAPPY:
        happyStateAction();
        break;
      case DANCY:
        dancyStateAction();
        break;
      case TALKING:
        talkingStateAction();
        break;
      case BORED:
        boredStateAction();
        break;
    }
  }
}

void setup() {

  Serial.begin(57600); // Starts the serial communication

	//Attach each joint servo
	//and write each init position
  body.attach(servo_pins[0]);
  body.writeMicroseconds(pos_init[0]);
  
  headPan.attach(servo_pins[1]);
  headPan.writeMicroseconds(pos_init[1]);
  
  headTilt.attach(servo_pins[2]);
  headTilt.writeMicroseconds(pos_init[2]);

  shoulder.attach(servo_pins[3]);
	shoulder.writeMicroseconds(pos_init[3]);

	elbow.attach(servo_pins[4]);
	elbow.writeMicroseconds(pos_init[4]);
	
	gripper.attach(servo_pins[5]);
  gripper.writeMicroseconds(pos_init[5]);

  //Initilize curr_pos and new_servo_val vectors
  byte i;
  for (i=0; i<(sizeof(pos_init)/sizeof(int)); i++){
    curr_pos[i] = pos_init[i];
    new_servo_val[i] = curr_pos[i];
  }

	delay(2000);
}

void loop() {

  // Check if data is available on the serial port
  if (Serial.available() > 0) {
    // Read the incoming string
    String moodStr = Serial.readStringUntil('\n');
    moodStr.trim();  // Remove any extraneous whitespace or newline characters
    updateEmotion(moodStr);
  }

  delay(2000);  // Delay between checks -> Find the optima
}