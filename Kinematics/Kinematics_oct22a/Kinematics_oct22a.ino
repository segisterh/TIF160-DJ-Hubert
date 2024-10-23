#include <Arduino.h>
#include <Servo.h>

// Timing variables for testing state transitions
unsigned long lastStateTime = 0;
const unsigned long detectInterval = 500;  // Change emotion every 0.5 second

//Servos
Servo body;
Servo headPan;
Servo headTilt;
Servo shoulder;
Servo elbow;
Servo gripper;

//Init position of all servos
const int servo_pins[] = {3, 5, 6, 9, 10, 11};

const int pos_init[] = {1400, 1500, 2000, 2200, 1650, 1600};
int curr_pos[6];
int new_servo_val[6];

const int pos_min[] = {560, 550, 950, 750, 550, 550};
const int pos_max[] = {2330, 2340, 2400, 2200, 2400, 2150};

const int pos_move[] = {2200, 1500, 2000, 1100, 2300, 1600};

// Variables for FSM and emotions
enum Instruction {PRESS, CHEERS, WAVE, DANCE, WAITING};  // States for FSM
Instruction currentState = WAITING;              // Default state
int lastPosition = pos_init[0];

//Servo update function
void servo_body_ex(const int new_pos, int delta) {

  int diff, steps, now, CurrPwm, NewPwm;

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

int generate_body_pos(){
  int newPosition;
  do {
    newPosition = random(1800, 2100);  // generate a random number in [1900, 2330)
  } while (newPosition == lastPosition);  
  lastPosition = newPosition;
  return newPosition;
}

// Fake button press motion
void press_button() { 
  int posButton = generate_body_pos();  // Press the place randomly, different with last time
//  servo_shoulder(2200);
//  delay(100);
  servo_elbow(2350);
  delay(100);
  servo_body_ex(posButton, 6);
  servo_elbow(2050);
  delay(100);
  servo_elbow(2350);
  delay(100);
  servo_body_ex(pos_init[0], 6);
  delay(100);
  servo_elbow(pos_init[4]);
  delay(100);
}

// Wave end-effector motion
void wave_end_effector(int tempo) {   // -> different tempo -> wave1/wave2/wave3 
  servo_shoulder(1200);
  delay(tempo);
  for(int i = 0; i < 3; i++) {
    servo_elbow(2250);
    delay(tempo);
    servo_elbow(1150);
    delay(tempo);
  }
  servo_shoulder(pos_init[3]);
  delay(100);
  servo_elbow(pos_init[4]);
  delay(100);
}

void dance(){ 
  servo_shoulder(1900);
  delay(100);
  for (int j = 0; j < 3; j++){
    servo_elbow(2350);
    delay(10);
    servo_body_ex(600, 10);
    delay(10);
    servo_elbow(pos_init[4]);
    delay(10);
    servo_body_ex(1100, 10);
    delay(10);
    servo_elbow(2350);
    delay(10);
    servo_body_ex(pos_init[0], 10);
    delay(10);
    servo_elbow(pos_init[4]);
    delay(10);
  }
  servo_shoulder(pos_init[3]);
  delay(100);
}

void cheers(){ 
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
  servo_body_ex(1600, 6);
  delay(100);
  servo_gripper_ex(800);
  delay(100);
  servo_elbow(1950);
  delay(100);
  servo_gripper_ex(pos_init[5]);
  delay(100);
  servo_elbow(2000);
  delay(100);
  servo_body_ex(700, 6);
  delay(500);
  servo_body_ex(1600, 6);
  delay(100);
//  servo_elbow(1950);
//  delay(100);
  servo_gripper_ex(800);
  delay(100);
  servo_elbow(2350);
  delay(100);
  servo_gripper_ex(pos_init[5]);
  delay(100);
  servo_body_ex(pos_init[0], 6);
  delay(100);
  servo_elbow(pos_init[4]);
  delay(100);
  servo_shoulder(pos_init[3]);
  delay(100);
}

void set_initial() {
  servo_elbow(pos_init[4]);
  delay(50);
  servo_shoulder(pos_init[3]);
  delay(50);
  servo_body_ex(pos_init[0], 6);
  delay(50);
}

// Function to handle state changes
void update_instructions(String instrucStr) {
  Instruction newState;
  int tempo = 100;

  if (instrucStr.startsWith("wave")) {
    newState = WAVE;
    int length = instrucStr.length();
    if (length > 4) {
      char tempoChar = instrucStr.charAt(length - 1);
      tempo = (tempoChar - '0') * 100;  // Adjust tempo based on wave1, wave2, wave3
    }
  } else if (instrucStr == "press") {
    newState = PRESS;
  } else if (instrucStr == "cheers") {
    newState = CHEERS;
  } else if (instrucStr == "dance") {
    newState = DANCE;
  } else if (instrucStr == "waiting") {
    newState = WAITING;
  }

//  if (currentState != newState){
  if (newState != CHEERS){
    currentState = newState;
    // Perform actions based on the new emotion
    switch (currentState) {
      case PRESS:
        press_button();
        break;
//      case CHEERS:
//        cheers();
//        break;
      case WAVE:
        wave_end_effector(tempo);  // Pass the tempo for the wave action
        break;
      case DANCE:
        dance();
        break;
    }
  } else if (currentState != newState){
    currentState = newState;
    cheers();
    }
}

void setup() {

  Serial.begin(57600); // Starts the serial communication
  randomSeed(analogRead(A0));

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
  // Get the current time
  unsigned long currentTime = millis();

  if (currentTime - lastStateTime >= detectInterval) {
    // Check if data is available on the serial port
    if (Serial.available() > 0) {
      set_initial();

      // Read the incoming string
      String instructStr = Serial.readStringUntil('\n');
      instructStr.trim();  // Remove any extraneous whitespace or newline characters
      
      // Split the string by comma to extract each instruction
      int index = 0;
      while ((index = instructStr.indexOf(',')) != -1) {
        String firstInstruction = instructStr.substring(0, index);
        update_instructions(firstInstruction);  // Process each instruction separately
        instructStr = instructStr.substring(index + 1);  // Move to the next instruction
      }
      
      // Handle the last remaining instruction
      if (instructStr.length() > 0) {
        update_instructions(instructStr);
      }

    }
  }

  delay(1000); 
}
