import cv2
import numpy as np
import pyaudio
from cvzone.PoseModule import PoseDetector  # Import PoseDetector properly
import serial
import pygame
import matplotlib.pyplot as plt
import mediapipe as mp
from deepface import DeepFace
import torch


volume_threshold = 200
movement_threshold = 100000000
hubert_press_button_delay = 5
disableCamera = False           # for ex sound testing
serialCommunication = False 


# Initialize MediaPipe Pose (FOR CV)
if not disableCamera:
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.classes = [0]  # Filter for person class only

# Initialize serial communication
def init_serial(port='/dev/ttyUSB0', baudrate=9600):
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        print(f"Connected to {port} at {baudrate} baudrate.")
        return ser
    except serial.SerialException as e:
        print(f"Error: {e}")
        return None

# Send string data
def send_string(ser, data):
    if ser is not None:
        try:
            # Send string data as byte stream
            ser.write(data.encode('utf-8'))
            print(f"Sent: {data}")
        except serial.SerialTimeoutException as e:
            print(f"Timeout Error: {e}")
    else:
        print("Serial port not initialized.")



# FUNCTIONS FOR CV
def cvMotion(landmarks_prev, landmarks_curr, threshold=0.05):
    if landmarks_prev is not None and landmarks_curr is not None:
        prev_coords = np.array([[l.x, l.y] for l in landmarks_prev.landmark])
        curr_coords = np.array([[l.x, l.y] for l in landmarks_curr.landmark])
        motion = np.linalg.norm(curr_coords - prev_coords)
        return 1 if motion >= threshold else 0
    return 0

# FUNCTIONS FOR CV
def cvFaces(img, face_locations):
    emotions = []
    try:
        for (top, right, bottom, left) in face_locations:
            face_img = img[top:bottom, left:right]
            result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
            dominant_emotion = result[0]['dominant_emotion'] if isinstance(result, list) else result['dominant_emotion']
            emotions.append(0 if dominant_emotion in ['neutral', 'disgusted'] else 1)
    except Exception as e:
        print("Error detecting emotion:", e)
    return emotions

# FUNCTIONS FOR CV
def process_frame(frame, prev_landmarks_list):
    # YOLOv5 person detection
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()
    
    current_landmarks_list = []
    motion_output = []
    face_locations = []

    for detection in detections:
        if detection[5] == 0:  # Class 0 is person
            x1, y1, x2, y2 = map(int, detection[:4])
            person_frame = frame[y1:y2, x1:x2]
            face_locations.append((y1, x2, y2, x1))  # top, right, bottom, left
            
            # MediaPipe pose estimation
            results_pose = pose.process(cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB))
            
            if results_pose.pose_landmarks:
                current_landmarks_list.append(results_pose.pose_landmarks)
                
                # Calculate motion
                prev_landmarks = prev_landmarks_list[len(current_landmarks_list)-1] if len(current_landmarks_list) <= len(prev_landmarks_list) else None
                motion = cvMotion(prev_landmarks, results_pose.pose_landmarks)
                
                motion_output.append(motion)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                motion_output.append(0)  # No motion detected

    # Emotion detection
    emotions = cvFaces(frame, face_locations)
    
    # Fill in emotion output for people without detected faces
    emotion_output = [1 if i >= len(emotions) else emotions[i] for i in range(len(motion_output))]

    return frame, current_landmarks_list, motion_output, emotion_output



# Pose Detector Initialization
if disableCamera:
    detector = 0
else:
    detector = PoseDetector()

def recordSurroundings(cap, detector, nSeconds = 15):
    # SOUND INITIALIZATION
    samples_per_second = 1000  # Sampling rate (samples per second)
    samples_per_batch = 200  # Number of samples per buffer
    batches_per_second = int(samples_per_second/samples_per_batch)
    
    # INITIALIZE AUDIO  
    format = pyaudio.paInt16  # PCM - Pulse Code Modulation, 16 bits per sample
    audio = pyaudio.PyAudio()
    stream = audio.open(format=format, channels=1,
                        rate=samples_per_second, input=True,
                        frames_per_buffer=samples_per_batch)

    volume = np.zeros(nSeconds*batches_per_second)
    #movement = np.zeros(nSeconds*batches_per_second)

    # INITIALIZE CV
    if not disableCamera:
        cap = cv2.VideoCapture(0)
    prev_landmarks_list = []
    motion = []
    faces =  []


    # AUDIO LOOP
    for i in range(nSeconds*batches_per_second):
        data = stream.read(samples_per_batch, exception_on_overflow=False)
        # abs because negative values occur for negative waves
        audio_data = np.frombuffer(data, dtype=np.int16)
        audio_data = np.abs(audio_data)
        # Volume = RMS
        volume[i] = np.sqrt(np.mean(audio_data) ** 2)

        # CV
        if not disableCamera:
            ret, frame = cap.read()

            if not ret:
                print("Failed to capture frame")
                break
            
            frame, current_landmarks_list, motion_output, emotion_output = process_frame(frame, prev_landmarks_list)
            #cv2.imshow('Multi-Person Detection', frame)

            motion.append(motion_output)
            faces.append(emotion_output) 
            
            prev_landmarks_list = current_landmarks_list

    #cap.release()
    if not disableCamera:
        cap.release()
        #cv2.destroyAllWindows()

    averageVol = np.mean(volume)
    stream.stop_stream()
    stream.close()
    audio.terminate()

    return averageVol, motion, faces



# Perception Function
def perception(cap, detector, nSeconds):

    vol, motion, faces = recordSurroundings(cap, detector, nSeconds) # Bottle-neck, takes nSeconds

    print('Average volume', vol)
    print('Motion', motion)
    print('Faces', faces)

    motionAvg = np.zeros(len(motion))
    facesAvg = np.zeros(len(motion))


    nPeopleMax = 0 #max(len(i) for i in motion)
    for i, mo, fa in enumerate(zip(motion,faces)):
        nBodies = len(mo)
        nFaces = len(fa)

        if nBodies == 0: 
            motionAvg[i] = 0
        else:
            motionAvg[i] = np.sum(mo)/nBodies

        if nFaces == 0:
            facesAvg[i] = 0
        else:
            facesAvg[i] = np.sum(mo)/nFaces

        if nBodies > nPeopleMax or nFaces > nPeopleMax:
            nPeopleMax = max(nBodies, nFaces)

        
    if nPeopleMax == 0:
        print('no people to be seen')
        return 'empty'


    mood = 'bored' # TODO make intelligent mood interpretation
    """moods = [['bored', 'dancy'], 
             ['talking', 'happy']] # TODO add more moods and thresholds

    highVol = int(vol > volume_threshold)  
    highMovement = int(movement > movement_threshold)  

    mood = moods[highVol][highMovement]"""

    return mood



# Hubert Class
class Hubert:
    def __init__(self):
        self.actions = ['press', 'wave1', 'wave2', 'dance', 'nothing'] # TODO sync with exact strings from kinematics
        self.currentAction = 'nothing'
    
    def changeAction(self, action):
        self.currentAction= action
        print('Hubert action is', action)

        if serialCommunication:
            send_string(ser, action)
        else:
            pass

def playSong(song):
    pygame.mixer.init()
    pygame.mixer.music.load(song)
    pygame.mixer.music.play()


# CONVERT SOME SONG TO .wav: 
# 1: rename song: remove spaces & set ending = .m4a
# 2: PASTE THE FOLLOWING LINE IN TERMINAL, REPLACE "OUTPUT" & "INPUT" WITH SONG TITLE 
# ffmpeg -i input.m4a output.wav

# Music Class
class Music:
    def __init__(self):
        
        self.library = {
            'bee': ['x2mate.com - bee. (128 kbps).mp3'],
            'spanish': ['ritmo.wav'],
            'neil': ['SweetCaroline.wav']
        } # TODO add more songs, .mp3 or .wav


        self.library = {'dance' : ['Avicii - For A Better Day (LYRICS).mp3', 'Avicii - Waiting For Love (Lyric Video).mp3', 'Avicii - Wake Me Up (Official Lyric Video).mp3', 'Avicii - Without You (Lyrics) ft. Sandro Cavazza.mp3', 'Avicii - You Make Me (Lyrics).mp3', 'Avicii-Pure Grinding (Lyrics).mp3', 
                                   'Calvin Harris & Disciples - How Deep Is Your Love (Lyrics).mp3', 'Calvin Harris - Bounce feat. Kelis HD (Lyrics) With Download Link!.mp3', 'Calvin Harris - Outside (Lyrics) ft. Ellie Goulding.mp3', 'CamelPhat & Elderbrook - Cola (Lyric Video).mp3', 
                                   'DMNDS, Fallen Roses - Calabria (Lyrics).mp3', 'Deorro - Five More Hours ft. Chris Brown (Lyrics).mp3', 'FISHER - Losing It (Lyrics version).mp3', 'Galantis - Runaway (U & I) [ Lyrics ].mp3', 'I could be the one [Nicktim] (lyrics) - Avicii ft. Nicky Romero.mp3', 
                                   'KVSH - Tokyo Drift Lyrics.mp3', 'Levels - Avicii (Lyrics).mp3', 'MGMT - Electric Feel Lyrics.mp3', 'Röyksopp & Robyn - Do It Again (lyric video).mp3', 'Tove Lo - 2 Die 4 (Lyrics).mp3'],
                                   'pop' : ['Cheat Codes & Dante Klein - Let Me Hold You (Turn Me On) [Official Lyric Video].mp3', "Macklemore & Ryan Lewis - Can't Hold Us (Lyrics) ft. Ray Dalton.mp3", 'Riton x Nightcrawlers ft. Mufasa & Hypeman - Friday (Lyrics) Dopamine Re-Edit.mp3', 'Robin Schulz - Sugar (Lyrics) feat. Francesco Yates.mp3', 'The Weeknd - The Hills (Lyrics).mp3', 'Tiësto - Wasted (Lyric Video) ft. Matthew Koma.mp3', 'Will.i.am, Britney Spears - Scream And Shout (Lyrics) (Tiktok).mp3'],'spanish' : ['Con Calma - Daddy Yankee & Snow (Lyrics).mp3', 'ritmo.wav'],
                                   'y2' : ['David Guetta - Sexy Bitch (feat. Akon)  Lyrics.mp3', 'David Guetta Snoop Dogg - Sweat Lyrics.mp3', 'Flo Rida feat. Ke$ha - Right Round  Lyrics.mp3', 'Sean Paul - Get Busy (lyrics).mp3', 'Timbaland - The Way I Are (Lyrics) ft. Keri Hilson, D.O.E..mp3', 'Usher - Yeah! (Lyrics) ft. Lil Jon, Ludacris.mp3'],
                                   'sixties' : ['SweetCaroline.m4a', 'SweetCaroline.wav']} 
        # TODO add more songs, .mp3 or .wav

        self.genres = list(self.library.keys())
        self.currentGenre = 'sixties'
        self.currentSong = self.library[self.currentGenre][0]
        self.tempo = 0   # TODO we could label each song with tempo to determine wave-speed of hubert

    def changeGenre(self):
        while True:
            randomGenre = self.genres[np.random.choice(len(self.genres))]
            if randomGenre == self.currentGenre:
                continue
            else:
                self.currentGenre = randomGenre
                break

        self.currentSong = self.library[self.currentGenre][np.random.choice(len(self.library[self.currentGenre]))]
        print('Genre changed to:', self.currentGenre, 'and song is:', self.currentSong)
        playSong('Music/'+self.currentSong)

    def changeSong(self):
        # TODO this might not be used, or maybe only change song when song is finished?
        self.currentSong = self.library[self.currentGenre][np.random.choice(len(self.library[self.currentGenre]))]
        print('Song changed to:', self.currentSong)

# Decision Function
def decision():
    hubert = Hubert()
    music = Music()
    
    # TODO: Should depend also on current music, ex if dancy and calm music --> more dancy music
    musicActions = {'bored': 'changeGenre', 'dancy': 'nothing', 'talking': 'nothing', 'happy': 'nothing'}

    #if not disableCamera:
        # Initialize the camera
        #cap = cv2.VideoCapture(0)
    #else:
    cap=None
    
    hubert.changeAction('press')
    #perception(cap, detector, hubert_press_button_delay) # just for delay #TODO: tune delay
    music.changeGenre() 

    nSeconds = 18 # over how many seconds to average sound volume. Determines frequency of mood check

    while True:
        mood = perception(cap, detector, nSeconds)
        print('Mood is:', mood)
        
        # Music instructions
        musicAction = musicActions[mood]  # TODO Add some memory here, ex if bored: change genre every other time, else dance

        # Hubert instructions
        send_actions = []

        # Hubert instruciton depends on mood
        if mood == 'happy':
            send_actions.append('wave1')  # Can be changed to append(musicAction)

        elif mood == 'dancy':
            send_actions.append('dance')
        
        elif mood == 'talking':
            send_actions.append('wave2')

        elif mood == 'bored':
            send_actions.append('press')
            send_actions.append('cheers')

        # Send all actions to Hubert in one go
        if send_actions: # it said "actions" which were undefined so i changed to send_actions i guess you meant that
            hubert.changeAction(send_actions)

            if send_actions[0] == 'press': 
                music.changeGenre()

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit the loop
            break

    if not disableCamera:
        #cap.release()
        cv2.destroyAllWindows()

# Set the serial port and baudrate
port = '/dev/ttyACM0'  # Change the port number as per your system
baudrate = 57600       # Adjust the baudrate as per your settings

# Main function
if __name__ == "__main__":
    # Initialize the serial port
    if serialCommunication:
        ser = init_serial(port, baudrate)
    else:
        ser = None

    try:
        # Run the decision function
        decision()
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        if ser is not None:
            ser.close()  # Close the serial connection before exiting