import cv2
import numpy as np
import pyaudio
from cvzone.PoseModule import PoseDetector  # Import PoseDetector properly
import serial
import pygame
import matplotlib.pyplot as plt


volume_threshold = 200
movement_threshold = 100000000
hubert_press_button_delay = 5
disableCamera = False           # for ex sound testing
serialCommunication = True 


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



# Pose Detector Initialization
if disableCamera:
    detector = 0
else:
    detector = PoseDetector()

def getVolAndMovement(cap, detector, nSeconds = 15):
    # SOUND INITIALIZATION
    format = pyaudio.paInt16  # PCM - Pulse Code Modulation, 16 bits per sample
    samples_per_second = 1000  # Sampling rate (samples per second)
    samples_per_batch = 200  # Number of samples per buffer
    batches_per_second = int(samples_per_second/samples_per_batch)

    audio = pyaudio.PyAudio()

    # Open the stream
    stream = audio.open(format=format, channels=1,
                        rate=samples_per_second, input=True,
                        frames_per_buffer=samples_per_batch)

    volume = np.zeros(nSeconds*batches_per_second)
    movement = np.zeros(nSeconds*batches_per_second)


    # AUDIO LOOP
    for i in range(nSeconds*batches_per_second):
        data = stream.read(samples_per_batch, exception_on_overflow=False)
        # abs because negative values occur for negative waves
        audio_data = np.frombuffer(data, dtype=np.int16)
        audio_data = np.abs(audio_data)
        # Volume = RMS
        volume[i] = np.sqrt(np.mean(audio_data) ** 2)

        # VIDEO
        if i==0:
            success, img = cap.read() # Get frame
            if not success:
                print('image from camera not successful')
                return
        else:
            oldImg = img 
            success, img = cap.read() # Get frame as np.ndarray
            if not success:
                print('image from camera not successful')
                return
            movement[i] = np.sum(np.abs(img-oldImg), dtype=np.float32)
            cv2.imshow('Dance_Floor',img)
            cv2.waitKey(10)  # Wait 500 ms between frames



    averageVol = np.mean(volume)
    stream.stop_stream()
    stream.close()
    audio.terminate()

    averageMovement = float(np.sum(movement[1:])/movement[1:].shape)

    # MOVEMENT DETECTION
    #img = detector.findPose(img)  # Detect pose in the image

    #lmlist, bbox = detector.findPosition(img)  # Get landmark positions

    # Assume movement is calculated as the difference between landmark positions
    #if len(lmlist) > 0:
        #movement = np.std(np.array(lmlist)[:, :2])  # Movement based on landmark position variance 
        # TODO make movement based on difference between frames, not distance between landmarks in the same frame
    #else:
        #movement = 0  # If no landmarks are found, movement is 0

    # Display the result in a window
    #cv2.imshow("Dance_Floor", img)
    #cv2.waitKey(1)  # Wait 1ms between frames

    return averageVol, averageMovement




# Perception Function
def perception(cap, detector, nSeconds):

    vol, movement = getVolAndMovement(cap, detector, nSeconds) # Bottle-neck, takes nSeconds

    print('Average volume', vol)
    print('Average movement', movement)

    moods = [['bored', 'dancy'], 
             ['talking', 'happy']] # TODO add more moods and thresholds

    highVol = int(vol > volume_threshold)  
    highMovement = int(movement > movement_threshold)  

    mood = moods[highVol][highMovement]
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
            'spanish': ['ritmo.mp3'],
            'neil': ['SweetCaroline.mp3']
        } # TODO add more songs, .mp3 or .wav

        self.genres = list(self.library.keys())
        self.currentGenre = 'bee'
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
        playSong('../Music/'+self.currentSong)

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

    if not disableCamera:
        # Initialize the camera
        cap = cv2.VideoCapture(0)
    else:
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
        if actions:
            hubert.changeAction(actions)

            if actions[0] == 'changeGenre': 
                music.changeGenre()

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit the loop
            break

    if not disableCamera:
        cap.release()
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