import cv2
import numpy as np
import pyaudio
from cvzone.PoseModule import PoseDetector  # Import PoseDetector properly
import serial
import pygame
# import time


volume_threshold = 200
movement_threshold = 400
hubert_press_button_delay = 5
disableCamera = True           # for ex sound testing


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

# Audio Volume Data Function
def volumeData(nSeconds = 15):
    format = pyaudio.paInt16  # PCM - Pulse Code Modulation, 16 bits per sample
    samples_per_second = 1000  # Sampling rate (samples per second)
    samples_per_batch = 500  # Number of samples per buffer
    batches_per_second = int(samples_per_second/samples_per_batch)

    audio = pyaudio.PyAudio()

    # Open the stream
    stream = audio.open(format=format, channels=1,
                        rate=samples_per_second, input=True,
                        frames_per_buffer=samples_per_batch)

    volume = np.zeros(nSeconds*batches_per_second)
    for i in range(nSeconds*batches_per_second):
        data = stream.read(samples_per_batch, exception_on_overflow=False)
        # abs because negative values occur for negative waves
        audio_data = np.frombuffer(data, dtype=np.int16)
        audio_data = np.abs(audio_data)

        # Volume = RMS
        volume[i] = np.sqrt(np.mean(audio_data) ** 2)

    averageVol = np.mean(volume)
    stream.stop_stream()
    stream.close()
    audio.terminate()

    return averageVol

# Video Pose Detection Function
def videoDataEx(cap, detector): # TODO: add variable to change how many seconds we check
    success, img = cap.read()
    if not success:
        return 0  # If no video is captured, return 0 movement

    img = detector.findPose(img)  # Detect pose in the image
    lmlist, bbox = detector.findPosition(img)  # Get landmark positions

    # Assume movement is calculated as the difference between landmark positions
    if len(lmlist) > 0:
        movement = np.std(np.array(lmlist)[:, :2])  # Movement based on landmark position variance 
        # TODO make movement based on difference between frames, not distance between landmarks in the same frame
    else:
        movement = 0  # If no landmarks are found, movement is 0

    # Display the result in a window
    cv2.imshow("Dance_Floor", img)
    cv2.waitKey(1)  # Wait 1ms between frames

    return movement

# Perception Function
def perception(cap, detector, nSeconds):

    vol = volumeData(nSeconds) # Bottle-neck, takes nSeconds

    if disableCamera:
        movement = 10
    else:
        movement = videoDataEx(cap, detector) # TODO: add variable to change how many seconds we check

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
    
    send_string(ser, 'press') # Start with Hubert press button --> start some random song
    perception(cap, detector, hubert_press_button_delay) # just for delay #TODO: tune delay
    music.changeGenre() 

    nSeconds = 15 # over how many seconds to average sound volume. Determines frequency of mood check

    while True:
        mood = perception(cap, detector, nSeconds)
        print('Mood is:', mood)
        
        # Music instructions
        musicAction = musicActions[mood]  # TODO Add some memory here, ex if bored: change genre every other time, else dance

        # Hubert instructions
        # first check if change song --> hubert should only press button
        if musicAction == 'changeGenre':
            music.changeGenre()
            hubert.currentAction = 'press'
            print('hubert press button')
            send_string(ser, 'press')

        # if not press button, hubert instruciton depends on mood
        else: 
            if mood == 'happy':
                hubert.currentAction = 'wave'
                send_string(ser, 'wave')
                print('Hubert waves')

            elif mood == 'dancy':
                hubert.currentAction = 'dance'
                send_string(ser, 'dance')
                print('Hubert dances')
            
            elif mood == 'talking':
                hubert.currentAction = 'wave'
                send_string(ser, 'wave')
                print('Hubert waves')

            elif mood == 'bored':
                hubert.currentAction = 'cheers'
                send_string(ser, 'cheers')
                print('Hubert says cheers!')


        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit the loop
            break

    if not disableCamera:
        cap.release()
        cv2.destroyAllWindows()

# Set the serial port and baudrate
port = '/dev/ttyACM0'  # Change the port number as per your system
baudrate = 57600        # Adjust the baudrate as per your settings

# Main function
if __name__ == "__main__":
    # Initialize the serial port
    ser = init_serial(port, baudrate)

    try:
        # Run the decision function
        decision()
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        if ser is not None:
            ser.close()  # Close the serial connection before exiting