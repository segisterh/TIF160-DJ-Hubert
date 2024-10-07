import cv2
import numpy as np
import pyaudio
from cvzone.PoseModule import PoseDetector  # Import PoseDetector properly
import pygame
import time



volume_threshold = 20
movement_threshold = 400
hubert_press_button_delay = 5
soundTesting = True

if soundTesting:
    detector = 0
else:
    # Pose Detector Initialization
    detector = PoseDetector()

# Audio Volume Data Function
def volumeData(nSeconds=30):
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
def videoData(cap, detector):
    success, img = cap.read()
    if not success:
        return 0  # If no video is captured, return 0 movement

    img = detector.findPose(img)  # Detect pose in the image
    lmlist, bbox = detector.findPosition(img)  # Get landmark positions

    # Assume movement is calculated as the difference between landmark positions
    if len(lmlist) > 0:
        movement = np.std(np.array(lmlist)[:, :2])  # Movement based on landmark position variance
    else:
        movement = 0  # If no landmarks are found, movement is 0

    # Display the result in a window
    cv2.imshow("Dance_Floor", img)
    cv2.waitKey(1)  # Wait 1ms between frames

    return movement

# Disable camera for sound testing
def noVideoData():
    return 10


# Perception Function
def perception(cap, detector, nSeconds=30):
    vol = volumeData(nSeconds)  # takes nSeconds
    if soundTesting:
        movement = noVideoData()
    else:
        movement = videoData(cap, detector)
    print('Average volume', vol)
    print('Average movement', movement)

    moods = [['bored', 'dancy'],
             ['talking', 'happy']]

    highVol = int(vol > volume_threshold)  # Using volume threshold
    highMovement = int(movement > movement_threshold)  # Using movement threshold

    mood = moods[highVol][highMovement]
    return mood

# Hubert Class
class Hubert:
    def __init__(self):
        self.actions = ['pressButton', 'waveFast', 'waveSlow', 'robotDance', 'nothing']
        self.currentAction = 'nothing'

def playSong(song):

    pygame.mixer.init()

    pygame.mixer.music.load(song)

    pygame.mixer.music.play()

    # Wait while the music is playing
    #while pygame.mixer.music.get_busy():
        #time.sleep(1)  # Sleep for 1 second to avoid using too much CPU


# CONVERT SOME SONG TO .wav: 
# 1: rename: remove spaces & set ending = .m4a
# 2: PASTE THE FOLLOWING LINE IN TERMINAL, REPLACE "OUTPUT" & "INPUT" WITH SONG TITLE 
# ffmpeg -i input.m4a output.wav

# Music Class
class Music:
    def __init__(self):
        self.genres2 = ['pop', 'EDM', '2000s', 'house', 'jazz', 'rap', 'rock', 'metal']
        self.library2 = {
            'silent': [None], 
            'bee': ['x2mate.com - bee. (128 kbps).mp3'],
            'pop': ['popsong1', 'popsong2'],
            'EDM': ['edmsong1', 'edmsong2'], '2000s': ['2000ssong1', '2000ssong2'],
            'house': ['housesong1', 'housesong2'], 'jazz': ['jazzsong1', 'jazzsong2'],
            'rap': ['rapsong1', 'rapsong2'], 'rock': ['rocksong1', 'rocksong2'],
            'metal': ['metalsong1', 'metalsong2']
        }
        self.library = {
            'bee': ['x2mate.com - bee. (128 kbps).mp3'],
            'spanish': ['ritmo.wav'],
            'neil': ['SweetCaroline.wav']
        }
        self.genres = list(self.library.keys())

        self.currentGenre = self.genres[0]
        self.currentSong = self.library[self.currentGenre][0]
        self.tempo = 0

    def changeGenre(self):
        while True:
            randomGenre = self.genres[np.random.choice(len(self.genres))]
            if randomGenre == self.currentGenre:
                continue
            else:
                self.currentGenre = randomGenre
                break
                
        self.currentSong = self.library[self.currentGenre][np.random.choice(len(self.library[self.currentGenre]))]
        print('Genre changes to:', self.currentGenre, 'and song is:', self.currentSong)
        _ = volumeData(hubert_press_button_delay)
        playSong(self.currentSong)


    def changeSong(self):
        self.currentSong = self.library[self.currentGenre][np.random.choice(len(self.library[self.currentGenre]))]
        print('Song changed to:', self.currentSong)

 

# Decision Function
def decision():
    hubert = Hubert()
    music = Music()

    musicActions = {'bored': 'changeGenre', 'dancy': 'nothing', 'talking': 'nothing', 'happy': 'nothing'}

    # Initialize the camera
    cap = cv2.VideoCapture(0)
    i = 0
    while True:
        i+=1
        if i ==1:
            nSeconds = 5
        else:
            nSeconds=30
        mood = perception(cap, detector, nSeconds)
        print('Mood is:', mood)

        musicAction = musicActions[mood]

        if musicAction == 'changeGenre':
            music.changeGenre()
            hubert.currentAction = 'pressButton'

        elif musicAction == 'nothing':
            hubert.currentAction = 'wave'
            print('Hubert waves')

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit the loop
            break

    cap.release()
    cv2.destroyAllWindows()




# Run the decision function
decision()

