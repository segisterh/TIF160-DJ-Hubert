import cv2
import numpy as np
import pyaudio
from cvzone.PoseModule import PoseDetector  # Import PoseDetector properly
import serial
# import time

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
detector = PoseDetector()

# Audio Volume Data Function
def volumeData():
    format = pyaudio.paInt16  # PCM - Pulse Code Modulation, 16 bits per sample
    samples_per_second = 1000  # Sampling rate (samples per second)
    samples_per_batch = 500  # Number of samples per buffer

    audio = pyaudio.PyAudio()

    # Open the stream
    stream = audio.open(format=format, channels=1,
                        rate=samples_per_second, input=True,
                        frames_per_buffer=samples_per_batch)

    data = stream.read(samples_per_batch, exception_on_overflow=False)
    # abs because negative values occur for negative waves
    audio_data = np.frombuffer(data, dtype=np.int16)
    audio_data = np.abs(audio_data)

    # Volume = RMS
    volume = np.sqrt(np.mean(audio_data) ** 2)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    return volume

# Video Pose Detection Function
def videoDataEx(cap, detector):
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

# Perception Function
def perception(cap, detector):
    vol = volumeData()
    movement = videoDataEx(cap, detector)
    print(movement)

    moods = [['bored', 'dancy'],
             ['talking', 'happy']]

    highVol = int(vol > 20)  # Using volume threshold = 2
    highMovement = int(movement > 400)  # Using movement threshold = 0.5

    mood = moods[highVol][highMovement]
    return mood

# Hubert Class
class Hubert:
    def __init__(self):
        self.actions = ['pressButton', 'waveFast', 'waveSlow', 'robotDance', 'nothing']
        self.currentAction = 'nothing'

# Music Class
class Music:
    def __init__(self):
        self.genres = ['pop', 'EDM', '2000s', 'house', 'jazz', 'rap', 'rock', 'metal']
        self.library = {
            'silent': [None], 'pop': ['popsong1', 'popsong2'],
            'EDM': ['edmsong1', 'edmsong2'], '2000s': ['2000ssong1', '2000ssong2'],
            'house': ['housesong1', 'housesong2'], 'jazz': ['jazzsong1', 'jazzsong2'],
            'rap': ['rapsong1', 'rapsong2'], 'rock': ['rocksong1', 'rocksong2'],
            'metal': ['metalsong1', 'metalsong2']
        }
        self.currentGenre = 'pop'
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
        print('Genre changed to:', self.currentGenre, 'and song is:', self.currentSong)

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

    while True:
        mood = perception(cap, detector)
        print('Mood is:', mood)

        musicAction = musicActions[mood]

        if musicAction == 'changeGenre':
            music.changeGenre()
            hubert.currentAction = 'pressButton'
            send_string(ser, 'press')

        elif musicAction == 'nothing':
            hubert.currentAction = 'wave'
            send_string(ser, 'wave')
            print('Hubert waves')

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit the loop
            break

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