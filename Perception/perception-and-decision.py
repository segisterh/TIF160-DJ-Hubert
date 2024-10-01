import pyaudio
import numpy as np

volumeThreshold = 2
movementThreshold = 0.5

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

    # abs bc. negative values occur for negative waves
    audio_data = np.frombuffer(data, dtype=np.int16)
    audio_data = np.abs(audio_data)
            
    # Volume = RMS
    volume = np.sqrt(np.mean(audio_data)**2)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    return volume

def videoDataEx():
    return 0.6

def perception():
    vol = volumeData()
    movement = videoDataEx()

    moods = [['bored', 'dancy'],
             ['talking', 'happy']]
    
    highVol = int(vol>volumeThreshold)
    highMovement = int(movement > movementThreshold)

    mood = moods[highVol][highMovement]
    return mood

class Hubert:
    def __init__(self):
        self.actions = ['pressButton', 'waveFast', 'waveSlow', 'robotDance', 'nothing']
        self.currentAction = 'nothing'
    
     
class Music:
    def __init__(self):
        self.genres = ['pop', 'EDM', '2000s','house', 'jazz', 'rap', 'rock', 'metal'] #{0:'pop', 1:'EDM', 2:'2000s',3:'house', 4:'jazz', 5:'rap', 6:'rock', 7:'metal'}
        self.library = {'silent':[None], 'pop':['popsong1', 'popsong2'], 'EDM':['edmsong1', 'edmsong2'], '2000s':['2000ssong1', '2000ssong2'], 'house':['housesong1', 'housesong2'], 'jazz':['jazzsong1', 'jazzsong2'], 'rap':['rapsong1', 'rapsong2'], 'rock':['rocksong1', 'rocksong2'], 'metal':['metalsong1', 'metalsong2']}
        self.currentGenre = 'pop'
        self.currentSong = self.library[self.currentGenre][0]
        self.tempo = 0
    
    def changeGenre(self): 
        # Maybe make list "self.newgenres" so it remembers/so we dont try same twice?
        while True:
            randomGenre = self.genres[np.random.choice(len(self.genres))]
            if randomGenre == self.currentGenre:
                continue
            else:
                self.currentGenre = randomGenre
                break

        self.currentSong = self.library[self.currentGenre][np.random.choice(len(self.library[self.currentGenre]))]
        print('genre changed to:', self.currentGenre, 'and song is:', self.currentSong)

    def changeSong(self):
        self.currentSong = self.library[self.currentGenre][np.random.choice(len(self.library[self.currentGenre]))]
        print('song changed to:', self.currentSong)


def decision():

    hubert = Hubert()
    music = Music()
    
    musicActions = {'bored':'changeGenre', 'dancy':'nothing', 'talking':'nothing', 'happy':'nothing'} # simple as a start

    while True:
        mood = perception()
        print('mood is', mood)
        
        musicAction = musicActions[mood]
            
        if musicAction == 'changeGenre':
            music.changeGenre()
            hubert.currentAction = 'pressButton'
        
        elif musicAction == 'nothing':
            hubert.currentAction = 'wave'
            print('hubert waves')

        break

decision()