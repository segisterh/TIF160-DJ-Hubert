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
facesWeight = 2
bodiesWeight = 3


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

def recordSurroundings(detector, nSeconds=15):
    samples_per_second = 1000
    samples_per_batch = 200
    batches_per_second = int(samples_per_second / samples_per_batch)

    # Initialize Audio
    format = pyaudio.paInt16
    audio = pyaudio.PyAudio()
    stream = audio.open(format=format, channels=1, rate=samples_per_second, input=True, frames_per_buffer=samples_per_batch)

    volume = np.zeros(nSeconds * batches_per_second)

    if not disableCamera:
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return averageVol, motion, faces

    prev_landmarks_list = []
    motion = []
    faces = []

    for i in range(nSeconds * batches_per_second):
        data = stream.read(samples_per_batch, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
        audio_data = np.abs(audio_data)
        volume[i] = np.sqrt(np.mean(audio_data) ** 2)

        if not disableCamera:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            frame, current_landmarks_list, motion_output, emotion_output = process_frame(frame, prev_landmarks_list)
            cv2.imshow('Family Dance Floor', frame)
            
            # Wait for at least 1 ms, to refresh and check for window events
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            motion.append(motion_output)
            faces.append(emotion_output)
            prev_landmarks_list = current_landmarks_list

    if not disableCamera:
        cap.release()
        cv2.destroyAllWindows()

    averageVol = np.mean(volume)
    stream.stop_stream()
    stream.close()
    audio.terminate()

    return averageVol, motion, faces




# Perception Function
def perception(detector, nSeconds):

    vol, motion, faces = recordSurroundings(detector, nSeconds) # Bottle-neck, takes nSeconds

    print('Average volume', vol)
    print('Motion', motion)
    print('Faces', faces)

    motionAvg = np.zeros(len(motion))
    facesAvg = np.zeros(len(motion))


    nPeopleMax = 0 
    for i, (mo, fa) in enumerate(zip(motion,faces)):
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

    satisfaction = nPeopleMax + facesWeight * np.mean(facesAvg) + bodiesWeight * np.mean(motionAvg)
    return satisfaction

# Hubert Class
class Hubert:
    def __init__(self):
        self.currentActions = []
        self.actions =[]
    
    def addAction(self, action):
        self.actions.append(action)

    def sendActions(self):
        print('Hubert action is', self.actions)
        if serialCommunication:
            send_string(ser, self.actions)
        else:
            pass

        self.actions = [] 
        self.currentActions = self.actions


# Music Class
class Music:
    def __init__(self):
        
        self.library = {
            'bee': ['x2mate.com - bee. (128 kbps).mp3'],
            'spanish': ['ritmo.wav'],
            'neil': ['SweetCaroline.wav']
        } 


        self.library = {'dance' : ['Avicii - For A Better Day (LYRICS).mp3', 'Avicii - Waiting For Love (Lyric Video).mp3', 'Avicii - Wake Me Up (Official Lyric Video).mp3', 'Avicii - Without You (Lyrics) ft. Sandro Cavazza.mp3', 'Avicii - You Make Me (Lyrics).mp3', 'Avicii-Pure Grinding (Lyrics).mp3', 'Calvin Harris & Disciples - How Deep Is Your Love (Lyrics).mp3', 'Calvin Harris - Bounce feat. Kelis HD (Lyrics) With Download Link!.mp3', 'Calvin Harris - Outside (Lyrics) ft. Ellie Goulding.mp3', 'CamelPhat & Elderbrook - Cola (Lyric Video).mp3', 'DMNDS, Fallen Roses - Calabria (Lyrics).mp3', 'Deorro - Five More Hours ft. Chris Brown (Lyrics).mp3', 'FISHER - Losing It (Lyrics version).mp3', 'Galantis - Runaway (U & I) [ Lyrics ].mp3', 'I could be the one [Nicktim] (lyrics) - Avicii ft. Nicky Romero.mp3', 'KVSH - Tokyo Drift Lyrics.mp3', 'Levels - Avicii (Lyrics).mp3', 'MGMT - Electric Feel Lyrics.mp3', 'Röyksopp & Robyn - Do It Again (lyric video).mp3', 'Tove Lo - 2 Die 4 (Lyrics).mp3', 'AYYBO - RIZZ (Lyrics).mp3', 'Alok x Mondello x CERES x Tribbs - LETS GET FKD UP (Lyrics).mp3', 'Calvin Harris - Feel So Close (Lyrics).mp3', 'David Guetta - Turn Me On ft. Nicki Minaj (Lyric Video).mp3', 'David Guetta vs Benny Benassi - Satisfaction (Lyrics)  push me and then just touch me.mp3', 'Halsey - Alone (Calvin Harris RemixAudio) ft. Stefflon Don.mp3', 'LUM!X, Gabry Ponte - Monster (Lyrics).mp3', 'Tiësto & KSHMR feat. Vassy - Secrets (Lyrics).mp3', 'Tiësto feat. 21 Savage, BIA - BOTH  Lyrics.mp3', 'Timmy Trumpet ft. Savage - Freaks (Lyrics).mp3', 'Where Dem Girls At - David Guetta Ft Flo Rida and Nicki Minaj (Lyrics _).mp3'] ,
'pop' : ['Cheat Codes & Dante Klein - Let Me Hold You (Turn Me On) [Official Lyric Video].mp3', "Macklemore & Ryan Lewis - Can't Hold Us (Lyrics) ft. Ray Dalton.mp3", 'Riton x Nightcrawlers ft. Mufasa & Hypeman - Friday (Lyrics) Dopamine Re-Edit.mp3', 'Robin Schulz - Sugar (Lyrics) feat. Francesco Yates.mp3', 'The Weeknd - The Hills (Lyrics).mp3', 'Tiësto - Wasted (Lyric Video) ft. Matthew Koma.mp3', 'Will.i.am, Britney Spears - Scream And Shout (Lyrics) (Tiktok).mp3', 'Alesso - Words (Lyrics) ft. Zara Larsson.mp3', 'Ava Max - Sweet but Psycho (Lyrics).mp3', 'Benjamin Ingrosso - All Night Long (Lyrics).mp3', "Benjamin Ingrosso - Look Who's Laughing Now (Lyrics).mp3", 'Billie Eilish - bad guy (Lyrics).mp3', 'Billie Eilish - bury a friend (Lyrics).mp3', 'Bruno Mars - 24K Magic (Lyrics).mp3', 'Calvin Harris - This Is What You Came For (Lyrics) ft. Rihanna.mp3', 'Daft Punk - Get Lucky (Lyrics) ft. Pharrell Williams, Nile Rodgers.mp3', 'Doja Cat - Paint The Town Red (Lyrics).mp3', 'E.T- Katy Perry Lyrics (without Kayne West).mp3', 'Eminem - The Monster (Lyrics) ft. Rihanna.mp3', 'Imagine Dragons x JID - Enemy (Lyrics).mp3', 'John Newman - Love Me Again (Lyrics).mp3', 'Justin Bieber - What Do You Mean (Lyrics).mp3', 'Katy Perry - Firework (Lyrics).mp3', 'Katy Perry - Last Friday Night (T.G.I.F) (Lyrics).mp3', 'Katy Perry - Swish Swish (Lyrics) ft. Nicki Minaj.mp3', 'Kesha - Die Young (Lyrics).mp3', 'Kesha - TiK ToK (Lyrics).mp3', 'Lady Gaga - Bad Romance (Lyrics).mp3', 'Lady Gaga - Just Dance (Lyrics).mp3', 'Lady Gaga - Poker Face (Lyrics).mp3', 'M.I.A. - Paper Planes (Lyrics).mp3', 'Mark Ronson - Uptown Funk (Lyrics) ft. Bruno Mars.mp3', 'Nicki Minaj Ft. Lil Uzi Vert - Everybody (Lyrics).mp3', 'Pharrell Williams - Happy (Lyrics).mp3', 'Purple Disco Machine & Benjamin Ingrosso - Honey Boy (Lyrics) [feat. Shenseea & Nile Rodgers].mp3', 'Rihanna - Disturbia (Lyrics).mp3', "Rihanna - Don't Stop The Music (Lyrics).mp3", 'Rihanna - Only Girl (In the World) (Lyrics).mp3', 'Rihanna - SOS (Lyrics).mp3', 'Rihanna - Umbrella (Lyrics).mp3', 'Sam Smith - Unholy ft. Kim Petras.mp3', 'Starships - Nicki Minaj - Lyrics.mp3', 'The Black Eyed Peas - I Gotta Feeling (Lyrics).mp3', 'The Weeknd - A lonely night lyrics.mp3'] ,
'spanish' : ['Con Calma - Daddy Yankee & Snow (Lyrics).mp3', 'ritmo.wav', 'Bad Bunny - Tití Me Preguntó (LetraLyricsSong).mp3'] ,
'y2k' : ['David Guetta - Sexy Bitch (feat. Akon)  Lyrics.mp3', 'David Guetta Snoop Dogg - Sweat Lyrics.mp3', 'Flo Rida feat. Ke$ha - Right Round  Lyrics.mp3', 'Sean Paul - Get Busy (lyrics).mp3', 'Timbaland - The Way I Are (Lyrics) ft. Keri Hilson, D.O.E..mp3', 'Usher - Yeah! (Lyrics) ft. Lil Jon, Ludacris.mp3', 'Black Eyed Peas - Pump It (Lyrics).mp3', 'Boom Boom Pow - Black Eyed Peas (Lyrics).mp3', 'Britney Spears - Womanizer (Lyrics).mp3', 'David Guetta - Memories (Lyrics) (tiktok) ft. Kid Cudi.mp3', 'Empire State Of Mind - Jay-Z Feat. Alicia Keys  Lyrics On Screen [HD].mp3', 'Far East Movement - Like A G6 (Lyrics) ft. The Cataracs, DEV.mp3', 'Flo Rida - Low ft. T-Pain [Apple Bottom Jeans] (Lyrics).mp3', 'I Got It From My Mama With Lyrics.mp3', 'Iyaz - Replay  Lyrics.mp3', 'Jennifer Lopez - On The Floor (Lyrics) ft. Pitbull.mp3', 'Madonna  - 4 Minutes (Lyrics)  ft. Justin Timberlake & Timbaland.mp3', 'Miley Cyrus - Party In The USA (Lyrics).mp3', 'OutKast - Hey Ya! (Lyrics).mp3', 'Pitbull - Hotel Room Service (Lyrics).mp3', 'Pitbull - Timber (Lyrics) ft. Ke$ha.mp3', 'Rihanna - Pon de Replay (Lyrics).mp3', 'T.I., Rihanna - Live Your Life (Lyrics).mp3', 'Taio Cruz - Break Your Heart (Lyrics) ft. Ludacris.mp3', "Usher - DJ Got Us Fallin' In Love (Lyrics) ft. Pitbull.mp3", 'Wheatus - Teenage Dirtbag (Lyrics).mp3'] ,
'sixties' : ['SweetCaroline.wav'] ,
'schlager' : ['Alla flickor.mp3', 'Carola_ Främling.mp3', 'Det gör ont.mp3', 'Håll om mig.mp3', 'Jag ljuger så bra.mp3', 'Markoolio & Linda Bengtzing - Värsta schlagern  LYRICS.mp3', 'När vindarna viskar mitt namn.mp3'] ,
'boyband' : ["Backstreet Boys - Everybody (Backstreet's Back) (Lyrics).mp3", 'Best Song Ever - One Direction (Lyrics).mp3', "NSYNC - It's Gonna Be Me (Lyrics).mp3", 'One Direction - What Makes You Beautiful(Lyrics).mp3', 'Steal My Girl - One Direction (Lyrics).mp3'] ,
's80' : ['Ray Parker Jr. - Ghostbusters (Lyrics).mp3','ABBA - Dancing Queen (Lyrics).mp3', 'ABBA - Does Your Mother Know (Lyrics).mp3', 'ABBA - Gimme! Gimme! Gimme! (A Man After Midnight) [Lyrics].mp3', 'ABBA - Lay All Your Love On Me Lyrics.mp3','Beat It - Michael Jackson (Lyrics).mp3', 'Billy Idol - Dancing With Myself (Lyrics).mp3', 'Bonnie Tyler - Total Eclipse of the Heart (Official Lyric Video).mp3', 'Bruce Springsteen - Dancing in the Dark (Lyrics).mp3', 'Michael Jackson - Billie Jean (Lyrics).mp3', 'Michael Jackson - Smooth Criminal (Lyrics).mp3', 'Michael Jackson - Thriller (2003 Edit) (Lyrics version).mp3', 'Queen - Killer Queen (Official Lyric Video).mp3', 'Rick James - Super Freak (Lyrics HD).mp3', "Rockwell - Somebody's Watching Me Lyrics.mp3", 'Simple Minds - Dont You (Forget About Me) (Lyrics).mp3', 'Toto - Africa (Lyrics).mp3', 'Whitney Houston - I Wanna Dance With Somebody (Lyrics).mp3', 'a-ha - Take On Me (Lyrics).mp3'] ,
'dubstep' : ['FIRST OF THE YEAR EQUINOX - SKRILLEX (Lyrics).mp3', 'Kanye West - Blood on The Leaves (Lyrics).mp3', 'Kendrick Lamar  HUMBLE (Lyrics  Lyrics Video) (Skrillex Remix).mp3', 'Skrillex - Scary Monsters and Nice Sprites w Lyrics.mp3'] ,
'rap' : ['cassö, RAYE, D-Block Europe - Prada (Lyrics).mp3','Kanye West - Monster (Lyrics).mp3', 'Kendrick Lamar - Not Like Us (Lyrics) Drake Diss.mp3', 'Opps - Vince Staples ft Yugen Blakrok (Lyrics).mp3', 'Panda lyrics Desiigner.mp3', "SAYGRACE - You Don't Own Me (Lyrics) ft. G-Eazy.mp3"] ,
'rock' : ['Linkin Park - The Emptiness Machine (Lyrics).mp3', 'Zombie - The Cranberries (Lyrics).mp3'] }


        self.genres = list(self.library.keys())
        self.genreScores = {'pop':10, 'dance':10, 'spanish':9,'y2k':10, 'sixties':6, 'schlager':6, 'boyband':6 , 's80':6, 'dubstep':4, 'rap':4, 'rock':4}
        self.triedGenres = set([])
        self.currentGenre = 'sixties'
        self.currentSong = self.library[self.currentGenre][0]
        self.tempo = 0   # TODO we could label each song with tempo to determine wave-speed of hubert
        self.musicHistory = []
        
    
    def musicDecision(self, satisfaction, satisfactionHistory, musicHistory): # histories until t-1
        if satisfaction < 3:
            changeGenreAfter = 8 # for how many timesteps to try the same genre
            changeSongAfter = 2 # for how many timesteps to try the same song
        elif satisfaction < 6:
            changeGenreAfter = 12 
            changeSongAfter = 3
        elif satisfaction <9:
            changeGenreAfter = 16 
            changeSongAfter = 4
        else: #satisfaction high
            changeGenreAfter = np.inf
            changeSongAfter = 6
        
        howLongSameGenre = 0
        howLongSameSong = 0

        musicHistory = musicHistory.reverse() # local change, original list in decision should remain the same

        for gen, song in musicHistory:
            if gen == self.currentGenre:
                howLongSameGenre+=1
                if song==self.currentSong:
                    howLongSameSong += 1
            else:
                break
        
        
        if howLongSameGenre >= changeGenreAfter:
            self.changeGenre(satisfactionHistory, musicHistory, howLongSameGenre)

        if howLongSameSong >= changeSongAfter:
            self.changeSong(satisfactionHistory, musicHistory, howLongSameGenre)


    def changeGenre(self, satisfactionHistory, howLongSameGenre,assignScore=True): # assignScore needs to be false first time, since we dont have history then
        if assignScore:
            genreScore = sum(satisfactionHistory[-howLongSameGenre:])/howLongSameGenre
            self.genreScores[self.currentGenre] = genreScore

        randomWeightedGenre = []
        for genre in self.genres:
            score = self.genreScores[genre]
            for i in range(int(score)):
                randomWeightedGenre.append(genre) # if high score --> added many times --> higher probability to be chosen
        self.currentGenre = randomWeightedGenre[np.random.random_integers(len(randomWeightedGenre)-1)] # choose genre randomly from "weighted" list

        print('Genre changed to:', self.currentGenre)
        self.changeSong(satisfactionHistory, howLongSameGenre)
        print('and song is', self.currentSong)
        playSong('Music/'+self.currentSong)
        self.triedGenres.add(self.currentGenre) # for now: not used, might use later
    


    def changeSong(self, satisfactionHistory, howLongSameGenre):
        songsLeft = len(self.library[self.currentGenre])
        if songsLeft <1:
            self.changeGenre(satisfactionHistory, howLongSameGenre)

        randomIndex = np.random.choice(len(self.library[self.currentGenre]))
        self.currentSong = self.library[self.currentGenre][randomIndex]
        print('Song changed to:', self.currentSong)
        self.library[self.currentGenre].pop(randomIndex) # pop song, to prevent same song being played twice

def playSong(song):
    pygame.mixer.init()
    pygame.mixer.music.load(song)
    pygame.mixer.music.play()



# Decision Function
def decision():
    hubert = Hubert()
    music = Music()
    
    
    hubert.addAction('press')
    #perception(cap, detector, hubert_press_button_delay) # just for delay #TODO: tune delay
    music.changeGenre([],[],False) 
    songs = 0
    for key in list(music.library.keys()):
        songs+= len(music.library[key])
    
    print('#songs = ',songs)
    nSeconds = 18 # over how many seconds to average sound volume. Determines frequency of mood check
    satisfactionHistory = []
    musicHistory = []

    while True:
    #while True:
        satisfaction = perception(detector, nSeconds)
        print('Satisfaction is:', satisfaction)
        
        # Music instructions
        music.musicDecision(satisfaction, satisfactionHistory, musicHistory) 
        satisfactionHistory.append(satisfaction)
        musicHistory.append((music.currentGenre, music.currentSong))

        # Hubert instructions
        send_actions = []

        # Hubert instruciton depends on mood
        #if mood == 'happy':
            #hubert.addAction('wave1')  # Can be changed to append(musicAction)

        #elif mood == 'dancy':
            #hubert.addAction('dance')
        
        '''elif mood == 'talking':
            hubert.addAction('wave2')

        elif mood == 'bored':
            hubert.addAction('press')
            hubert.addAction('cheers')

        # Send all actions to Hubert in one go
        if send_actions: # it said "actions" which were undefined so i changed to send_actions i guess you meant that
            hubert.sendActions()

            if send_actions[0] == 'press': 
                music.changeGenre()'''

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

# CONVERT SOME SONG TO .wav: 
# 1: rename song: remove spaces & set ending = .m4a
# 2: PASTE THE FOLLOWING LINE IN TERMINAL, REPLACE "OUTPUT" & "INPUT" WITH SONG TITLE 
# ffmpeg -i input.m4a output.wav

