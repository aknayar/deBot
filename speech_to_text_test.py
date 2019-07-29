import speech_recognition as sr
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

fs = 48000  # Sample rate
seconds = 3  # Duration of recording
sd.default.device = "MIC NAME HERE"
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)

sd.wait()  # Wait until recording is finished
y = (np.iinfo(np.int32).max * (myrecording / np.abs(myrecording).max())).astype(np.int32)
write('output.wav', fs, y)
audio = 'output.wav'
# use the audio file as the audio source

r = sr.Recognizer()

with sr.AudioFile(audio) as source:
	# reads the audio file. Here we use record instead of
	# listen
	audio = r.record(source)

try:
	print("The audio file contains: " + r.recognize_google(audio))

except sr.UnknownValueError:
	print("Google Speech Recognition could not understand audio")

except sr.RequestError as e:
	print("Could not request results from Google Speech Recognition service; {0}".format(e))