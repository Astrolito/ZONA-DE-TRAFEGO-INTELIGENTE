import pyaudio
import wave
import av

# Captura com PyAudio
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
OUTPUT_FILENAME = "output.wav"

audio = pyaudio.PyAudio()

stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

frames = []
print("Gravando...")

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Gravação finalizada.")
stream.stop_stream()
stream.close()
audio.terminate()

# Salvar como WAV
wf = wave.open(OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

# Processar com PyAV
container = av.open(OUTPUT_FILENAME)
for packet in container.demux():
    for frame in packet.decode():
        print(frame)