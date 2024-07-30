import numpy as np
from pydub import AudioSegment
import os
from scipy.io import wavfile
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from tensorflow.keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam
from scipy import signal

raw_data_path = "C:/Users/amank/Documents/coursera/ML/Reports/Trigger_word/raw_data"
#C:\Users\amank\Documents\coursera\ML\Reports\CNN\Trigger_word\raw_data
# Function to load raw audio data
def load_raw_audio(raw_data_path):
    activates = []
    negatives = []
    backgrounds = []

    # for filename in os.listdir(os.path.join(raw_data_path, "activates")):
    #     if filename.endswith(".wav"):
    #         activate = AudioSegment.from_wav(os.path.join(raw_data_path, "activates", filename))
    #         activates.append(activate)
    # for filename in os.listdir(os.path.join(raw_data_path, "negatives")):
    #     if filename.endswith(".wav"):
    #         negative = AudioSegment.from_wav(os.path.join(raw_data_path, "negatives", filename))
    #         negatives.append(negative)
    # for filename in os.listdir(os.path.join(raw_data_path, "backgrounds")):
    #     if filename.endswith(".wav"):
    #         background = AudioSegment.from_wav(os.path.join(raw_data_path, "backgrounds", filename))
    #         backgrounds.append(background)
    # return activates, negatives, backgrounds


    for folder_name, list_to_fill in [("activates", activates), ("negatives", negatives), ("backgrounds", backgrounds)]:
        folder_path = os.path.join(raw_data_path, folder_name)
        print(f"Loading files from {folder_path}")
        if not os.path.exists(folder_path):
            print(f"Error: The path {folder_path} does not exist.")
            continue
        for filename in os.listdir(folder_path):
            if filename.endswith(".wav"):
                file_path = os.path.join(folder_path, filename)
                print(f"Loading {file_path}")
                audio = AudioSegment.from_wav(file_path)
                list_to_fill.append(audio)
    return activates, negatives, backgrounds
# Load audio segments using pydub
activates, negatives, backgrounds = load_raw_audio(raw_data_path)

def get_random_time_segment(segment_ms):
    segment_start = np.random.randint(low=0, high=10000-segment_ms)
    segment_end = segment_start + segment_ms - 1
    return (segment_start, segment_end)

def is_overlapping(segment_time, previous_segments):
    segment_start, segment_end = segment_time
    overlap = False
    for previous_start, previous_end in previous_segments:
        if segment_start <= previous_end and segment_end >= previous_start:
            overlap = True
            break
    return overlap

def insert_audio_clip(background, audio_clip, previous_segments):
    segment_ms = len(audio_clip)
    segment_time = get_random_time_segment(segment_ms)
    retry = 5
    while is_overlapping(segment_time, previous_segments) and retry >= 0:
        segment_time = get_random_time_segment(segment_ms)
        retry -= 1
    if not is_overlapping(segment_time, previous_segments):
        previous_segments.append(segment_time)
        new_background = background.overlay(audio_clip, position=segment_time[0])
    else:
        new_background = background
        segment_time = (10000, 10000)
    return new_background, segment_time

def insert_ones(y, segment_end_ms):
    _, Ty = y.shape
    segment_end_y = int(segment_end_ms * Ty / 10000.0)
    if segment_end_y < Ty:
        for i in range(segment_end_y + 1, segment_end_y + 51):
            if i < Ty:
                y[0, i] = 1
    return y

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


def graph_spectrogram(wav_file):
    rate, data = wavfile.read(wav_file)
    frequencies, times, spectrogram = signal.spectrogram(data, rate)
    return spectrogram

def create_training_example(background, activates, negatives, Ty):
    background = background - 20
    y = np.zeros((1, Ty))
    previous_segments = []
    number_of_activates = np.random.randint(0, 5)
    random_indices = np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]
    for one_random_activate in random_activates:
        background, segment_time = insert_audio_clip(background, one_random_activate, previous_segments)
        segment_start, segment_end = segment_time
        y = insert_ones(y, segment_end)
    number_of_negatives = np.random.randint(0, 3)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]
    for random_negative in random_negatives:
        background, _ = insert_audio_clip(background, random_negative, previous_segments)
    
    background = match_target_amplitude(background, -20.0)
    
    file_handle = background.export("train.wav", format="wav")
    x = graph_spectrogram("train.wav")
    return x, y

# Set Ty to math your desired output time steps, e.g., 1375 if you want 1375 time steps.
Ty = 1375
Tx = 5511
np.random.seed(4543)
nsamples = 32
X = []
Y = []
for i in range(0, nsamples):
    if i % 10 == 0:
        print(f"Generating sample {i}")
    x, y = create_training_example(backgrounds[i % 2], activates, negatives, Ty)
    print(f"shape of small x : {x.shape}")
    X.append(x.swapaxes(0, 1))
    Y.append(y.swapaxes(0, 1))
X = np.array(X)
Y = np.array(Y)

print(f"Shape of X: {X.shape}")
print(f"Shape of Y: {Y.shape}")

X_dev = np.load("C:/Users/amank/Documents/coursera/ML/Reports/Trigger_word/XY_train/Y.npy")
Y_dev = np.load("C:/Users/amank/Documents/coursera/ML/Reports/Trigger_word/XY_dev/Y_dev.npy")

print(f"Shape of X_dev: {X_dev.shape}")
print(f"Shape of Y_dev: {Y_dev.shape}")


def modelf(input_shape):
    X_input = Input(shape=input_shape)
    X = Conv1D(filters=196, kernel_size=15, strides=4)(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Dropout(0.8)(X)
    X = GRU(units=128, return_sequences=True)(X)
    X = Dropout(0.8)(X)
    X = BatchNormalization()(X)
    X = GRU(units=128, return_sequences=True)(X)
    X = Dropout(0.8)(X)
    X = BatchNormalization()(X)
    X = TimeDistributed(Dense(1, activation='sigmoid'))(X)
    model = Model(inputs=X_input, outputs=X)
    return model

n_freq = 101
model = modelf(input_shape=(Tx, n_freq))
model.summary()
opt = Adam(learning_rate=0.0001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

model.fit(X, Y, batch_size=5, epochs=1)

loss, acc = model.evaluate(X_dev, Y_dev)
print("Dev set accuracy = ", acc)

# Save the model
model.save("trigger_word_model.h5")