# Task3
# music_generator.py

import glob
import pickle
import numpy as np
import random
import string
from music21 import converter, instrument, note, chord, stream
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense
from keras.utils import to_categorical
import os

# Create folders
os.makedirs("data", exist_ok=True)
os.makedirs("output", exist_ok=True)

# STEP 1: Load and preprocess MIDI files
def load_midi_files():
    notes = []
    for file in glob.glob("data/*.mid"):
        midi = converter.parse(file)
        parts = instrument.partitionByInstrument(midi)
        elements = parts.parts[0].recurse() if parts else midi.flat.notes
        for element in elements:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    with open("notes.pkl", "wb") as f:
        pickle.dump(notes, f)
    return notes

# STEP 2: Prepare sequences for training
def prepare_sequences(notes, seq_len=100):
    encoder = LabelEncoder()
    note_ints = encoder.fit_transform(notes)
    n_vocab = len(set(note_ints))
    
    X, y = [], []
    for i in range(len(note_ints) - seq_len):
        X.append(note_ints[i:i+seq_len])
        y.append(note_ints[i+seq_len])

    X = np.reshape(X, (len(X), seq_len, 1)) / float(n_vocab)
    y = to_categorical(y, num_classes=n_vocab)

    return X, y, encoder, n_vocab

# STEP 3: Build the LSTM model
def build_model(input_shape, n_vocab):
    model = Sequential()
    model.add(LSTM(512, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# STEP 4: Generate new music
def generate_music(model, encoder, n_vocab, seq_len=100, length=300):
    with open("notes.pkl", "rb") as f:
        notes = pickle.load(f)
    note_ints = encoder.transform(notes)
    start = np.random.randint(0, len(note_ints) - seq_len)
    pattern = note_ints[start:start+seq_len]
    pattern = np.reshape(pattern, (1, seq_len, 1)) / float(n_vocab)

    generated = []
    for _ in range(length):
        prediction = model.predict(pattern, verbose=0)
        index = np.argmax(prediction)
        result = encoder.inverse_transform([index])[0]
        generated.append(result)

        # Update pattern
        pattern = np.append(pattern[:,1:,:], [[[index / float(n_vocab)]]], axis=1)
    return generated

# STEP 5: Convert generated notes to MIDI file
def create_midi(prediction_output, output_path="output/generated.mid"):
    output_notes = []
    for item in prediction_output:
        if '.' in item:
            notes_in_chord = [int(n) for n in item.split('.')]
            chord_notes = [note.Note(n) for n in notes_in_chord]
            new_chord = chord.Chord(chord_notes)
            output_notes.append(new_chord)
        else:
            output_notes.append(note.Note(item))
    midi_stream = stream.Stream(output_notes)
    midi_stream.write("midi", fp=output_path)
    print(f"Generated music saved to: {output_path}")

# MAIN EXECUTION
if __name__ == "__main__":
    print("Step 1: Loading MIDI files...")
    notes = load_midi_files()

    print("Step 2: Preparing sequences...")
    sequence_length = 100
    X, y, encoder, n_vocab = prepare_sequences(notes, sequence_length)

    print("Step 3: Building the model...")
    model = build_model((X.shape[1], X.shape[2]), n_vocab)

    print("Step 4: Training the model (please wait)...")
    model.fit(X, y, epochs=50, batch_size=64)

    model.save("model.h5")
    print("Model saved as model.h5")

    print("Step 5: Generating music...")
    prediction_output = generate_music(model, encoder, n_vocab, sequence_length)

    print("Step 6: Saving to MIDI...")
    create_midi(prediction_output)
