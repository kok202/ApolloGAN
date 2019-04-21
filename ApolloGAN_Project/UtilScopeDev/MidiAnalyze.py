import Util.MidiIO as midi
import environments as env
from mido import MidiFile
from mido.midifiles import tempo2bpm

def info(path):
    midiFile = MidiFile(path)
    print('Ticks per beat : ', midiFile.ticks_per_beat)
    print('BPM : ', tempo2bpm(midiFile.ticks_per_beat))
    print('beat per time', 0.06/tempo2bpm(midiFile.ticks_per_beat) )

    print('Number of Tracks : ', len(midiFile.tracks))
    for i, track in enumerate(midiFile.tracks):
        print('\tTrack {} : {} '.format(i, track.name))
    for i, track in enumerate(midiFile.tracks):
        print('=================================================================================')
        for message in track:
            print(message)




def analyze(path):
    timeListLoaded, noteListLoaded = midi.load(path)
    print(timeListLoaded)
    print(noteListLoaded)
    trackNumber = len(timeListLoaded)
    print("track Number [by time] : ", len(timeListLoaded))
    print("track Number [by note] : ", len(noteListLoaded))
    for i in range(trackNumber):
        print("time number of each track ", i, " : ", len(timeListLoaded[i]))
        print("note number of each track ", i, " : ", len(noteListLoaded[i]))





info('../{}/Data_TestSet/test00.mid'.format(env.DATA_INPUT_PATH))
analyze('../{}/Data_TestSet/test00.mid'.format(env.DATA_INPUT_PATH))
#loaded = midi.load('../{}/Data_MidiOrigin/A Sleepin\' Bee.mid'.format(env.DATA_INPUT_PATH))