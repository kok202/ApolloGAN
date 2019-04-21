import Util.MidiIO as midi
import environments as env
import numpy as np
import os
from PIL import Image

def getBitArrayFromMidi(srcPath):
    timeListLoaded, noteListLoaded = midi.load(srcPath)
    bitArray = np.zeros((env.MIDI_NOTE_DISTRIBUTION, env.MIDI_LENGTH)).astype('uint8')
    for trackCnt in range(len(timeListLoaded)):
        for timeCnt, time in enumerate(timeListLoaded[trackCnt]):
            imageX = int(time / env.MIDI_LENGTH_UNIT)
            for noteCnt in range(len(noteListLoaded[trackCnt][timeCnt])):
                note = noteListLoaded[trackCnt][timeCnt][noteCnt][0]
                duration = noteListLoaded[trackCnt][timeCnt][noteCnt][1]
                imageY = note - env.MIDI_NOTE_DISTRIBUTION_START
                length = int(duration / env.MIDI_LENGTH_UNIT)
                for offset in range(length):
                    if imageX + offset >= env.MIDI_LENGTH:
                        break
                    bitArray[imageY][imageX + offset] = 1
    return bitArray



def getImgArrayFromBitArray(bitArray):
    imgArray = np.zeros((env.MIDI_NOTE_DISTRIBUTION, env.MIDI_LENGTH, 3)).astype('uint8')
    for i in range(env.MIDI_NOTE_DISTRIBUTION):
        for j in range(env.MIDI_LENGTH):
            if bitArray[i][j] >= env.MIDI_THRESHOLD:
                imgArray[i][j][0] = 255
                imgArray[i][j][1] = 255
                imgArray[i][j][2] = 255
    return imgArray



def getImageFromBitArray(bitArray, dstPath):
    imgArray = getImgArrayFromBitArray(bitArray)
    image = Image.fromarray(imgArray).convert('RGB')
    image.save(dstPath)



def getImageFromMidi(srcPath, dstPath):
    bitArray = getBitArrayFromMidi(srcPath)
    imgArray = getImgArrayFromBitArray(bitArray)
    image = Image.fromarray(imgArray).convert('RGB')
    image.save(dstPath)



def getImageFromWeight(weightArray, dstPath):
    imgArray = np.zeros((env.MIDI_NOTE_DISTRIBUTION, env.MIDI_LENGTH, 3)).astype('uint8')
    for i in range(env.MIDI_NOTE_DISTRIBUTION):
        for j in range(env.MIDI_LENGTH):
            imgArray[i][j][0] = int(255 * weightArray[i][j])
            imgArray[i][j][1] = int(255 * weightArray[i][j])
            imgArray[i][j][2] = int(255 * weightArray[i][j])
    image = Image.fromarray(imgArray).convert('RGB')
    image.save(dstPath)



def getMidiFromBitArray(bitArray, dstPath):
    timeList = []
    noteList = []
    noteOn = [-1 for i in range(env.MIDI_NOTE_DISTRIBUTION)]
    noteOnTime = [-1 for i in range(env.MIDI_NOTE_DISTRIBUTION)]
    noteIndex = 0

    for j in range(env.MIDI_LENGTH):
        currentTimingNote = []
        for i in range(env.MIDI_NOTE_DISTRIBUTION):
            if bitArray[i][j] >= env.MIDI_THRESHOLD:
                if noteOn[i] == -1:
                    # When find out note which should be pressed at this timing
                    noteOn[i] = noteIndex
                    noteOnTime[i] = j * env.MIDI_LENGTH_UNIT
                    
                    # If there is no note at this timing, appendding it
                    if len(currentTimingNote) == 0:
                        timeList.append(j * env.MIDI_LENGTH_UNIT)
                        
                    # Append it to list which is saving pressed note at this timing
                    currentTimingNote.append([i, env.MIDI_LENGTH_UNIT])
            else:
                if noteOn[i] != -1:
                    refNum = noteOn[i]
                    for note in noteList[refNum]:
                        if note[0] == i:
                            note[1] = j * env.MIDI_LENGTH_UNIT - noteOnTime[i]
                            break
                    noteOn[i] = -1
        if len(currentTimingNote) > 0:
            noteList.append(currentTimingNote)
            noteIndex += 1

    for i in range(len(noteList)):
        for j in range(len(noteList[i])):
            noteList[i][j][0] += env.MIDI_NOTE_DISTRIBUTION_START

    midi.save(dstPath, [timeList], [noteList])



def getBitArrayFromHotArray(hotArray):
    bitArray = np.zeros((env.MIDI_NOTE_DISTRIBUTION, env.MIDI_LENGTH)).astype('uint8')
    for i in range(env.MIDI_NOTE_DISTRIBUTION):
        for j in range(env.MIDI_LENGTH):
            if hotArray[j][i] >= env.MIDI_THRESHOLD:
                bitArray[i][j] = 1
    return bitArray



def getHotArrayFromBitArray(bitArray):
    hotArray = []
    for j in range(env.MIDI_LENGTH):
        hotVector = [0 for i in range(env.MIDI_NOTE_DISTRIBUTION)]
        for i in range(env.MIDI_NOTE_DISTRIBUTION):
            if bitArray[i][j] >= env.MIDI_THRESHOLD:
                hotVector[i] = 1
        hotArray.append(hotVector)
    return np.array(hotArray)



def cloneBitArray(layback, src):
    dst = np.zeros((env.MIDI_NOTE_DISTRIBUTION, env.MIDI_LENGTH)).astype('uint8')
    for i in range(env.MIDI_NOTE_DISTRIBUTION):
        for j in range(layback, env.MIDI_LENGTH):
            dst[i][j] = src[i][j-layback]
    return dst


'''
# test code of functions
bitArray = getBitArrayFromMidi('../{}/Data_TestSet/test00.mid'.format(env.DATA_INPUT_PATH))
print(bitArray.shape)
print(bitArray.tolist())
hotArray = getHotArrayFromBitArray(bitArray)
print(hotArray.shape)
print(hotArray.tolist())
bitArray = getBitArrayFromHotArray(hotArray)
print(bitArray.shape)
print(bitArray.tolist())
getMidiFromBitArray(bitArray, '../{}/Data_TestSet/test00rrre.mid'.format(env.DATA_INPUT_PATH))
'''