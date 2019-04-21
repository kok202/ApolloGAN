import Util.MidiLoader as midi
import os
import environments as env



def makeImageFromMidiFolder(srcFolderPath, dstFolderPath):
    dataPathList = os.listdir(srcFolderPath)
    for dataPath in dataPathList:
        srcPath = srcFolderPath + dataPath
        srcBitArray = midi.getBitArrayFromMidi(srcPath)
        for layback in range(0, 3):
            dstPath = dstFolderPath.format(str(layback)+dataPath)
            tempBitArray = midi.cloneBitArray(layback, srcBitArray)
            midi.getImageFromBitArray(tempBitArray, dstPath)



#layback : 4
#makeImageFromMidiFolder('../{}/Data_midiNotinghamClassify/jigs/'.format(env.DATA_INPUT_PATH), '../{}/Data_midiNotinghamImage/jigs/'.format(env.DATA_INPUT_PATH) + '{}.png')
#layback : 3
makeImageFromMidiFolder('../{}/Data_midiNotinghamClassify/reels/'.format(env.DATA_INPUT_PATH), '../{}/Data_midiNotinghamImage/reels/'.format(env.DATA_INPUT_PATH) + '{}.png')