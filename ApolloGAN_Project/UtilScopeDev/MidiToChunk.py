import Util.MidiLoader as midi
import environments as env
import os
import pickle
import numpy as np

def makeChunkFromMidiFolder(srcFolderPath, dstFolderPath):
    print(srcFolderPath, "trying")
    chunkSize = 512
    chunkNumber = -1
    chunkLoadNum = 0
    dataPathList = os.listdir(srcFolderPath)
    for i, dataPath in enumerate(dataPathList):
        print(i, "trying")
        srcPath = srcFolderPath + dataPath
        bitArray = midi.getBitArrayFromMidi(srcPath)

        for layback in range(0, 4):
            if chunkNumber != int(chunkLoadNum / chunkSize):
                chunkNumber = int(chunkLoadNum / chunkSize)
                file = open(dstFolderPath + "-{}.data".format(str(chunkNumber).rjust(4, '0')), 'wb')

            tempBitArray = midi.cloneBitArray(layback, bitArray)
            pickle.dump(tempBitArray.tolist(), file)
            chunkLoadNum += 1

            if chunkLoadNum % chunkSize == 0:
                file.close()



#layback : 4
makeChunkFromMidiFolder('../{}/Data_midiNotinghamClassify/jigs/'.format(env.DATA_INPUT_PATH), "../{}/Data_midiTrain/jigs/chunkBit512".format(env.DATA_INPUT_PATH))
#layback : 3
#makeChunkFromMidiFolder('../{}/Data_midiNotinghamClassify/reels/'.format(env.DATA_INPUT_PATH), "../{}/Data_midiTrain/reels/chunkBit512".format(env.DATA_INPUT_PATH))
