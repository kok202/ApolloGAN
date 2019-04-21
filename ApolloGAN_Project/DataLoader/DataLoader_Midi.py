import os
import numpy as np
import environments as env
import pickle
from Util.SimpleFunc import print_debug
from torch.utils.data import DataLoader as DataLoader
from torch.utils.data import Dataset as DataSet

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Data Loader
# written by kok202
#--------------------------------------------------------------------------------------------------
# class MusicDataset
# class MusicDataLoader
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# MusicDataset
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
class MusicDataset(DataSet):
    def __init__(self):
        self.data_sets = []
        ''''''''''''''''''''''''''''''''''''''''''''' Get all data path '''''''''''''''''''''''''''''''''''''''''''''
        loaderPath = '{}/{}/'.format(env.DATA_INPUT_PATH, env.DATA_LOADER_PATH)
        dataPathList = os.listdir(loaderPath)
        loadFileIndex = 0

        ''''''''''''''''''''''''''''''''''''''''''''' Data loading '''''''''''''''''''''''''''''''''''''''''''''
        print_debug("- Music data loading Start.-")
        for dataPath in dataPathList:
            chunkFilePathMelody = loaderPath + dataPath
            file = open(chunkFilePathMelody, 'rb')
            while True:
                try:
                    loadedListData = pickle.load(file)
                    self.data_sets.append(loadedListData)
                    loadFileIndex += 1

                    self.showLoadPercent(loadFileIndex)
                    if loadFileIndex >= env.GLOBAL_DATA_LOAD_NUM:
                        break
                except EOFError:
                    break

            if loadFileIndex >= env.GLOBAL_DATA_LOAD_NUM:
                break
        ''''''''''''''''''''''''''''''''''''''''''''' Preprocessing '''''''''''''''''''''''''''''''''''''''''''''
        self.data_sets = np.array(self.data_sets)
        self.data_sets = self.data_sets.reshape(-1, 1, env.GAN_INPUT_HEIGHT, env.GAN_INPUT_WIDTH)



    def __len__(self):
        return self.data_sets.shape[0]



    def __getitem__(self, idx):
        return self.data_sets[idx]



    def showLoadPercent(self, idx):
        entireDataLength = env.GLOBAL_DATA_LOAD_NUM
        onePercentLength = int(entireDataLength / 100)
        if onePercentLength != 0 and (idx + 1) % onePercentLength == 0:
            print_debug("Music Dataset Loading : {} % \tLoaded data number : {} / {}".format((idx + 1) / onePercentLength, idx, entireDataLength))



'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# MusicDataLoader
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
class MusicDataLoader():
    def __init__(self, batch_size, shuffle, num_worker):
        trainSet = MusicDataset()
        self.batch_size = batch_size
        self.trainLoader = DataLoader(dataset=trainSet, batch_size=batch_size, shuffle=shuffle, num_workers=num_worker)



    def getLoader(self):
        return self.trainLoader



    def getBatchSize(self):
        return self.batch_size