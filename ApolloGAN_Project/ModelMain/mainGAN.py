import Util.SimpleFunc as func
import environments as env
import os
import random
import time
import Model.GAN as GAN
import DataLoader.DataLoader_Midi as DataLoader
import Util.MidiLoader as midi
from Util.MidiInstChanger import changeToPiano
from Util.SimpleFunc import print_debug


def main_train(modelLoadNumber, modelName):
    print_debug("Data load path : {}".format(env.DATA_LOADER_PATH))
    print_debug("Data load size : {}".format(env.GLOBAL_DATA_LOAD_NUM))
    print_debug("Data batch size : {}".format(env.GLOBAL_BATCH_SIZE))
    print_debug("Model Name : {}".format(modelName))

    model = GAN.GAN(modelLoadNumber, modelName)
    loader = DataLoader.MusicDataLoader(env.GAN_BATCH_SIZE, True, 0)
    modelLoadNumber = 0 if (modelLoadNumber == -1) else modelLoadNumber
    for epoch in range(modelLoadNumber, env.GLOBAL_EPOCH_NUM):
        train(model, loader, epoch)
        generating(model, epoch)
        modelSave(model, epoch)





def main_test(modelLoadNumber, modelName):
    model = GAN.GAN(modelLoadNumber, modelName)
    generating(model, modelLoadNumber)





'''########################################################################################################################################'''
def train(model, loader, epoch):
    print_debug(time.ctime())
    print_debug("epoch : {} Training Start----------------------------------------------------".format(epoch))
    for i, (miniBatch) in enumerate (loader.getLoader()):
        miniBatch = miniBatch.numpy()
        d_loss, g_loss = model.train(loader.getBatchSize(), miniBatch)
        if i % env.GLOBAL_RECORD_STEPS_FOR_LOSS == 0:
            d_loss = func.tensor_to_numpy(d_loss)
            g_loss = func.tensor_to_numpy(g_loss)
            print_debug("step : {}, \tDiscriminator loss : {}, \tGenerator loss : {}".format(i, d_loss, g_loss))
            accuracy(model, miniBatch)





def generating(model, epoch):
    if env.GLOBAL_IS_PUBLISHED == False and epoch % env.GLOBAL_EPOCH_GEN_TERM != 0:
        return
    if env.GLOBAL_IS_PUBLISHED == True:
        epoch = random.randrange(1, 100000000)

    data = model.generate(1)
    data = func.tensor_to_numpy(data)
    data = data[0][0]

    saveDirectory = func.getGeneratingDirectory()
    saveDirectory_MidiRaw = '{}/MidiRaw'.format(saveDirectory)
    saveDirectory_MidiPiano = '{}/MidiPiano'.format(saveDirectory)
    saveDirectory_ImgBit = '{}/ImageBit'.format(saveDirectory)
    saveDirectory_ImgWeight = '{}/ImageWeight'.format(saveDirectory)
    print_debug("epoch : {} Trying saving generated data".format(epoch))

    if not (os.path.isdir(saveDirectory_MidiRaw)):
        os.makedirs(os.path.join(saveDirectory_MidiRaw))
    if not (os.path.isdir(saveDirectory_MidiPiano)):
        os.makedirs(os.path.join(saveDirectory_MidiPiano))
    if not (os.path.isdir(saveDirectory_ImgBit)):
        os.makedirs(os.path.join(saveDirectory_ImgBit))
    if not (os.path.isdir(saveDirectory_ImgWeight)):
        os.makedirs(os.path.join(saveDirectory_ImgWeight))

    dstMidRaw = '{}/GeneratedMidi-{}.mid'.format(saveDirectory_MidiRaw, epoch)
    dstMidPiano = '{}/GeneratedMidi-{}.mid'.format(saveDirectory_MidiPiano, epoch)
    dstImgBit = '{}/GeneratedMidi-{}.png'.format(saveDirectory_ImgBit, epoch)
    dstImgWeight = '{}/GeneratedMidi_weight-{}.png'.format(saveDirectory_ImgWeight, epoch)

    if env.GLOBAL_IS_CREATE_RAW_MIDI == True:
        midi.getMidiFromBitArray(data, dstMidRaw)
    if env.GLOBAL_IS_CREATE_PIANO_MIDI == True:
        changeToPiano(dstMidRaw, dstMidPiano)
    if env.GLOBAL_IS_CREATE_BIT_IMAGE == True:
        midi.getImageFromMidi(dstMidRaw, dstImgBit)
    if env.GLOBAL_IS_CREATE_WEIGHT_IMAGE == True:
        midi.getImageFromWeight(data, dstImgWeight)
    if env.GLOBAL_IS_PUBLISHED == True:
        print(os.path.abspath(dstImgBit))





def modelSave(model, epoch):
    if epoch % env.GLOBAL_EPOCH_SAVE_TERM != 0:
        return
    model.saveModel(epoch)





def accuracy(model, realData):
    # print discriminator accuracy
    if env.GLOBAL_BATCH_SIZE < env.GLOBAL_BATCH_SIZE_FOR_ACCURACY:
        env.GLOBAL_BATCH_SIZE_FOR_ACCURACY = env.GLOBAL_BATCH_SIZE
    realData = realData[0:env.GLOBAL_BATCH_SIZE_FOR_ACCURACY]
    accuracy_real, accuracy_fake, accuracy_loss = model.accuracy(env.GLOBAL_BATCH_SIZE_FOR_ACCURACY, realData)
    accuracy_real = func.tensor_to_numpy(accuracy_real)
    accuracy_fake = func.tensor_to_numpy(accuracy_fake)
    accuracy_loss = func.tensor_to_numpy(accuracy_loss)

    threshold = 0.3
    answer_real = [0, 0, 0]  # validate result : [real, ambiguous, fake]
    answer_fake = [0, 0, 0]  # validate result : [real, ambiguous, fake]
    for i in range(accuracy_real.size):
        if accuracy_real[i] > 1 - threshold:
            answer_real[0] += 1
        elif accuracy_real[i] < threshold:
            answer_real[2] += 1
        else:
            answer_real[1] += 1

        if accuracy_fake[i] > 1 - threshold:
            answer_fake[0] += 1
        elif accuracy_fake[i] < threshold:
            answer_fake[2] += 1
        else:
            answer_fake[1] += 1

    print_debug("\t\t\tAccuracy for real data : real : {}, ambg : {}, fake : {}".format(answer_real[0], answer_real[1], answer_real[2]))
    print_debug("\t\t\tAccuracy for fake data : real : {}, ambg : {}, fake : {}".format(answer_fake[0], answer_fake[1], answer_fake[2]))
    print_debug("\t\t\tAccuracy detail")  # just use tolist for getting good looking
    print_debug("\t\t\t" + str(accuracy_real.tolist()))
    print_debug("\t\t\t" + str(accuracy_fake.tolist()))
    print_debug("\t\t\t" + str(accuracy_loss.tolist()))