'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# TOTAL
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
SET_CUDA_AVAIL = -1
SET_CUDA_FALSE = 0
SET_CUDA_TRUE = 1
ACCEPT_USING_CUDA = SET_CUDA_TRUE
GLOBAL_IS_DEBUG = True
GLOBAL_IS_PUBLISHED = False
GLOBAL_IS_CREATE_RAW_MIDI = True
GLOBAL_IS_CREATE_PIANO_MIDI = True
GLOBAL_IS_CREATE_BIT_IMAGE = True
GLOBAL_IS_CREATE_WEIGHT_IMAGE = True

GLOBAL_MODEL_LOAD_NUM_TRAIN = -1
GLOBAL_MODEL_LOAD_NUM_PUBLISHED = -1
GLOBAL_MODEL_LOAD_MODEL_NAME = 'GAN'
GLOBAL_MODEL_LOAD_DIRECTORY_PUBLISHED = 'Published'
GLOBAL_MODEL_LOAD_DIRECTORY_TRAIN = 'MIDI_GAN'
GLOBAL_MODEL_SAVE_DIRECTORY_TRAIN = 'MIDI_GAN'

GLOBAL_EPOCH_NUM = 1000
GLOBAL_EPOCH_GEN_TERM = 10
GLOBAL_EPOCH_SAVE_TERM = 60
GLOBAL_RECORD_STEPS_FOR_LOSS = 32

GLOBAL_DATA_LOAD_BY_CHUNK = True
GLOBAL_DATA_LOAD_NUM = 1024
GLOBAL_BATCH_SIZE = 16
GLOBAL_BATCH_SIZE_FOR_ACCURACY = 8

GLOBAL_LEARNING_RATE = 0.0002
GLOBAL_TRAIN_BALANCE_D_STEPS = 1  # Recommended : don't try to balance
GLOBAL_TRAIN_BALANCE_G_STEPS = 1  # Recommended : don't try to balance

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# SETTING
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
MIDI_LENGTH = 256
MIDI_LENGTH_UNIT = 512
MIDI_THRESHOLD = 0.6

# note range of mid file : 0~127
# but midi file's notes are distributed at 30 to 90
# generally 30~60 : chord, 50~90 : melody
MIDI_NOTE_NUMBER = 128
MIDI_NOTE_DISTRIBUTION_START = 30
MIDI_NOTE_DISTRIBUTION_END = 94
MIDI_NOTE_DISTRIBUTION = MIDI_NOTE_DISTRIBUTION_END - MIDI_NOTE_DISTRIBUTION_START
DATA_INPUT_PATH = '../ApolloGAN_Database'
DATA_OUTPUT_PATH = './DataOutput'
DATA_LOADER_PATH = 'Data_midiTrain/reels'
DATA_PUBLISHED_PATH = 'ABSOLUTE_PATH_OF_MODEL_DATA'

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# DCGAN
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
GAN_BATCH_SIZE = GLOBAL_BATCH_SIZE
GAN_LEARNING_RATE = GLOBAL_LEARNING_RATE
GAN_NOISE_CHANNEL = 7
GAN_NOISE_HEIGHT = 2  # Don't mind ( fixed value because of model )
GAN_NOISE_WIDTH = 8  # Don't mind ( fixed value because of model )
GAN_INPUT_WIDTH = MIDI_LENGTH
GAN_INPUT_HEIGHT = MIDI_NOTE_DISTRIBUTION
GAN_INPUT_CHANNEL = 1  # (for channel number)
GAN_FILTER_DEPTH = 128
GAN_KERNEL_SIZE = 4
GAN_STRIDE_SIZE = 2
GAN_PADDING_SIZE = 1