import ModelMain.mainGAN as MAIN
import environments as env
import Util.SimpleFunc as func



if __name__ == "__main__":
    env.GLOBAL_IS_PUBLISHED = True
    env.GLOBAL_IS_DEBUG = True
    env.GLOBAL_IS_CREATE_RAW_MIDI = True
    env.GLOBAL_IS_CREATE_PIANO_MIDI = True
    env.GLOBAL_IS_CREATE_BIT_IMAGE = True
    env.GLOBAL_IS_CREATE_WEIGHT_IMAGE = True
    #env.GLOBAL_MODEL_LOAD_MODEL_NAME = 'GAN_REELS'
    #env.GLOBAL_MODEL_LOAD_NUM_PUBLISHED = 300
    env.MIDI_THRESHOLD = 0.8
    func.beforeMain()
    func.check_is_cuda_useable()
    MAIN.main_test(env.GLOBAL_MODEL_LOAD_NUM_PUBLISHED, env.GLOBAL_MODEL_LOAD_MODEL_NAME)
