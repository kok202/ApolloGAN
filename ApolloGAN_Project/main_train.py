import ModelMain.mainGAN as MAIN
import environments as env
import Util.SimpleFunc as func



if __name__ == "__main__":
    env.GLOBAL_IS_PUBLISHED = False
    env.GLOBAL_IS_DEBUG = True
    env.GLOBAL_IS_CREATE_RAW_MIDI = True
    env.GLOBAL_IS_CREATE_PIANO_MIDI = True
    env.GLOBAL_IS_CREATE_BIT_IMAGE = True
    env.GLOBAL_IS_CREATE_WEIGHT_IMAGE = True
    env.MIDI_THRESHOLD = 0.5
    func.beforeMain()
    func.check_is_cuda_useable()
    MAIN.main_train(env.GLOBAL_MODEL_LOAD_NUM_TRAIN, env.GLOBAL_MODEL_LOAD_MODEL_NAME)
