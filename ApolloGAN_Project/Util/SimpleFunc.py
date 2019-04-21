import numpy as np
import torch
import random
import environments as env
import os
from torch.autograd import Variable



def print_debug(msg):
    if env.GLOBAL_IS_DEBUG == True:
        print(msg)



def check_is_cuda_useable():
    if env.ACCEPT_USING_CUDA == env.SET_CUDA_AVAIL:
        if torch.cuda.is_available():
            env.ACCEPT_USING_CUDA = int(input("SYSTEM : Cuda can work on this computer. do you want it to use? [no(0) / yes(1)]: "))
        else:
            print_debug("SYSTEM : Cuda doesn't work on this computer")
            env.ACCEPT_USING_CUDA = env.SET_CUDA_FALSE

    if env.ACCEPT_USING_CUDA == env.SET_CUDA_TRUE:
        if torch.cuda.is_available() == False:
            env.ACCEPT_USING_CUDA = env.SET_CUDA_FALSE

    if env.ACCEPT_USING_CUDA == env.SET_CUDA_FALSE:
        print_debug("SYSTEM : running program without cuda.")
    else:
        print_debug("SYSTEM : running program with cuda.")



def cuda(x):
    if torch.cuda.is_available():
        x = x.cuda() if (env.ACCEPT_USING_CUDA==1) else x
    return x



def make_cuda_var(x):
    return Variable(cuda(x))



def tensor_to_numpy(x):
    x = x.data.cpu().numpy()
    # for getting data from gpu to cpu, if running at gpu,
    # it's okay although running at cpu, because cpu data has same method cpu()
    # if torch.cuda.is_available() and env.ACCEPT_USING_CUDA == 1:
    #     x = x.data.numpy()
    # else:
    #     x = x.numpy()
    return x




def make_cuda_var_randn(batch_size, length):
    return Variable(cuda(torch.randn(batch_size, length)))



def np_to_floatTensor(x):
    return torch.FloatTensor(x)



def monkeyPatch():
    import torch._utils
    try:
        torch._utils._rebuild_tensor_v2
    except AttributeError:
        def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
            tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
            tensor.requires_grad = requires_grad
            tensor._backward_hooks = backward_hooks
            return tensor
        torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2



def beforeMain():
    manualSeed = 999
    if env.GLOBAL_IS_PUBLISHED == True:
        manualSeed = random.randint(1, 1000000)
    print_debug("Random Seed : {}".format(manualSeed))
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    monkeyPatch()



def getGeneratingDirectory():
    ''' For getting data save path '''
    if env.GLOBAL_IS_PUBLISHED == True:
        if env.GLOBAL_IS_DEBUG == True:
            saveDirectory = '{}/GeneratedPublished'.format(env.DATA_OUTPUT_PATH)
        else:
            saveDirectory = format(env.DATA_PUBLISHED_PATH)
    else:
        saveDirectory = '{}/Generated/'.format(env.DATA_OUTPUT_PATH)
    if not(os.path.isdir(saveDirectory)):
        os.makedirs(os.path.join(saveDirectory))
    return saveDirectory



def getLoadDirectory():
    ''' For getting saved model data path '''
    if env.GLOBAL_IS_PUBLISHED == True:
        loadDirectory = './ModelSave/' + env.GLOBAL_MODEL_LOAD_DIRECTORY_PUBLISHED + '/'
    else:
        loadDirectory = './ModelSave/' + env.GLOBAL_MODEL_LOAD_DIRECTORY_TRAIN + '/'
    return loadDirectory