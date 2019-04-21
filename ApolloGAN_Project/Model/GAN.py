import torch
import torch.nn as nn
import Util.SimpleFunc as func
import environments as env
import os

# DCGAN custom complete
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Discriminator
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        inputDimension = env.GAN_INPUT_CHANNEL
        depthSize = env.GAN_FILTER_DEPTH
        kernelSize = env.GAN_KERNEL_SIZE
        strideSize = env.GAN_STRIDE_SIZE
        paddingSize = env.GAN_PADDING_SIZE
        self.Layer1_conv = nn.Conv2d(in_channels=inputDimension, out_channels=depthSize, kernel_size=kernelSize, stride=strideSize, padding=paddingSize, bias=False)
        self.Layer1_relu = nn.LeakyReLU(0.2, inplace=True)
        self.Layer2_conv = nn.Conv2d(in_channels=depthSize, out_channels=depthSize*2, kernel_size=kernelSize, stride=strideSize, padding=paddingSize, bias=False)
        self.Layer2_norm = nn.BatchNorm2d(depthSize * 2)
        self.Layer2_relu = nn.LeakyReLU(0.2)
        self.Layer3_conv = nn.Conv2d(in_channels=depthSize*2, out_channels=depthSize*4, kernel_size=kernelSize, stride=strideSize, padding=paddingSize, bias=False)
        self.Layer3_norm = nn.BatchNorm2d(depthSize * 4)
        self.Layer3_relu = nn.LeakyReLU(0.2)
        self.Layer4_conv = nn.Conv2d(in_channels=depthSize*4, out_channels=depthSize*8, kernel_size=kernelSize, stride=strideSize, padding=paddingSize, bias=False)
        self.Layer4_norm = nn.BatchNorm2d(depthSize * 8)
        self.Layer4_relu = nn.LeakyReLU(0.2)
        self.Layer5_conv = nn.Conv2d(in_channels=depthSize*8, out_channels=depthSize, kernel_size=kernelSize, stride=strideSize, padding=paddingSize, bias=False)
        self.Layer5_norm = nn.BatchNorm2d(depthSize)
        self.Layer5_relu = nn.LeakyReLU(0.2)
        self.Layer6_fcnn = nn.Linear(2048, 512)
        self.Layer6_relu = nn.LeakyReLU(0.2)
        self.Layer7_fcnn = nn.Linear(512, 1)
        self.Layer7_sigmoid = nn.Sigmoid()



    def forward(self, batch_size, x):
        conv1 = self.Layer1_conv(x)
        relu1 = self.Layer1_relu(conv1)
        conv2 = self.Layer2_conv(relu1)
        norm2 = self.Layer2_norm(conv2)
        relu2 = self.Layer2_relu(norm2)
        conv3 = self.Layer3_conv(relu2)
        norm3 = self.Layer3_norm(conv3)
        relu3 = self.Layer3_relu(norm3)
        conv4 = self.Layer4_conv(relu3)
        norm4 = self.Layer4_norm(conv4)
        relu4 = self.Layer4_relu(norm4)
        conv5 = self.Layer5_conv(relu4)
        norm5 = self.Layer5_norm(conv5)
        relu5 = self.Layer5_relu(norm5)
        relu5 = relu5.view(-1, 2048)
        fcnn6 = self.Layer6_fcnn(relu5)
        relu6 = self.Layer6_relu(fcnn6)
        fcnn7 = self.Layer7_fcnn(relu6)
        result = self.Layer7_sigmoid(fcnn7)
        return result.view(-1, 1).squeeze(1)





'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Generator
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        inputDimension = env.GAN_NOISE_CHANNEL
        outputDimension = env.GAN_INPUT_CHANNEL
        depthSize = env.GAN_FILTER_DEPTH
        kernelSize = env.GAN_KERNEL_SIZE
        strideSize = env.GAN_STRIDE_SIZE
        paddingSize = env.GAN_PADDING_SIZE
        # [-1, 7, 2, 8] -> [-1, 1024, 4, 16]
        self.Layer1_conv = nn.ConvTranspose2d(in_channels=inputDimension, out_channels=depthSize * 8, kernel_size=kernelSize, stride=strideSize, padding=paddingSize, bias=False)
        self.Layer1_norm = nn.BatchNorm2d(depthSize * 8)
        self.Layer1_relu = nn.LeakyReLU(0.2)
        self.Layer2_conv = nn.ConvTranspose2d(in_channels=depthSize * 8, out_channels=depthSize * 4, kernel_size=kernelSize, stride=strideSize, padding=paddingSize, bias=False)
        self.Layer2_norm = nn.BatchNorm2d(depthSize * 4)
        self.Layer2_relu = nn.LeakyReLU(0.2)
        self.Layer3_conv = nn.ConvTranspose2d(in_channels=depthSize * 4, out_channels=depthSize * 2, kernel_size=kernelSize, stride=strideSize, padding=paddingSize, bias=False)
        self.Layer3_norm = nn.BatchNorm2d(depthSize * 2)
        self.Layer3_relu = nn.LeakyReLU(0.2)
        self.Layer4_conv = nn.ConvTranspose2d(in_channels=depthSize * 2, out_channels=depthSize, kernel_size=kernelSize, stride=strideSize, padding=paddingSize, bias=False)
        self.Layer4_norm = nn.BatchNorm2d(depthSize)
        self.Layer4_relu = nn.LeakyReLU(0.2)
        self.Layer5_conv = nn.ConvTranspose2d(in_channels=depthSize, out_channels=outputDimension, kernel_size=kernelSize, stride=strideSize, padding=paddingSize, bias=False)
        self.Layer5_sigmoid = nn.Sigmoid()



    def forward(self, batch_size, x):
        conv1 = self.Layer1_conv(x)
        relu1 = self.Layer1_relu(conv1)
        conv2 = self.Layer2_conv(relu1)
        norm2 = self.Layer2_norm(conv2)
        relu2 = self.Layer2_relu(norm2)
        conv3 = self.Layer3_conv(relu2)
        norm3 = self.Layer3_norm(conv3)
        relu3 = self.Layer3_relu(norm3)
        conv4 = self.Layer4_conv(relu3)
        norm4 = self.Layer4_norm(conv4)
        relu4 = self.Layer4_relu(norm4)
        conv5 = self.Layer5_conv(relu4)
        result = self.Layer5_sigmoid(conv5)
        return result





'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# GAN
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
class GAN:
    def __init__(self, loadNumber, modelName):
        self.d_model = Discriminator()
        self.g_model = Generator()
        self.loadNumber = loadNumber
        self.modelName = modelName
        if loadNumber != -1:
            loadDirectory = func.getLoadDirectory()
            self.d_model.load_state_dict(torch.load(loadDirectory + self.modelName + '_discriminator{}.pkl'.format(loadNumber), map_location=lambda storage, loc: storage))
            self.g_model.load_state_dict(torch.load(loadDirectory + self.modelName + '_generator{}.pkl'.format(loadNumber), map_location=lambda storage, loc: storage))
            func.print_debug("{}'th Model Data Loaded".format(loadNumber))
        self.criterion_MSE = nn.MSELoss()
        self.criterion_BCE = nn.BCELoss()
        self.criterion_HEL = nn.HingeEmbeddingLoss()
        self.d_optimizer = torch.optim.Adam(self.d_model.parameters(), lr=env.GAN_LEARNING_RATE, betas=(0.5, 0.999))
        self.g_optimizer = torch.optim.Adam(self.g_model.parameters(), lr=env.GAN_LEARNING_RATE, betas=(0.5, 0.999))
        self.d_model = func.cuda(self.d_model)
        self.g_model = func.cuda(self.g_model)
        if loadNumber == -1:
            self.d_model.apply(self.weights_init)
            self.g_model.apply(self.weights_init)



    def weights_init(self, model):
        classname = model.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(model.weight.data, 1.0, 0.02)
            nn.init.constant_(model.bias.data, 0)



    def criterion_Feat(self, real_feats, fake_feats):
        retloss = 0
        for i in range(len(real_feats)):
            real_feat = real_feats[i]
            fake_feat = fake_feats[i]
            loss_MS = (real_feat.mean(0) - fake_feat.mean(0)) * \
                      (real_feat.mean(0) - fake_feat.mean(0))
            oneMatrix = func.make_cuda_var(torch.ones(loss_MS.size()))
            loss = self.criterion_HEL(loss_MS, oneMatrix)
            retloss += loss
        return retloss



    def generate(self, batch_size):
        self.g_model.eval()
        fake_Tensor = torch.randn(batch_size, env.GAN_NOISE_CHANNEL, env.GAN_NOISE_HEIGHT, env.GAN_NOISE_WIDTH)
        fake_VarCud = func.make_cuda_var(fake_Tensor)
        fake_GenOut = self.g_model(batch_size, fake_VarCud)
        self.g_model.train()
        return fake_GenOut



    def accuracy(self, batch_size, input):
        Labels_real = func.make_cuda_var(torch.ones(batch_size))
        Labels_fake = func.make_cuda_var(torch.zeros(batch_size))

        real_Tensor = func.np_to_floatTensor(input)
        real_VarCud = func.make_cuda_var(real_Tensor.view(batch_size, env.GAN_INPUT_CHANNEL, env.GAN_INPUT_HEIGHT, env.GAN_INPUT_WIDTH))
        real_d_outs = self.d_model(batch_size, real_VarCud)
        real_d_loss = self.criterion_BCE(real_d_outs, Labels_real)  # real labels set all 1

        fake_Tensor = torch.randn(batch_size, env.GAN_NOISE_CHANNEL, env.GAN_NOISE_HEIGHT, env.GAN_NOISE_WIDTH)
        fake_VarCud = func.make_cuda_var(fake_Tensor)
        fake_GenOut = self.g_model(batch_size, fake_VarCud)
        fake_d_outs = self.d_model(batch_size, fake_GenOut.detach())  # 0 or 1
        fake_d_loss = self.criterion_BCE(fake_d_outs, Labels_fake)  # fake labels set all 0
        d_loss = real_d_loss * 0.5 + fake_d_loss * 0.5
        return real_d_outs, fake_d_outs, d_loss



    def train(self, batch_size, input):
        # change train method -> eval method

        ''''''''''''''''''''''''''' Train Discrimitor '''''''''''''''''''''''''''''''''''
        for i in range(env.GLOBAL_TRAIN_BALANCE_D_STEPS):
            self.d_model.zero_grad()
            self.g_model.zero_grad()
            Labels_real = func.make_cuda_var(torch.ones(batch_size))
            Labels_fake = func.make_cuda_var(torch.zeros(batch_size))

            real_Tensor = func.np_to_floatTensor(input)
            real_VarCud = func.make_cuda_var(real_Tensor.view(batch_size, env.GAN_INPUT_CHANNEL, env.GAN_INPUT_HEIGHT, env.GAN_INPUT_WIDTH))
            real_d_outs = self.d_model(batch_size, real_VarCud)
            real_d_loss = self.criterion_BCE(real_d_outs, Labels_real)  # real labels set all 1

            fake_Tensor = torch.randn(batch_size, env.GAN_NOISE_CHANNEL, env.GAN_NOISE_HEIGHT, env.GAN_NOISE_WIDTH)
            fake_VarCud = func.make_cuda_var(fake_Tensor)
            fake_GenOut = self.g_model(batch_size, fake_VarCud)
            fake_d_outs = self.d_model(batch_size, fake_GenOut.detach())  # 0 or 1
            fake_d_loss = self.criterion_BCE(fake_d_outs, Labels_fake)  # fake labels set all 0
            d_loss = real_d_loss * 0.5 + fake_d_loss * 0.5
            d_loss.backward(retain_graph=True)
            self.d_optimizer.step()

        ''''''''''''''''''''''''''' Train Generator '''''''''''''''''''''''''''''''''''
        for i in range(env.GLOBAL_TRAIN_BALANCE_G_STEPS):
            self.g_model.zero_grad()
            fake_d_outs = self.d_model(batch_size, fake_GenOut)
            g_lossFraud = self.criterion_BCE(fake_d_outs, Labels_real)
            g_loss = g_lossFraud
            g_loss.backward()
            self.g_optimizer.step()
        return d_loss, g_loss



    def saveModel(self, number):
        saveDirectory = './ModelSave/' + env.GLOBAL_MODEL_SAVE_DIRECTORY_TRAIN
        if not(os.path.isdir(saveDirectory)):
            os.makedirs(os.path.join(saveDirectory))
        
        saveDirectory = saveDirectory + '/'
        torch.save(self.g_model.state_dict(), saveDirectory + self.modelName + '_generator{}.pkl'.format(number))
        torch.save(self.d_model.state_dict(), saveDirectory + self.modelName + '_discriminator{}.pkl'.format(number))
