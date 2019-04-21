
def getConv1DOutputSize(inputSize, kernelSize, stride, padding):
    output = int((inputSize + 2 * padding - kernelSize) / stride + 1)
    return output

def getConv2DOutputSize(inputSize, kernelSize, stride, padding):
    output = [getConv1DOutputSize(inputSize[0], kernelSize, stride, padding),
              getConv1DOutputSize(inputSize[1], kernelSize, stride, padding)]
    return output

def getConvT1DOutputSize(inputSize, kernelSize, stride, padding):
    output = int((inputSize - 1) * stride + kernelSize - 2 * padding)
    return output

def getConvT2DOutputSize(inputSize, kernelSize, stride, padding):
    output = [getConvT1DOutputSize(inputSize[0], kernelSize, stride, padding),
              getConvT1DOutputSize(inputSize[1], kernelSize, stride, padding)]
    return output

print("Conv")
input = [64, 64]
print(input)
input = getConv2DOutputSize(input, 4, 2, 1)
print(input)
input = getConv2DOutputSize(input, 4, 2, 1)
print(input)
input = getConv2DOutputSize(input, 4, 2, 1)
print(input)
input = getConv2DOutputSize(input, 4, 2, 1)
print(input)

print("Conv Transpose")
input = [1, 1]
input = getConvT2DOutputSize(input, 4, 2, 0)
print(input)
input = getConvT2DOutputSize(input, 4, 2, 1)
print(input)
input = getConvT2DOutputSize(input, 4, 2, 1)
print(input)
input = getConvT2DOutputSize(input, 4, 2, 1)
print(input)
input = getConvT2DOutputSize(input, 4, 2, 1)
print(input)