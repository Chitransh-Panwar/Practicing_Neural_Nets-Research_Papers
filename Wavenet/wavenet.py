from torch import nn
import numpy as np 


class CasualDilatedConv1D(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,dilation=1):
        super().__init()
        self.conv1D=nn.conv1d(in_channels,out_channels,kernel_size,dilation=dilation,bias=False,padding='same')
        self.ignoreOutIndex=(kernel_size-1)*dilation
    def forward(self,x):
        return self.conv1D(x)[...,:-self.ignoreOutIndex]
    

class ResBlock(nn.Module):
    def __init__(self,res_channels,skip_channels,kernel_size,dilation):
        super().__init__()
        self.casualDilatedConv1D=CasualDilatedConv1D(res_channels,res_channels,kernel_size,dilation=1)
        self.resConv1D=CasualDilatedConv1D(res_channels,res_channels,kernel_size=1,dilation=1)
        self.skipConv1D=CasualDilatedConv1D(res_channels,skip_channels,kernel_size=1,dilation=1)
        self.tanh=nn.Tanh()
        self.sigmoid=nn.Sigmoid()

    def forward(self,inputx):
        x=self.casualDilatedConv1D(inputx)
        x1=self.tanh(x)
        x2=self.sigmoid(x)
        x=x1*x2
        resOutput=self.resConv1D(x)+inputx
        skipOutput=self.skipConv1D(x)

        return resOutput,skipOutput


class StackofResBlocks(nn.Module):
    def __init__(self,stack_size,layer_size,res_channels,skip_channels,kernel_size):
        super().__init__()
        buildDilationFunc=np.vectorize(self.buildDilation)
        dilations=buildDilationFunc(stack_size,layer_size)
        self.resBlocks=[]
        for dilationperstack in dilations:
            for dilation in dilationperstack:
                self.resBlocks.append(ResBlock(res_channels,skip_channels,kernel_size,dilation))


    def buildDilation(self,stack_size,layer_size):
        dilationsForAllStack=[]
        for stack in range(stack_size):
            dilations=[]
            for layer in range(layer_size):
                dilations.append(2**layer)
            dilationsForAllStack.append(dilations)
        return dilationsForAllStack
      
    def forward(self):
        resOutput=x
        skipOutputs=[]
        for resBlock in self.resBlocks:
            resOutput,skipOutput=resBlock(resOutput)
            skipOutputs.append(skipOutput)
        return resOutput,torch.stack(skipOutputs)


class WaveNet(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stack_size,layer_size):
        super().__init__()
        self.stackResBlock=StackofResBlocks(stack_size,layer_size,in_channels,out_channels,kernel_size)
    def forward(self,x):
        pass


