import os
import torch
from torch.nn import AdaptiveAvgPool2d, Conv2d, ConvTranspose2d,BatchNorm2d,Linear,LeakyReLU, Dropout2d
import torch.nn.functional as F
import torch.nn as nn
from torchsummary import summary 
from cbam import CBAM

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


    
class AtentionBlock(nn.Module):
    def __init__(self,in_channels):
        super(AtentionBlock,self).__init__() 
        self.global_avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Linear(in_channels,in_channels//2)
        self.fc2 = Linear(in_channels//2,in_channels)
    def forward(self,x):
        batch_size,channels,_,_ = x.size() 
        se = self.global_avg_pool(x).view(batch_size,channels) # Global average pooling
        se = F.selu(self.fc1(se)) # Dense layer with SELU activation
        # print(se.shape)
        se = torch.sigmoid(self.fc2(se)) # Dense with sigmoid activation
        se = se.view(batch_size,channels,1,1) #reshape to (batch_size,channels,1,1)
        se = x * se 
        return se 

class ResidualBlock(nn.Module):
    def __init__(self,in_channels,filters):
        super(ResidualBlock,self).__init__() 
        self.conv1 = Conv2d(in_channels=in_channels,
                            out_channels=filters,
                            kernel_size=3,
                            stride=1,padding=1,bias=True)
        self.bn1  = BatchNorm2d(filters)
        self.leaky_relu = LeakyReLU()

        self.conv2 = Conv2d(in_channels=in_channels,
                            out_channels=filters,
                            kernel_size=2, stride=1, padding=1, 
                            dilation=2,bias = True)
        # self.conv21 = Conv2d(in_channels=in_channels,
        #                      out_channels=filters,
        #                      kernel_size=2, stride=1, padding=0,bias = True)
        self.bn2   = BatchNorm2d(filters)

        self.conv3 = Conv2d(in_channels=in_channels,
                            out_channels=filters, 
                            kernel_size=3,stride=1, padding=1,
                            bias=True)
        self.bn3 = BatchNorm2d(filters)

        self.attention = CBAM(gate_channels=filters) 
        
    def forward(self,input_x):

        # Branch 1 
        x1 = self.conv1(input_x)
        x1 = self.bn1(x1)
        x1 = self.leaky_relu(x1)

        # Branch 2 
        x2 = self.conv2(input_x)
        # x2 = self.conv21(x2)
        x2 = self.bn2(x2)
        x2 = self.leaky_relu(x2)

        # Fusion
        # print(x1.shape,x2.shape) 
        x = x1 + x2 

        # Attention conv + BN + activation 
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)

        # Attention block 
        # print(x.shape)
        x = self.attention(x)

        return x + input_x 

class MultiResolutionSegmentationHead(nn.Module):
    def __init__(self,num_classes=2,in_channels=640,filters = 64):
        super(MultiResolutionSegmentationHead,self).__init__() 
        # Convolutions 
        self.conv1 = Conv2d(in_channels=in_channels,out_channels=filters,
                            kernel_size=3,stride=2,padding=1) #64x128x128
        self.conv_transpose = ConvTranspose2d(in_channels=in_channels,
                                              out_channels=filters,
                                              kernel_size=3,stride=2,
                                              padding=1,output_padding=1) #64x256x256

        # Output probabilities 
        self.out1 = Conv2d(in_channels=filters,out_channels=num_classes,kernel_size=3,padding=1)
        self.out2 = Conv2d(in_channels=in_channels,out_channels=num_classes,kernel_size=1)
        self.out3 = Conv2d(in_channels=filters,out_channels=num_classes,kernel_size=3,padding=1)
    def forward(self,x):
        x128 = x # 640x128x128
        x64 = F.relu(self.conv1(x))  # 64x64x64
        # print('x64',x64.shape) 
        x256 = F.relu(self.conv_transpose(x)) # 64x256x256 
        # print('x256',x256.shape)
        out1 = F.softmax(self.out1(x256),dim=1) #Bx2x256x256
        # print('out1',out1.shape)
        
        out2 = F.softmax(self.out2(x128),dim=1) #Bx2x128x128
        # print('out2',out2.shape)

        out3 = F.softmax(self.out3(x64),dim=1) #Bx2x64x64 
        # print('out3',out3.shape)

        return out1,out2,out3
class RauNet(nn.Module):
    def __init__(self,input_channels=10,num_classes=2,dropout_ratio=0.25):
        super(RauNet,self).__init__()
        filters = 128 
        self.inital_conv = Conv2d(in_channels=input_channels,
                                  out_channels=filters,
                                  kernel_size=3,padding=1)
        self.initial_bn  = BatchNorm2d(filters)
        self.initial_relu = nn.LeakyReLU(inplace=True)

        self.res_block1 = ResidualBlock(in_channels=filters,filters=filters) 
        self.downsample1= Conv2d(in_channels=filters,
                                 out_channels=filters,
                                 kernel_size=2,stride=2,padding=0)

        self.res_block2 = ResidualBlock(in_channels=filters,
                                        filters= filters)
        self.downsample2= Conv2d(in_channels=filters,
                                 out_channels=filters,
                                 kernel_size=2,
                                 stride=2,
                                 padding=0)

        self.res_block3= ResidualBlock(in_channels=filters,
                                       filters=filters)
        self.downsample3=Conv2d(in_channels=filters,
                                out_channels=filters,
                                kernel_size=2,
                                stride=2,
                                padding=0)
        self.res_block4 = ResidualBlock(in_channels=filters,
                                        filters=filters)
        self.downsample4=Conv2d(in_channels=filters,
                                out_channels=filters,
                                kernel_size=2,
                                stride=2,padding=0)

        self.bottleneck = ResidualBlock(in_channels=filters,
                                        filters=filters)

        self.upsample4 = ConvTranspose2d(in_channels=filters,
                                         out_channels=filters,
                                         kernel_size=2,
                                         stride=2)
        self.res_block5= ResidualBlock(in_channels=filters*2,
                                       filters=filters*2)

        self.upsample3 = ConvTranspose2d(in_channels=filters*2,
                                         out_channels=filters*2,
                                         kernel_size=2,
                                         stride=2)
        self.res_block6= ResidualBlock(in_channels=filters*3,filters=filters*3)

        self.upsample2 = ConvTranspose2d(in_channels=filters*3,
                                         out_channels=filters*3,
                                         kernel_size=2,
                                         stride=2)
        self.res_block7= ResidualBlock(in_channels=filters*4,filters=filters*4) 

        self.upsample1 = ConvTranspose2d(in_channels=filters*4,
                                         out_channels=filters*4,
                                         kernel_size=2,stride=2)
        self.res_block8= ResidualBlock(in_channels=filters*5,
                                       filters=filters*5) 

        self.dropout = Dropout2d(p=dropout_ratio)
        self.segmentation_head = MultiResolutionSegmentationHead(num_classes=num_classes,
                                                                 in_channels=filters*5,
                                                                 filters = 64)
    def forward(self,x):
        c0 = self.inital_conv(x)
        c0 = self.initial_bn(c0)
        c0 = self.initial_relu(c0)
        # print('c0',c0.shape)
        c1 = self.res_block1(c0)
        p1 = F.selu(self.downsample1(c1))
        # print('c1',c1.shape)
        c2 = self.res_block2(p1)
        p2 = F.selu(self.downsample2(c2))
        # print('c2',c2.shape)
        c3 = self.res_block3(p2)
        p3 = F.selu(self.downsample3(c3))

        c4 = self.res_block4(p3)
        p4 = F.selu(self.downsample4(c4))
    
        b = self.bottleneck(p4)

        u4 = F.selu(self.upsample4(p4))
        u4 = torch.cat((u4,c4),dim=1)
        # print(u4.shape)
        c5 = self.res_block5(u4)
        # print('c5',c5.shape)
        u3 = F.selu(self.upsample3(c5))
        u3 = torch.cat((u3,c3),dim=1)
        c6 = self.res_block6(u3)
        # print('c6',c6.shape)
        u2 = F.selu(self.upsample2(c6))
        u2 = torch.cat((u2,c2),dim=1)
        c7 = self.res_block7(u2)
        # print('c7',c7.shape)
        u1 = F.selu(self.upsample1(c7))
        u1 = torch.cat((u1,c1),dim=1)
        # print('u1',u1.shape)
        x = self.res_block8(u1)
        x = self.dropout(x)
        # print('x',x.shape)
        out1,out2,out3 = self.segmentation_head(x)
        return out1,out2,out3 

class ClassifierHead(nn.Module):
    def __init__(self,in_channel=640):
        super(ClassifierHead,self).__init__() 
        self.conv1 = nn.Conv2d(in_channels=in_channel,out_channels=1,kernel_size=1)
        self.bn = nn.BatchNorm2d(num_features=1)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(in_features=128**2,out_features=512)
        self.bn1 = nn.BatchNorm1d(num_features=512)
        self.fc2 = nn.Linear(in_features=512,out_features=64)
        self.bn2 = nn.BatchNorm1d(num_features=64)
        self.fc3 = nn.Linear(in_features=64,out_features=1)
        self.soft = nn.Sigmoid()
    def forward(self,x):
        # print(x.shape)
        x = F.leaky_relu(self.bn(self.conv1(x)))
        # print(x.shape)
        x = self.flat(x)
        # print(x.shape)
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.soft(self.fc3(x))
        return x

class DenseConnect(nn.Module):
    """
    Add x1,x2,x3,x4 from each stage to make dense layer 

    """
    def __init__(self,size1,size2,size3,size4,out_channels):
        super(DenseConnect,self).__init__() 
        if size1>size4:
            self.max1=nn.MaxPool2d(kernel_size=size1[1]//size4[1])
        else:
            self.max1=nn.UpsamplingNearest2d(scale_factor=size4[1]//size1[1])
        self.conv1=nn.Conv2d(in_channels=size1[0],out_channels=size4[0],kernel_size=3,padding=1)
        self.bn1=nn.BatchNorm2d(num_features=size4[0])

        if size2>size4:
            self.max2=nn.MaxPool2d(kernel_size=size2[1]//size4[1])
        else:
            self.max2=nn.UpsamplingNearest2d(scale_factor=size4[1]//size2[1])
        self.conv2=nn.Conv2d(in_channels=size2[0],out_channels=size4[0],kernel_size=3,padding=1)
        self.bn2=nn.BatchNorm2d(num_features=size4[0])
        
        if size3>size4:
            self.max3=nn.MaxPool2d(kernel_size=size3[1]//size4[1])
        else:
            self.max3=nn.UpsamplingNearest2d(scale_factor=size4[1]//size3[1])
        self.conv3=nn.Conv2d(in_channels=size3[0],out_channels=size4[0],kernel_size=3,padding=1)
        self.bn3=nn.BatchNorm2d(num_features=size4[0])
        
        self.conv=nn.Conv2d(in_channels=size4[0],out_channels=out_channels,kernel_size=3,padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
    def forward(self,x1,x2,x3,x4):
        x1 = self.max1(x1)
        x1 = F.leaky_relu(self.bn1(self.conv1(x1)))
        x2 = self.max2(x2)
        x2 = F.leaky_relu(self.bn2(self.conv2(x2)))
        x3 = self.max3(x3)
        x3 = F.leaky_relu(self.bn3(self.conv3(x3)))
        x = x1+x2+x3+x4 
        x=self.bn(self.conv(x))
        return x

class RauNet12(nn.Module):
    def __init__(self,input_channels=3,num_classes=2,dropout_ratio=0.25):
        super(RauNet12,self).__init__()
        filters = 128 
        self.inital_conv = Conv2d(in_channels=input_channels,
                                  out_channels=filters,
                                  kernel_size=3,padding=1)
        self.initial_bn  = BatchNorm2d(filters)
        self.initial_relu = nn.LeakyReLU(inplace=True)

        self.res_block01 = ResidualBlock(in_channels=filters,filters=filters) 
        self.res_block001 = ResidualBlock(in_channels=filters,filters=filters) 
        self.res_block1 = ResidualBlock(in_channels=filters,filters=filters) 
        self.downsample1= Conv2d(in_channels=filters,
                                 out_channels=filters,
                                 kernel_size=2,stride=2,padding=0)

        self.res_block02 = ResidualBlock(in_channels=filters,filters= filters)
        self.res_block002 = ResidualBlock(in_channels=filters,filters= filters)
        self.res_block2 = ResidualBlock(in_channels=filters,filters= filters)
        self.downsample2= Conv2d(in_channels=filters,
                                 out_channels=filters,
                                 kernel_size=2,
                                 stride=2,
                                 padding=0)

        self.res_block03= ResidualBlock(in_channels=filters,filters=filters)
        self.res_block003= ResidualBlock(in_channels=filters,filters=filters)
        self.res_block3= ResidualBlock(in_channels=filters,filters=filters)
        self.downsample3=Conv2d(in_channels=filters,
                                out_channels=filters,
                                kernel_size=2,
                                stride=2,
                                padding=0)
        self.res_block04 = ResidualBlock(in_channels=filters,filters=filters)
        self.res_block004 = ResidualBlock(in_channels=filters,filters=filters)
        self.res_block4 = ResidualBlock(in_channels=filters,filters=filters)
        # self.dense_connect_down = DenseConnect(size1=(filters,128),size2=(filters,64),size3=(filters,32),size4=(filters,16),out_channels=filters)
        self.downsample4=Conv2d(in_channels=filters,
                                out_channels=filters,
                                kernel_size=2,
                                stride=2,padding=0)

        self.bottleneck = ResidualBlock(in_channels=filters,
                                        filters=filters)

        self.upsample4 = ConvTranspose2d(in_channels=filters,
                                         out_channels=filters,
                                         kernel_size=2,
                                         stride=2)
        self.res_block5= ResidualBlock(in_channels=filters*2,
                                       filters=filters*2)

        self.upsample3 = ConvTranspose2d(in_channels=filters*2,
                                         out_channels=filters*2,
                                         kernel_size=2,
                                         stride=2)
        self.res_block6= ResidualBlock(in_channels=filters*3,filters=filters*3)

        self.upsample2 = ConvTranspose2d(in_channels=filters*3,
                                         out_channels=filters*3,
                                         kernel_size=2,
                                         stride=2)
        self.res_block7= ResidualBlock(in_channels=filters*4,filters=filters*4) 

        self.upsample1 = ConvTranspose2d(in_channels=filters*4,
                                         out_channels=filters*4,
                                         kernel_size=2,stride=2)
        self.res_block8= ResidualBlock(in_channels=filters*5,
                                       filters=filters*5) 
        # self.dense_connect_up = DenseConnect(size1=(filters*2,16),size2=(filters*3,32),size3=(filters*4,64),size4=(filters*5,128),
        #                                      out_channels=filters*5)
        self.dropout = Dropout2d(p=dropout_ratio)
        self.segmentation_head = MultiResolutionSegmentationHead(num_classes=num_classes,
                                                                 in_channels=filters*5,
                                                                 filters = 64)
        self.classifier_head  = ClassifierHead()
    def forward(self,x):
        c0 = self.inital_conv(x)
        c0 = self.initial_bn(c0)
        c0 = self.initial_relu(c0)
        # print('c0',c0.shape)
        c1 = self.res_block01(c0)
        c1 = self.res_block001(c1)
        c1 = self.res_block1(c1) #128
        p1 = F.selu(self.downsample1(c1))
        # print('c1',c1.shape)
        c2 = self.res_block02(p1)
        c2 = self.res_block002(c2)
        c2 = self.res_block2(c2) #64
        p2 = F.selu(self.downsample2(c2))
        # print('c2',c2.shape)
        c3 = self.res_block03(p2)
        c3 = self.res_block003(c3)
        c3 = self.res_block3(c3) #32
        p3 = F.selu(self.downsample3(c3))

        c4 = self.res_block04(p3)
        c4 = self.res_block004(c4)
        c4 = self.res_block4(c4) #16
        # c4 = self.dense_connect_down(c1,c2,c3,c4)
        p4 = F.selu(self.downsample4(c4))
        b = self.bottleneck(p4)#8

        u4 = F.selu(self.upsample4(p4))#16
        u4 = torch.cat((u4,c4),dim=1)
        # print(u4.shape)
        c5 = self.res_block5(u4)
        # print('c5',c5.shape)
        u3 = F.selu(self.upsample3(c5))#32
        u3 = torch.cat((u3,c3),dim=1)
        c6 = self.res_block6(u3)
        # print('c6',c6.shape)
        u2 = F.selu(self.upsample2(c6))#64
        u2 = torch.cat((u2,c2),dim=1)
        c7 = self.res_block7(u2)
        # print('c7',c7.shape)
        u1 = F.selu(self.upsample1(c7))#128
        u1 = torch.cat((u1,c1),dim=1)
        # print('u1',u1.shape)
        x = self.res_block8(u1)
        # x= self.dense_connect_up(c5,c6,c7,x)
        x = self.dropout(x)
        # print('x',x.shape)
        out1,out2,out3 = self.segmentation_head(x)
        out4 = self.classifier_head(x)
        return out1,out2,out3,out4


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = RauNet12(input_channels=23,num_classes=2).to(device)
    # print(model)
    # Print the detailed summary
    summary(model, input_size=(23, 128, 128))
    x = torch.rand(1,23,128,128).to(device)
    out1,out2,out3 = model(x)

if __name__=="__main__":
    main() 
