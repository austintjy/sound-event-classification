'''https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/pytorch/models.py'''

import imp
import torch
from torch import nn
import torch.nn.functional as F
from dynamic_convolutions import Dynamic_conv2d
from config import use_fdy

__author__ = "Andrew Koh Jin Jie, Anushka Jain and Soham Tiwari"
__credits__ = ["Prof Chng Eng Siong", "Yan Zhen", "Tanmay Khandelwal"]
__license__ = "GPL"
__version__ = "0.0.0"
__maintainer__ = "Soham Tiwari"
__email__ = "soham.tiwari800@gmail.com"
__status__ = "Development"

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_fdy=use_fdy):

        super(ConvBlock, self).__init__()

        # if use_fdy:
        #     self.conv2d = Dynamic_conv2d
        # else:
        self.conv2d = nn.Conv2d

        self.conv1 = self.conv2d(in_channels,
                               out_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.conv2 = self.conv2d(out_channels,
                               out_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        # init_layer(self.conv1)
        # init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


class Cnn10(nn.Module):
    def __init__(self):

        super(Cnn10, self).__init__()

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)

    def forward(self, input):
        # 1. Try pooling/linear layer
        # 2. Or change bn0 to 128, but ideally avoid this step right
        # 3. Remove all intermediate layers between input and pann 
        
        x = input.unsqueeze(1)   # -> (batch_size, 1, time_steps, mel_bins)
        x = x.transpose(1, 3)   # -> (batch_size, mel_bins, time_steps, 1)
        x = self.bn0(x)         # -> (batch_size, mel_bins, time_steps, 1)
        x = x.transpose(1, 3)   # -> (batch_size, 1, time_steps, mel_bins)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)     #(batch_size, 512, T/16, mel_bins/16)


        return x

class Cnn14(nn.Module):
    def __init__(self):

        super(Cnn14, self).__init__()

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)

    def forward(self, input, mixup_lambda=None):


        x = input.unsqueeze(1)   # (batch_size, 1, time_steps, mel_bins)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        return x


class Tag(nn.Module):
    def __init__(self,class_num):
        super(Tag, self).__init__()
        self.feature = Cnn10()
        self.fc1 = nn.Linear(512,512,bias=True)
        self.fc = nn.Linear(512,class_num,bias=True)
        self.init_weights()

    def init_weights(self):
        init_layer(self.fc1)
        init_layer(self.fc)

    def forward(self,input):
        '''
        :param input: (batch_size,time_steps, mel_bins)
        :return: ()
        '''
        x = self.feature(input)     #(batch_size, 512, T/16, mel_bins/16)
        x = torch.mean(x,dim=3)     #(batch_size, 512, T/16)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu_(self.fc1(x))
        #(batch_size,class_num)
        output = torch.sigmoid(self.fc(x))
        # output = self.fc(x)

        return output
