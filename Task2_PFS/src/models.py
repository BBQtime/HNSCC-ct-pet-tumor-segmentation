import torch
from torch import nn
from torch.nn import functional as F

from layers import BasicConv3d, FastSmoothSeNormConv3d, RESseNormConv3d, UpConv

class Encoder(nn.Module):
    def __init__(self, in_channels, n_cls, n_filters):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.n_cls = 1 if n_cls == 2 else n_cls
        self.n_filters = n_filters

        self.block_1_1_left = BasicConv3d(in_channels, n_filters, kernel_size=3, stride=1, padding=1)
        self.block_1_2_left = BasicConv3d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2)  # 64, 1/2
        self.block_2_1_left = BasicConv3d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_left = BasicConv3d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2)  # 128, 1/4
        self.block_3_1_left = BasicConv3d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_left = BasicConv3d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2)  # 256, 1/8
        self.block_4_1_left = BasicConv3d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_4_2_left = BasicConv3d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 *  n_filters*729, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)
    def forward(self, x):

        ds0 = self.block_1_2_left(self.block_1_1_left(x))
        ds1 = self.block_2_2_left(self.block_2_1_left(self.pool_1(ds0)))
        ds2 = self.block_3_2_left(self.block_3_1_left(self.pool_2(ds1)))
        x = self.block_4_2_left(self.block_4_1_left(self.pool_3(ds2)))
        x = self.flatten(x)
        x = self.fc3(self.fc2(self.fc1(x)))

        return F.relu(x)



class FastSmoothSENormDeepEncoder_supervision_skip_no_drop(nn.Module):
    """The model presented in the paper. This model is one of the multiple models that we tried in our experiments
    that it why it has such an awkward name."""

    def __init__(self, in_channels, n_cls, n_filters, reduction=2, return_logits=False):
        super(FastSmoothSENormDeepEncoder_supervision_skip_no_drop, self).__init__()
        self.in_channels = in_channels
        self.n_cls = 1 if n_cls == 2 else n_cls
        self.n_filters = n_filters
        self.return_logits = return_logits

        self.block_1_1_left = RESseNormConv3d(in_channels, n_filters, reduction, kernel_size=7, stride=1, padding=3)
        self.block_1_2_left = RESseNormConv3d(n_filters, n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block_2_1_left = RESseNormConv3d(n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_2_2_left = RESseNormConv3d(2 * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_2_3_left = RESseNormConv3d(2 * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block_3_1_left = RESseNormConv3d(2 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_3_2_left = RESseNormConv3d(4 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_3_3_left = RESseNormConv3d(4 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block_4_1_left = RESseNormConv3d(4 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_4_2_left = RESseNormConv3d(8 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_4_3_left = RESseNormConv3d(8 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.pool_4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block_5_1_left = RESseNormConv3d(8 * n_filters, 16 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_5_2_left = RESseNormConv3d(16 * n_filters, 16 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_5_3_left = RESseNormConv3d(16 * n_filters, 16 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * n_filters*729, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)
  
    def forward(self, x):

        ds0 = self.block_1_2_left(self.block_1_1_left(x))
        ds1 = self.block_2_3_left(self.block_2_2_left(self.block_2_1_left(self.pool_1(ds0))))
        ds2 = self.block_3_3_left(self.block_3_2_left(self.block_3_1_left(self.pool_2(ds1))))
        ds3 = self.block_4_3_left(self.block_4_2_left(self.block_4_1_left(self.pool_3(ds2))))
        x = self.block_5_3_left(self.block_5_2_left(self.block_5_1_left(self.pool_4(ds3))))
        x = self.flatten(x)
        x = self.fc3(self.fc2(self.fc1(x)))
        return F.relu(x)
