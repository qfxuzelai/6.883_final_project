import torch
import torch.nn as nn

# (conv => BN => ReLU) * 2
class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x

# input conv
class in_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(in_conv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)
    
    def forward(self, x):
        x = self.conv(x)
        return x

# down conv
class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )
    
    def forward(self, x):
        x = self.down_conv(x)
        return x

# up conv
class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.double_conv = double_conv(in_ch, out_ch)
    
    def forward(self, x, prex):
        x = self.up_conv(x)
        x = torch.cat([x, prex], dim=1)
        x = self.double_conv(x)
        return x

# output conv
class out_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(out_conv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
    
    def forward(self, x):
        x = self.conv(x)
        return x

'''
U-Net:
    __init__()
        num_ch: num of initial feature map channel (16)
        num_fm: num of output feature map channel (16)
    forward()
        input:  N x 1 x 256 x 256
        output: N x num_fm x 256 x 256
'''
class u_net(nn.Module):
    def __init__(self, num_ch, num_fm):
        super(u_net, self).__init__()
        self.inconv = in_conv(1, num_ch)
        self.down1 = down(num_ch, num_ch * 2)
        self.down2 = down(num_ch * 2, num_ch * 4)
        self.down3 = down(num_ch * 4, num_ch * 8)
        self.down4 = down(num_ch * 8, num_ch * 16)
        self.down5 = down(num_ch * 16, num_ch * 32)
        self.down6 = down(num_ch * 32, num_ch * 64)
        self.up6 = up(num_ch * 64, num_ch * 32)
        self.up5 = up(num_ch * 32, num_ch * 16)
        self.up4 = up(num_ch * 16, num_ch * 8)
        self.up3 = up(num_ch * 8, num_ch * 4)
        self.up2 = up(num_ch * 4, num_ch * 2)
        self.up1 = up(num_ch * 2, num_ch)
        self.outconv = out_conv(num_ch, num_fm)
    
    def forward(self, x):
        conx1 = self.inconv(x)
        conx2 = self.down1(conx1)
        conx3 = self.down2(conx2)
        conx4 = self.down3(conx3)
        conx5 = self.down4(conx4)
        conx6 = self.down5(conx5)
        conx7 = self.down6(conx6)
        x = self.up6(conx7, conx6)
        x = self.up5(x, conx5)
        x = self.up4(x, conx4)
        x = self.up3(x, conx3)
        x = self.up2(x, conx2)
        x = self.up1(x, conx1)
        x = self.outconv(x)
        return torch.sigmoid(x)
