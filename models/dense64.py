import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import v2
import os

class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, base, extras, head, dense_list, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        # TODO: implement __call__ in PriorBox
        self.priorbox = PriorBox(v2)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = 300

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm1 = L2Norm(256, 20)
        self.L2Norm = L2Norm(512, 20)
        self.L2Norm3 = L2Norm(1024, 10)
        self.L2Norm4 = L2Norm(512, 10)
        self.L2Norm5 = L2Norm(256, 10)
        self.extras = nn.ModuleList(extras)
        self.dense_list0 = nn.ModuleList(dense_list[0])
        self.dense_list1 = nn.ModuleList(dense_list[1])
        self.dense_list2 = nn.ModuleList(dense_list[2])
        self.dense_list3 = nn.ModuleList(dense_list[3])
        self.dense_list4 = nn.ModuleList(dense_list[4])
        self.dense_list5 = nn.ModuleList(dense_list[5])
        # for i in range(len(dense_list)):
        #     self.dense_list.append(nn.ModuleList(dense_list[i]))

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax()
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.
        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].
        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]
            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        
        for k in range(16):
            x = self.vgg[k](x)

        # 80x80
        dense1 = x
        dense1 = self.L2Norm1(dense1)
        dense1_p1 = self.dense_list0[0](dense1)
        dense1_p2 = self.dense_list0[1](dense1_p1)
        dense1_p3 = self.dense_list0[2](dense1_p2)
        dense1_p1_conv = self.dense_list0[3](dense1_p1)
        dense1_p2_conv = self.dense_list0[4](dense1_p2)
        dense1_p3_conv = self.dense_list0[5](dense1_p3)
        # p = self.add_conv[1](p)

        for k in range(16, 23):
            x = self.vgg[k](x)
        #40x40
        dense2 = x
        dense2 = self.L2Norm(dense2)
        dense2_p1 = self.dense_list1[0](dense2)
        dense2_p2 = self.dense_list1[1](dense2_p1)
        dense2_p3 = self.dense_list1[2](dense2_p2)
        dense2_p1_conv = self.dense_list1[3](dense2_p1)
        dense2_p2_conv = self.dense_list1[4](dense2_p2)
        dense2_p3_conv = self.dense_list1[5](dense2_p3)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)

        #20x20
        dense3 = x
        dense3 = self.L2Norm3(dense3)
        dense3_up_conv = self.dense_list2[0](dense3)
        dense3_up = self.dense_list2[1](dense3_up_conv)
        dense3_p1 = self.dense_list2[2](dense3)
        dense3_p2 = self.dense_list2[3](dense3_p1)
        dense3_p1_conv = self.dense_list2[4](dense3_p1)
        dense3_p2_conv = self.dense_list2[5](dense3_p2)

        x = F.relu(self.extras[0](x), inplace=True)
        x = F.relu(self.extras[1](x), inplace=True)

        #10x10
        dense4 = x
        dense4 = self.L2Norm4(dense4)
        dense4_up1_conv = self.dense_list3[0](dense4)
        dense4_up2_conv = self.dense_list3[1](dense4)
        dense4_up1 = self.dense_list3[2](dense4_up1_conv)
        dense4_up2 = self.dense_list3[3](dense4_up2_conv)
        dense4_p = self.dense_list3[4](dense4)
        dense4_p_conv = self.dense_list3[5](dense4_p)

        x = F.relu(self.extras[2](x), inplace=True)
        x = F.relu(self.extras[3](x), inplace=True)

        #5x5
        dense5 = x
        dense5 = self.L2Norm5(dense5)
        dense5_up1_conv = self.dense_list4[0](dense5)
        dense5_up2_conv = self.dense_list4[1](dense5)
        dense5_up3_conv = self.dense_list4[2](dense5)
        dense5_up1 = self.dense_list4[3](dense5_up1_conv)
        dense5_up2 = self.dense_list4[4](dense5_up2_conv)
        dense5_up3 = self.dense_list4[5](dense5_up3_conv)

        dense_out1 = torch.cat((dense1_p1_conv, dense2, dense3_up, dense4_up2, dense5_up3), 1)
        dense_out1 = F.relu(self.dense_list5[0](dense_out1))
        sources.append(dense_out1)

        dense_out2 = torch.cat((dense1_p2_conv, dense2_p1_conv, dense3, dense4_up1, dense5_up2), 1)
        dense_out2 = F.relu(self.dense_list5[1](dense_out2))
        sources.append(dense_out2)

        dense_out3 = torch.cat((dense1_p3_conv, dense2_p2_conv, dense3_p1_conv, dense4, dense5_up1), 1)
        dense_out3 = F.relu(self.dense_list5[2](dense_out3))
        sources.append(dense_out3)

        dense_out4 = torch.cat((dense2_p3_conv, dense3_p2_conv, dense4_p_conv, dense5), 1)
        dense_out4 = F.relu(self.dense_list5[3](dense_out4))
        sources.append(dense_out4)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            if k > 3:
                x = F.relu(v(x), inplace=True)
                if k % 2 == 1:
                    sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

    def finetune(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            pre_dict = torch.load(base_file, map_location=lambda storage, loc: storage)
            model_dict = self.state_dict()
            model_dict.update(pre_dict)
            self.load_state_dict(model_dict)
            # self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.') 

# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers

def dense_conv1():
    layers = []
    #75x75
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
    layers.append(nn.Conv2d(256, 64, kernel_size=3, padding=1))
    layers.append(nn.Conv2d(256, 32, kernel_size=3, padding=1))
    layers.append(nn.Conv2d(256, 32, kernel_size=3, padding=1))
    return layers

def dense_conv2():
    layers = []
    #38x38
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
    layers.append(nn.Conv2d(512, 64, kernel_size=3, padding=1))
    layers.append(nn.Conv2d(512, 32, kernel_size=3, padding=1))
    layers.append(nn.Conv2d(512, 16, kernel_size=3, padding=1))    
    return layers

def dense_conv3():
    #19x19
    layers = []
    layers.append(nn.Conv2d(1024, 64, kernel_size=3, padding=1))
    layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
    layers.append(nn.Conv2d(1024, 64, kernel_size=3, padding=1))
    layers.append(nn.Conv2d(1024, 32, kernel_size=3, padding=1))
    return layers

def dense_conv4():
    #10x10
    layers  = []
    layers.append(nn.Conv2d(512, 64, kernel_size=3, padding=1))
    layers.append(nn.Conv2d(512, 32, kernel_size=3, padding=1))
    layers.append(nn.Upsample(size=(19, 19), mode='bilinear'))
    layers.append(nn.Upsample(size=(38, 38), mode='bilinear'))
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
    layers.append(nn.Conv2d(512, 32, kernel_size=3, padding=1))    
    return layers

def dense_conv5():
    #5x5
    layers = []
    layers.append(nn.Conv2d(256, 64, kernel_size=3, padding=1))
    layers.append(nn.Conv2d(256, 32, kernel_size=3, padding=1))
    layers.append(nn.Conv2d(256, 32, kernel_size=3, padding=1))
    layers.append(nn.Upsample(size=(10, 10), mode='bilinear'))
    layers.append(nn.Upsample(size=(19, 19), mode='bilinear'))
    layers.append(nn.Upsample(size=(38, 38), mode='bilinear'))
    return layers

def dense_concat():
    layers = []
    layers.append(nn.Conv2d(704, 512, kernel_size=1, padding=0))
    layers.append(nn.Conv2d(1216, 1024, kernel_size=1, padding=0))
    layers.append(nn.Conv2d(704, 512, kernel_size=1, padding=0))
    layers.append(nn.Conv2d(336, 256, kernel_size=1, padding=0))
    return layers

def multibox(vgg, extra_layers, cfg, num_classes, dense_list):
    loc_layers = []
    conf_layers = []
    vgg_source = [24, -2]
    out_channels = [512, 1024]
    for k, v in enumerate(out_channels):
        loc_layers += [nn.Conv2d(v,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers), dense_list


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '300_extra' : [6, 6, 6, 6, 4, 4],
    '512': [],
}


def build_ssd(phase, size=300, num_classes=21):
    v2['use_extra_prior'] = False
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 300:
        print("Error: Sorry only SSD300 is supported currently!")
        return
    if v2["use_extra_prior"]:
        mbox_size = '300_extra'
        print("use more prior") 
    else:
        mbox_size = '300'

    dense_list = [dense_conv1(), dense_conv2(), dense_conv3(), dense_conv4(), dense_conv5(), dense_concat()]

    return SSD(phase, *multibox(vgg(base[str(size)], 3),
                                add_extras(extras[str(size)], 1024),
                                mbox[mbox_size], num_classes, dense_list), num_classes)