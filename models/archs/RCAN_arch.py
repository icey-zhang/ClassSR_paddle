import math
import paddle
from paddle.fluid.layers.nn import scale
import paddle.nn as nn
import models.archs.arch_util as arch_util
import numpy as np
import paddle.nn.initializer as init
# import arch_util
## Channel Attention (CA) Layer
class CALayer(nn.Layer):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2D(channel, channel // reduction, 1, padding=0, weight_attr =init.KaimingNormal(), bias_attr = init.Constant(value=0.)),
                nn.ReLU(),
                nn.Conv2D(channel // reduction, channel, 1, padding=0, weight_attr =init.KaimingNormal(), bias_attr = init.Constant(value=0.)),
                nn.Sigmoid()
        )
        self.initialize_weights(scale=0.1)
    def initialize_weights(self,scale):
        if not isinstance(self.conv_du, list):
            net_l = [self.conv_du]
        for net in net_l:
            for m in net:
                if isinstance(m, nn.Conv2D):
                    scale_weight = scale * m.weight.numpy()
                    m.weight.set_value(paddle.to_tensor(scale_weight))
                    if m.bias is not None:
                        scale_bias = 0 * m.bias.numpy()
                        m.bias.set_value(paddle.to_tensor(scale_bias))              
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Layer):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias_attr=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            temp_module = conv(n_feat, n_feat, kernel_size, bias_attr=bias_attr)
            # ############## initialize_weights ####################
            # arch_util.initialize_weights(temp_module, 0.1)
            # ############## initialize_weights ####################
            modules_body.append(temp_module)
            if bn: 
                temp_module = nn.BatchNorm2D(n_feat)
                # ############## initialize_weights ####################
                # arch_util.initialize_weights(temp_module, 0.1)
                # ############## initialize_weights ####################
                modules_body.append(temp_module)
            if i == 0: 
                modules_body.append(act)
        ############## initialize_weights ####################
        arch_util.initialize_weights_nonSequential(modules_body, 0.1)
        ############## initialize_weights ####################
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale
    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Layer):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias_attr=None, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        temp_module = conv(n_feat, n_feat, kernel_size)
        ############## initialize_weights ####################
        arch_util.initialize_weights_nonSequential(temp_module, 0.1)
        ############## initialize_weights ####################
        modules_body.append(temp_module)
        self.body = nn.Sequential(*modules_body)  
    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias_attr=None):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2D(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2D(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError
        arch_util.initialize_weights_nonSequential(m, 0.1)
        super(Upsampler, self).__init__(*m)

## Residual Channel Attention Network (RCAN)
class RCAN(nn.Layer):
    ''' modified RCAN '''
    
    def __init__(self, n_resgroups, n_resblocks, n_feats, res_scale, n_colors, rgb_range, scale, reduction, conv=arch_util.default_conv):
        super(RCAN, self).__init__()
        
        n_resgroups = n_resgroups
        n_resblocks = n_resblocks
        n_feats = n_feats
        kernel_size = 3
        reduction = reduction 
        scale = scale
        act = nn.ReLU(True)
        
        # RGB mean for DIV2K
        rgb_mean = np.array([0.4488, 0.4371, 0.4040])
        rgb_std = np.array([1.0, 1.0, 1.0])
        sign=-1
        ################ initialize #########################
        # self.sub_mean = arch_util.MeanShift(rgb_range, rgb_mean, rgb_std)
        # self.sub_mean = nn.Conv2D(3, 3, kernel_size=1)
        weight_attr1 = paddle.framework.ParamAttr(
            name='sub_mean_w_init1'+self._full_name,
            initializer=paddle.nn.initializer.Assign(np.eye(3).reshape([3, 3, 1, 1])/rgb_std.reshape(3, 1, 1, 1))
        )
        bias_attr1 = paddle.framework.ParamAttr(
            name='sub_mean_b_init1'+self._full_name,
            initializer=paddle.nn.initializer.Assign(sign * rgb_range * rgb_mean/rgb_std)
        )
        self.sub_mean = nn.Conv2D(3, 3, kernel_size=1, weight_attr=weight_attr1, bias_attr=bias_attr1)
        for param in self.sub_mean.parameters():
            param.trainable = False
        ################ initialize #########################

        # define head module
        modules_head = [conv(n_colors, n_feats, kernel_size, weight_attr =init.KaimingNormal(), bias_attr = init.Constant(value=0.))]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]
        temp_module = conv(n_feats, n_feats, kernel_size)
        ############## initialize_weights ####################
        arch_util.initialize_weights_nonSequential(temp_module, 0.1)
        ############## initialize_weights ####################
        modules_body.append(temp_module)

        # define tail module
        modules_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)]
        ############## initialize_weights for modules_tail ####################
        # arch_util.initialize_weights_nonSequential(modules_tail, 0.1)
        ############## initialize_weights for modules_tail ####################

        ################ initialize for add_mean #########################
        sign = 1
        weight_attr2 = paddle.framework.ParamAttr(
            name='sub_mean_w_init2'+self._full_name,
            initializer=paddle.nn.initializer.Assign(np.eye(3).reshape([3, 3, 1, 1])/rgb_std.reshape(3, 1, 1, 1))
        )
        bias_attr2 = paddle.framework.ParamAttr(
            name='sub_mean_b_init2'+self._full_name,
            initializer=paddle.nn.initializer.Assign(sign * rgb_range * rgb_mean/rgb_std)
        )
        # self.add_mean = arch_util.MeanShift(rgb_range, rgb_mean, rgb_std, 1)
        self.add_mean = nn.Conv2D(3, 3, kernel_size=1, weight_attr=weight_attr2, bias_attr=bias_attr2)
        for param in self.add_mean.parameters():
            param.trainable = False
        ################ initialize for add_mean #########################

        self.head = nn.Sequential(*modules_head)
        arch_util.initialize_weights([self.head], 0.1)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
        arch_util.initialize_weights([self.tail], 0.1)
        # arch_util.initialize_weights([self.head, self.body, self.tail], 0.1)
    def forward(self, x):
        x = self.sub_mean(x)
        # print(x.max())
        # print(x.min())

        x = self.head(x)
        # print(x.max())
        # print(x.min())
        res = self.body(x)
        # print(res.max())
        # print(res.min())
        res += x
        # print(res.max())
        # print(res.min())
        x = self.tail(res)
        # print(x.max())
        # print(x.min())
        x = self.add_mean(x)
        # print(x.max())
        # print(x.min())
        return x 
