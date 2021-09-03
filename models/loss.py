import paddle
import paddle.nn as nn
class class_loss_3class(nn.Layer):
    #Class loss
    def __init__(self):
        super(class_loss_3class, self).__init__()

    def forward(self, type_res):
        n = len(type_res)
        m = len(type_res[0]) - 1
        type_all = type_res
        loss = 0
        for i in range(n):
            sum_re = paddle.abs(type_all[i][0]-type_all[i][1]) + paddle.abs(type_all[i][0]-type_all[i][2]) + paddle.abs(type_all[i][1]-type_all[i][2])
            loss += (m - sum_re)
        return loss / n


class average_loss_3class(nn.Layer):
    #Average loss
    def __init__(self):
        super(average_loss_3class, self).__init__()

    def forward(self, type_res):
        n = len(type_res)
        m = len(type_res[0])
        type_all = type_res
        sum1 = 0
        sum2 = 0
        sum3 = 0

        for i in range(n):
            sum1 += type_all[i][0]
            sum2 += type_all[i][1]
            sum3 += type_all[i][2]

        return (paddle.abs(sum1-n/m) + paddle.abs(sum2-n/m) + paddle.abs(sum3-n/m)) / ((n/m)*4)

class CharbonnierLoss(nn.Layer):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = paddle.sum(paddle.sqrt(diff * diff + self.eps))
        return loss


# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Layer):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return paddle.empty_like(input).fill_(self.real_label_val)
        else:
            return paddle.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss

