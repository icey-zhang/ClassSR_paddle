import logging
from collections import OrderedDict
import paddle
import paddle.nn as nn
import models.networks as networks
from .base_model import BaseModel
from models.loss import CharbonnierLoss


logger = logging.getLogger('base')


class SRModel(BaseModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt)

        self.load()

        if self.is_train:
            self.netG.train()

            # loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss()
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss()
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss()
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

      

            # schedulers   
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0      
            if train_opt['lr_scheme'] == 'MultiStepLR':
                self.schedulers=paddle.optimizer.lr.MultiStepDecay(learning_rate=train_opt['lr_G'],weight_decay=wd_G,gamma=train_opt['lr_gamma'])
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                self.schedulers=paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=train_opt['lr_G'], T_max=train_opt['T_period'],eta_min=train_opt['eta_min'])
            self.optimizer_G = paddle.optimizer.Adam(parameters=self.netG.parameters(), learning_rate=self.schedulers, beta1=train_opt['beta1'], beta2=train_opt['beta2'])
            
            self.optimizers.append(self.optimizer_G)

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ']  # LQ
        if need_GT:
            self.real_H = data['GT']  # GT

    def optimize_parameters(self, step):
        self.optimizer_G.clear_grad()
        self.fake_H = self.netG(self.var_L)
        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        l_pix.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self):
        self.netG.eval()
        with paddle.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()


    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].cpu() #.float()
        out_dict['rlt'] = self.fake_H.detach()[0].cpu()#.float()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].cpu()#.float()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n.item()))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)

