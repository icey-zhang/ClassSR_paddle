import os
###显卡设置
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import math
import argparse
import random
import logging

import options.options as option
from utils import util
# from data import create_dataloader
from models import create_model
import numpy as np
from data.LQGT_rcan_dataset import LQGTDataset_rcan,create_dataloader
import paddle
# paddle.device.set_device("gpu:0")

def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str,default='options/train/train_RCAN3.yml', help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']


    # #### loading resume state if exists
    # if opt['path'].get('resume_state', None):
    #     # distributed resuming: all load into default GPU
    #     # device_id = torch.cuda.current_device()
    #     resume_state = paddle.load(opt['path']['resume_state'])#,
    #                               #map_location=lambda storage, loc: storage.cuda(device_id))
    #     option.check_resume(opt, resume_state['iter'])  # check resume options
    # else:
    #     resume_state = None
    try:
        resume_state = paddle.load(opt['path']['resume_state'])
    except:
        resume_state = None
        print("no use of resume")

    #### mkdir and loggers
    util.mkdir_and_rename(opt['path']['experiments_root'])  # rename experiment folder if exists
    util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                        and 'pretrain_model' not in key and 'resume' not in key))

    # config loggers. Before it, the log will not work
    util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                        screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    #### create train and val dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = LQGTDataset_rcan(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            train_loader = create_dataloader(train_set, dataset_opt, opt)
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                len(train_set), train_size))
            logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                total_epochs, total_iters))
        elif phase == 'val':
            val_set = LQGTDataset_rcan(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt)
            logger.info('Number of val images in [{:s}]: {:d}'.format(
                dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    #### create model
    model = create_model(opt)

    ### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        for i, train_data in enumerate(train_loader):
        # for train_data in train_loader:
            current_step += 1
            if current_step > total_iters:
                break
            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])
            #### training
            #print(train_data)
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '[epoch:{:3d}, iter:{:8,d}, lr:('.format(epoch, current_step)
                for v in model.get_current_learning_rate():
                    message += '{:.3e},'.format(v)
                message += ')] '
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                logger.info(message)
            ### validation
            if opt['datasets'].get('val', None) and current_step % opt['train']['val_freq'] == 0:
                # does not support multi-GPU validation
                pbar = util.ProgressBar(len(val_loader))
                avg_psnr = 0.
                idx = 0
                for val_data in val_loader:
                    idx += 1
                    img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
                    img_dir = os.path.join(opt['path']['val_images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(val_data)
                    model.test()

                    visuals = model.get_current_visuals()
                    if which_model =="RCAN":
                        sr_img = util.tensor2img(visuals['rlt'], np.uint8, min_max=(0, 255))  # uint8
                        gt_img = util.tensor2img(visuals['GT'], np.uint8, min_max=(0, 255))  # uint8
                    else:
                        sr_img = util.tensor2img(visuals['rlt'])  # uint8
                        gt_img = util.tensor2img(visuals['GT'])  # uint8

                    # Save SR images for reference
                    save_img_path = os.path.join(img_dir,
                                                    '{:s}_{:d}.png'.format(img_name, current_step))
                    util.save_img(sr_img, save_img_path)

                    # calculate PSNR
                    sr_img, gt_img = util.crop_border([sr_img, gt_img], opt['scale'])
                    avg_psnr += util.calculate_psnr(sr_img, gt_img)
                    pbar.update('Test {}'.format(img_name))

                avg_psnr = avg_psnr / idx

                # log
                logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))


            #### save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(current_step)
                model.save_training_state(epoch, current_step)

    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')


if __name__ == '__main__':
    main()
