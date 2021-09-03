import os
import math
import argparse
import random
import logging

import options.options as option
from utils import util
from data.LQGT_rcan_dataset import create_dataloader
from data.LQGT_rcan_dataset import LQGTDataset_rcan
# from data import create_dataloader, create_dataset
from models import create_model


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt',default='options/train/train_ClassSR_RCAN.yml', type=str, help='Path to option YAML file.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    
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
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = LQGTDataset_rcan(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            train_sampler = None
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

    #### resume training
    # if resume_state:
    #     logger.info('Resuming training from epoch: {}, iter: {}.'.format(
    #         resume_state['epoch'], resume_state['iter']))

    #     start_epoch = resume_state['epoch']
    #     current_step = resume_state['iter']
    #     model.resume_training(resume_state)  # handle optimizers and schedulers
    # else:
        # current_step = 0
        # start_epoch = 0
    current_step = 0
    start_epoch = 0

    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        for train_data in train_loader:
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
                if opt['model'] in ['vsr']:    # video restoration validation
                    pbar = util.ProgressBar(len(val_loader))
                    psnr_rlt = {}  # with border and center frames
                    psnr_rlt_avg = {}
                    psnr_total_avg = 0.
                    for val_data in val_loader:
                        folder = val_data['folder'][0]
                        idx_d = val_data['idx'].item()
                        if psnr_rlt.get(folder, None) is None:
                            psnr_rlt[folder] = []

                        model.feed_data(val_data)
                        model.test()
                        visuals = model.get_current_visuals()
                        rlt_img = util.tensor2img(visuals['rlt'])  # uint8
                        gt_img = util.tensor2img(visuals['GT'])  # uint8

                        # calculate PSNR
                        psnr = util.calculate_psnr(rlt_img, gt_img)
                        psnr_rlt[folder].append(psnr)
                        pbar.update('Test {} - {}'.format(folder, idx_d))
                    for k, v in psnr_rlt.items():
                        psnr_rlt_avg[k] = sum(v) / len(v)
                        psnr_total_avg += psnr_rlt_avg[k]
                    psnr_total_avg /= len(psnr_rlt)
                    log_s = '# Validation # PSNR: {:.4e}:'.format(psnr_total_avg)
                    for k, v in psnr_rlt_avg.items():
                        log_s += ' {}: {:.4e}'.format(k, v)
                    logger.info(log_s)

                else:
                    # does not support multi-GPU validation
                    pbar = util.ProgressBar(len(val_loader))
                    avg_psnr = 0.
                    idx = 0
                    num_ress = [0, 0,0]
                    psnr_ress=[0, 0,0]
                    for val_data in val_loader:
                        idx += 1
                        img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
                        img_dir = os.path.join(opt['path']['val_images'], img_name)
                        util.mkdir(img_dir)

                        model.feed_data(val_data)
                        model.test()

                        visuals = model.get_current_visuals()
                        sr_img = visuals['rlt'] # uint8
                        gt_img = visuals['GT'] # uint8
                        num_res = visuals['num_res']
                        psnr_res=visuals['psnr_res']
                        num_ress[0] += num_res[0]
                        num_ress[1] += num_res[1]
                        num_ress[2] += num_res[2]

                        psnr_ress[0] += psnr_res[0]
                        psnr_ress[1] += psnr_res[1]
                        psnr_ress[2] += psnr_res[2]


                        # Save SR images for reference
                        save_img_path = os.path.join(img_dir,
                                                        '{:s}_{:d}.png'.format(img_name, current_step))
                        util.save_img(sr_img, save_img_path)

                        # calculate PSNR
                        sr_img, gt_img = util.crop_border([sr_img, gt_img], opt['scale'])
                        avg_psnr += util.calculate_psnr(sr_img, gt_img)
                        pbar.update('Test {}'.format(img_name))

                    flops,percent=util.cal_FLOPs(which_model,num_ress)
                    if num_ress[0]==0:
                        num_ress[0]=1
                    if num_ress[1]==0:
                        num_ress[1]=1
                    if num_ress[2]==0:
                        num_ress[2]=1

                    # log
                    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                    logger.info('# Validation # FLOPs: {:.4e}'.format(flops))
                    logger.info('# Validation # Percent: {:.4e}'.format(percent))
                    logger.info('# Validation # TYPE num: {0} {1} {2} '.format(num_ress[0], num_ress[1],num_ress[2]))
                    logger.info('# Validation # PSNR Class: {0} {1} {2}'.format(psnr_ress[0]/num_ress[0],psnr_ress[1]/num_ress[1],psnr_ress[2]/num_ress[2]))


            #### save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                # if rank <= 0:
                logger.info('Saving models and training states.')
                model.save(current_step)
                model.save_training_state(epoch, current_step)

    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')


if __name__ == '__main__':
    main()
