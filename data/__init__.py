# """create dataset and dataloader"""
# import logging
# import paddle
# from paddle.io import Dataset, DataLoader



# def create_dataset(dataset_opt):
#     mode = dataset_opt['mode']
#     # datasets for image restoration
#     if mode == 'LQ':
#         from data.LQ_dataset import LQDataset as D
#     elif mode == 'LQGT':
#         from data.LQGT_dataset import LQGTDataset as D
#     elif mode == 'LQGT_rcan':
#         from data.LQGT_rcan_dataset import LQGTDataset_rcan as D
#     elif mode == 'LQ_label':
#         from data.LQ_label_dataset import LQ_label_Dataset as D
#     # datasets for video restoration
#     # elif mode == 'REDS':
#     #     from data.REDS_dataset import REDSDataset as D
#     # elif mode == 'Vimeo90K':
#     #     from data.Vimeo90K_dataset import Vimeo90KDataset as D
#     # elif mode == 'video_test':
#     #     from data.video_test_dataset import VideoTestDataset as D
#     else:
#         raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
#     dataset = D(dataset_opt)

#     logger = logging.getLogger('base')
#     logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
#                                                            dataset_opt['name']))
#     return dataset
