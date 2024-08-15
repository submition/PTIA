from os import path as osp
from PIL import Image

from basicsr.utils import scandir

import os
def generate_meta_info_div2k():
    """Generate meta info for DIV2K dataset.
    """

    # gt_folder = 'datasets/DIV2K/DIV2K_train_HR_sub/'
    # meta_info_txt = 'basicsr/data/meta_info/meta_info_DIV2K800sub_GT.txt'

    gt_folder = '/home/jq/Compression/CAVSR-master/datasets/test_18/gt/raw/'
    meta_info_txt = '/home/jq/Compression/CAVSR-master/basicsr/data/meta_info/meta_info_test18_GT.txt'

    # img_list = sorted(list(scandir(gt_folder)))
    video_list = sorted(os.listdir(gt_folder))
    img_list = []

    # for video in video_list:
    #     video_path = os.path.join(gt_folder, video)
    #     file_list = sorted(os.listdir(video_path))
    #     for file in file_list:
    #         file_path = os.path.join(video_path, file)
    #         img_list.append(file_path)
    # print(img_list)
    # print(len(img_list))

    with open(meta_info_txt, 'w') as f:

        for idx, video in enumerate(video_list):
            video_path = os.path.join(gt_folder, video)
            frames = len(os.listdir(video_path))
            img_path = os.path.join(video_path, '00000.png')
            img = Image.open(img_path)
            width, height = img.size
            mode = img.mode
            if mode == 'RGB':
                n_channel = 3
            elif mode == 'L':
                n_channel = 1
            else:
                raise ValueError(f'Unsupported mode {mode}.')
            info = f'{video} {frames} ({height},{width},{n_channel})'
            print(idx + 1, info)
            f.write(f'{info}\n')






            #
    # with open(meta_info_txt, 'w') as f:
    #     for idx, img_path in enumerate(img_list):
    #         img = Image.open(osp.join(gt_folder, img_path))  # lazy load
    #         width, height = img.size
    #         mode = img.mode
    #         if mode == 'RGB':
    #             n_channel = 3
    #         elif mode == 'L':
    #             n_channel = 1
    #         else:
    #             raise ValueError(f'Unsupported mode {mode}.')
    #
    #         info = f'{img_path} ({height},{width},{n_channel})'
    #         print(idx +1, info)
    #         f.write(f'{info}\n')

if __name__ == '__main__':
    generate_meta_info_div2k()