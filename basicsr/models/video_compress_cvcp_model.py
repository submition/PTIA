import torch
from collections import Counter
from os import path as osp
from torch import distributed as dist
from tqdm import tqdm

from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img, tensor2img_v1
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.registry import MODEL_REGISTRY
from .video_base_model import VideoBaseModel

from collections import OrderedDict

@MODEL_REGISTRY.register()
class VideoCompressCVCPModel(VideoBaseModel):

    def __init__(self, opt):
        super(VideoCompressCVCPModel, self).__init__(opt)
        if self.is_train:
            self.fix_flow_iter = opt['train'].get('fix_flow')
        self.opt = opt

    def setup_optimizers(self):
        train_opt = self.opt['train']
        flow_lr_mul = train_opt.get('flow_lr_mul', 1)
        logger = get_root_logger()
        logger.info(f'Multiple the learning rate for flow network with {flow_lr_mul}.')
        if flow_lr_mul == 1:
            optim_params = self.net_g.parameters()
        else:  # separate flow params and normal params for different lr
            normal_params = []
            flow_params = []
            for name, param in self.net_g.named_parameters():
                if 'spynet' in name:
                    flow_params.append(param)
                else:
                    normal_params.append(param)
            optim_params = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': train_opt['optim_g']['lr']
                },
                {
                    'params': flow_params,
                    'lr': train_opt['optim_g']['lr'] * flow_lr_mul
                },
            ]

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    # override
    def feed_data(self, data):
        # print("******************************8", data['lq'].shape)
        # print("******************************8",  data['gt'].shape)

        self.lq = data['lq'].to(self.device)

        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def freeze_para(self, current_iter):
        if self.fix_flow_iter:
            logger = get_root_logger()
            if current_iter == 1:
                logger.info(f'Fix flow network and feature extractor for {self.fix_flow_iter} iters.')
                for name, param in self.net_g.named_parameters():
                    if 'spynet' in name or 'edvr' in name:
                        param.requires_grad_(False)
            elif current_iter == self.fix_flow_iter:
                logger.warning('Train all the parameters.')
                self.net_g.requires_grad_(True)

        # super(VideoMetaRecurrentModel, self).optimize_parameters(current_iter)

    # override
    def optimize_parameters(self, current_iter):
        self.freeze_para(current_iter)
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()

        # pixel loss
        if self.cri_pix:
            # print("!!!!!!!!!!!!!!!!!!",self.output.shape, self.gt.shape)  #torch.Size([4, 7, 3, 256, 256]) torch.Size([4, 7, 3, 256, 256])
            l_pix = self.cri_pix(self.output, self.gt[:,:,0:1,:,:])
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)


    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset = dataloader.dataset
        dataset_name = dataset.opt['name']
        with_metrics = self.opt['val']['metrics'] is not None

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {}
                num_frame_each_folder = Counter(dataset.data_info['folder'])
                for folder, num_frame in num_frame_each_folder.items():
                    self.metric_results[folder] = torch.zeros(
                        num_frame, len(self.opt['val']['metrics']), dtype=torch.float32, device='cuda')
            # initialize the best metric results
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        rank, world_size = get_dist_info()
        if with_metrics:
            for _, tensor in self.metric_results.items():
                tensor.zero_()

        metric_data = dict()
        num_folders = len(dataset)
        num_pad = (world_size - (num_folders % world_size)) % world_size
        if rank == 0:
            pbar = tqdm(total=len(dataset), unit='folder')
        # Will evaluate (num_folders + num_pad) times, but only the first num_folders results will be recorded.
        # (To avoid wait-dead)

        for i in range(rank, num_folders + num_pad, world_size):
            idx = min(i, num_folders - 1)
            val_data = dataset[idx]
            folder = val_data['folder']

            # compute outputs
            val_data['lq'].unsqueeze_(0)
            val_data['gt'].unsqueeze_(0)
            self.feed_data(val_data)
            val_data['lq'].squeeze_(0)
            val_data['gt'].squeeze_(0)
            self.test()
            # self.crop_test()
            visuals = self.get_current_visuals()

            # tentative for out of GPU memory
            del self.lq
            del self.output
            if 'gt' in visuals:
                del self.gt
            torch.cuda.empty_cache()

            if self.center_frame_only:
                visuals['result'] = visuals['result'].unsqueeze(1)
                if 'gt' in visuals:
                    visuals['gt'] = visuals['gt'].unsqueeze(1)

            # evaluate
            if i < num_folders:
                for idx in range(visuals['result'].size(1)):
                    result = visuals['result'][0, idx, :, :, :]
                    result_img = tensor2img_v1([result], ycbcr2bgr=False)  # uint8, tensor2numpy
                    metric_data['img'] = result_img

                    if 'gt' in visuals:
                        gt = visuals['gt'][0, idx, :, :, :]
                        gt_img = tensor2img_v1(gt, ycbcr2bgr=False)   # uint8, tensor2numpy
                        metric_data['img2'] = gt_img[:,:,0]

                    if save_img:
                        if self.opt['is_train']:
                            raise NotImplementedError('saving image is not supported during training.')
                        else:
                            if self.center_frame_only:
                                clip_ = val_data['lq_path'].split('/')[-3]
                                seq_ = val_data['lq_path'].split('/')[-2]
                                name_ = f'{clip_}_{seq_}'
                                img_path = osp.join(self.opt['path']['visualization'], dataset_name, folder,
                                                    f"{name_}_{self.opt['name']}.png")
                            else:  # others
                                print("###########################################",idx)
                                img_path = osp.join(self.opt['path']['visualization'], dataset_name, folder,
                                                    f"{idx:08d}_{self.opt['name']}.png")
                            # image name only for REDS dataset
                        imwrite(result_img, img_path)

                    # calculate metrics
                    if with_metrics:
                        for metric_idx, opt_ in enumerate(self.opt['val']['metrics'].values()):
                            result = calculate_metric(metric_data, opt_)
                            # print("result:********************",result)
                            self.metric_results[folder][idx, metric_idx] += result


                # progress bar
                if rank == 0:
                    for _ in range(world_size):
                        pbar.update(1)
                        pbar.set_description(f'Folder: {folder}')
        #########

        if rank == 0:
            pbar.close()

        if with_metrics:
            if self.opt['dist']:
                # collect data among GPUs
                for _, tensor in self.metric_results.items():
                    dist.reduce(tensor, 0)
                dist.barrier()

            if rank == 0:
                self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def test(self):
        n = self.lq.size(1)
        self.net_g.eval()

        flip_seq = self.opt['val'].get('flip_seq', False)
        self.center_frame_only = self.opt['val'].get('center_frame_only', False)

        if flip_seq:
            self.lq = torch.cat([self.lq, self.lq.flip(1)], dim=1)

        with torch.no_grad():
            self.output = self.net_g(self.lq)

        if flip_seq:
            output_1 = self.output[:, :n, :, :, :]
            output_2 = self.output[:, n:, :, :, :].flip(1)
            self.output = 0.5 * (output_1 + output_2)

        if self.center_frame_only:
            self.output = self.output[:, n // 2, :, :, :]

        self.net_g.train()


    def crop_test(self):
        test_cong = self.opt['val']['crop_test']
        sf = test_cong['scale']
        tile = test_cong['tile']
        tile_overlap = test_cong['tile_overlap']
        window_size = test_cong['window_size']
        size_patch_testing = tile[1]
        assert size_patch_testing % window_size[-1] == 0, 'testing patch size should be a multiple of window_size.'

        if size_patch_testing:
            # divide the clip to patches (spatially only, tested patch by patch)
            overlap_size = tile_overlap[1]
            not_overlap_border = True

            # test patch by patch
            b, d, c, h, w = self.lq.size()

            stride = size_patch_testing - overlap_size
            h_idx_list = list(range(0, h - size_patch_testing, stride)) + [max(0, h - size_patch_testing)]
            w_idx_list = list(range(0, w - size_patch_testing, stride)) + [max(0, w - size_patch_testing)]
            E = torch.zeros(b, d, c, h * sf, w * sf)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = self.lq[..., h_idx:h_idx + size_patch_testing, w_idx:w_idx + size_patch_testing]
                    out_patch = self.net_g(in_patch).detach().cpu()
                    out_patch_mask = torch.ones_like(out_patch)

                    if not_overlap_border:
                        if h_idx < h_idx_list[-1]:
                            out_patch[..., -overlap_size // 2:, :] *= 0
                            out_patch_mask[..., -overlap_size // 2:, :] *= 0
                        if w_idx < w_idx_list[-1]:
                            out_patch[..., :, -overlap_size // 2:] *= 0
                            out_patch_mask[..., :, -overlap_size // 2:] *= 0
                        if h_idx > h_idx_list[0]:
                            out_patch[..., :overlap_size // 2, :] *= 0
                            out_patch_mask[..., :overlap_size // 2, :] *= 0
                        if w_idx > w_idx_list[0]:
                            out_patch[..., :, :overlap_size // 2] *= 0
                            out_patch_mask[..., :, :overlap_size // 2] *= 0

                    E[..., h_idx * sf:(h_idx + size_patch_testing) * sf,
                    w_idx * sf:(w_idx + size_patch_testing) * sf].add_(out_patch)
                    W[..., h_idx * sf:(h_idx + size_patch_testing) * sf,
                    w_idx * sf:(w_idx + size_patch_testing) * sf].add_(out_patch_mask)
            output = E.div_(W)

        else:
            _, _, _, h_old, w_old = self.lq.size()
            h_pad = (window_size[1] - h_old % window_size[1]) % window_size[1]
            w_pad = (window_size[2] - w_old % window_size[2]) % window_size[2]

            lq = torch.cat([self.lq, torch.flip(self.lq[:, :, :, -h_pad:, :], [3])], 3) if h_pad else self.lq
            self.lq = torch.cat([lq, torch.flip(lq[:, :, :, :, -w_pad:], [4])], 4) if w_pad else lq

            output = self.net_g(self.lq).detach().cpu()

            output = output[:, :, :, :h_old * sf, :w_old * sf]

        return output

        # n = self.lq.size(1)
        # self.net_g.eval()
        #
        # flip_seq = self.opt['val'].get('flip_seq', False)
        # self.center_frame_only = self.opt['val'].get('center_frame_only', False)
        #
        # if flip_seq:
        #     self.lq = torch.cat([self.lq, self.lq.flip(1)], dim=1)
        #
        # with torch.no_grad():
        #     self.output = self.net_g(self.lq)
        #
        # if flip_seq:
        #     output_1 = self.output[:, :n, :, :, :]
        #     output_2 = self.output[:, n:, :, :, :].flip(1)
        #     self.output = 0.5 * (output_1 + output_2)
        #
        # if self.center_frame_only:
        #     self.output = self.output[:, n // 2, :, :, :]

        self.net_g.train()

