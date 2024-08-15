from basicsr.modules import networks as N
import torch.nn as nn
import torch.nn.functional as F
from util.util import *
from basicsr.modules.spynet import SPyNet, flow_warp
from basicsr.utils.registry import ARCH_REGISTRY
import matplotlib.pyplot as plt


@ARCH_REGISTRY.register()
class PTIA(nn.Module):
    def __init__(self, n_flow=7, n_frame=7, n_resblock=30, n_feats=64, spynet_pretrained=None):
        super(PTIA, self).__init__()

        self.n_resblock = n_resblock
        self.n_frame = n_frame
        self.n_feats = n_feats
        self.n_flow = n_flow
        self.spynet = SPyNet(pretrained=spynet_pretrained)

        # feature extractor#############################################################################
        self.encoder = N.ContrasExtractorLayer(self.n_feats)   # 0.703M    VGG16

        # propagation and alignment module###########################################################
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        self.fusion = nn.ModuleDict()

        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']

        for i, module in enumerate(modules):
            self.deform_align[module] = N.TBIA(self.n_feats, self.n_feats, deformable_groups=8)
            self.backbone[module] = ResidualBlocksWithInputConv(
                (2 + i) * self.n_feats, self.n_feats, self.n_resblock)
            self.fusion[module] = nn.Conv2d(self.n_feats * 3, self.n_feats, 1, 1, 0, bias=True)

        # upsample##################################################################################
        self.reconstruction = ResidualBlocksWithInputConv(
            5 * self.n_feats, self.n_feats, 5)
        self.upsample1 = N.seq([N.conv(self.n_feats, self.n_feats * 2 * 2, mode='C'),
                                nn.PixelShuffle(2)])
        self.upsample2 = N.seq([N.conv(self.n_feats, self.n_feats * 2 * 2, mode='C'),
                                nn.PixelShuffle(2)])

        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def check_if_mirror_extended(self, lrs):
        """Check whether the input is a mirror-extended sequence.
        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.
        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        """

        self.is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lrs):
        """Compute optical flow using SPyNet for feature warping.
        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.
        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lrs.shape
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)  # lrs_2 -> lrs_1

        flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)  # lrs_1 -> lrs_2

        return flows_forward, flows_backward

    def forward(self, lrs):

        n, t, c, h, w = lrs.shape
        # assert h >= 64 and w >= 64, (
        #     'The height and width of inputs should be at least 64, '
        #     f'but got {h} and {w}.')

        # compute optical flow
        with torch.no_grad():
            flows_forward, flows_backward = self.compute_flow(lrs)
        # end = Time.time()

        feats = {}
        lr_flatten = lrs.view(-1, c, h, w)
        # L1
        lr_feature = self.encoder(lr_flatten)   #64, 64, 64
        # L2
        lr_feature_down2 = F.interpolate(lr_feature, scale_factor=0.5, mode='bilinear', align_corners=False)  #64, 32, 32
        # L3
        lr_feature_down4 = F.interpolate(lr_feature, scale_factor=0.25, mode='bilinear', align_corners=False)   ##64, 16, 16

        lr_feature = lr_feature.view(n, t, -1, h, w)
        lr_feature_down2 = lr_feature_down2.view(n, t, -1, h // 2, w // 2)
        lr_feature_down4 = lr_feature_down4.view(n, t, -1, h // 4, w // 4)

        feats['branch_L'] = [lr_feature[:, i, ...] for i in range(0, t)]
        feats['branch_M'] = [lr_feature_down2[:, i, ...] for i in range(0, t)]
        feats['branch_S'] = [lr_feature_down4[:, i, ...] for i in range(0, t)]

        for iter_ in [1, 2]:
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'  #backward_1, backward_2, forward_1, forward_2
                feats[module] = []

                if direction == 'backward':
                    flows = flows_backward
                else:
                    flows = flows_forward
                feats = self.propagate(feats, flows, module)


        out = self.upsample(lrs, feats)
        return out

    def propagate(self, feats, flows, module_name):
        """Propagate the latent features throughout the sequence.
        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            module_name (str): The name of the propgation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.

        Return:
            dict(list[tensor]): A dictionary containing all the propagated
                features. Each key in the dictionary corresponds to a
                propagation branch, which is represented by a list of tensors.
        """
        n, t, _, h, w = flows.size()

        # backward-time propgation
        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)
        mapping_idx = list(range(0, len(feats['branch_L'])))  # [0, 7)
        mapping_idx += mapping_idx[::-1]  # [0, ... 6, 6, ... 0]


        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]  # [6,... 0]
            flow_idx = frame_idx  # [6,... 0]
        feat_prop = flows.new_zeros(n, self.n_feats, h, w)

        # Start to propagate
        for i, idx in enumerate(frame_idx):    #i:[0,6], idx: [6,0]

            feat_current = feats['branch_L'][mapping_idx[idx]]  #[6,...,0]
            feat_current_down2 = feats['branch_M'][mapping_idx[idx]]
            feat_current_down4 = feats['branch_S'][mapping_idx[idx]]

            # second-order deformable alignment
            if i > 0:    # backward: idx=[5,1]
                current_feat = [feat_current, feat_current_down2, feat_current_down4]   #reference_feature: [5,...,0]  [1,...,6]
                if 'backward' in module_name:
                    nbr_feat = [feats['branch_L'][mapping_idx[idx + 1]],   #[6,...,1] neighbor_feature  #
                                feats['branch_M'][mapping_idx[idx + 1]],
                                feats['branch_S'][mapping_idx[idx + 1]]]

                else: #forward
                    nbr_feat = [feats['branch_L'][mapping_idx[idx - 1]],    #[0,...,5]
                                feats['branch_M'][mapping_idx[idx - 1]],
                                feats['branch_S'][mapping_idx[idx - 1]]]

                flow_n1 = flows[:, flow_idx[i], :, :, :]

                cond_n1 = self.deform_align[module_name](nbr_feat, current_feat, feat_prop, flow_n1)
                #feature_vis(cond_n1, 'feature_ours_1.png')
                # feature_vis(cond_n1, "feature1.jpg")

                # initialize second-order features
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:  # second-order features

                    feat_n2 = feats[module_name][-2]
                    if 'backward' in module_name:
                        nbr_feat = [feats['branch_L'][mapping_idx[idx + 2]],
                                    feats['branch_M'][mapping_idx[idx + 2]],
                                    feats['branch_S'][mapping_idx[idx + 2]]]
                    else:
                        nbr_feat = [feats['branch_L'][mapping_idx[idx - 2]],
                                    feats['branch_M'][mapping_idx[idx - 2]],
                                    feats['branch_S'][mapping_idx[idx - 2]]]

                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]

                    flow_n2 = flow_n1 + flow_warp(flow_n2,
                                                        flow_n1.permute(0, 2, 3, 1))


                    cond_n2 = self.deform_align[module_name](nbr_feat, current_feat, feat_n2, flow_n2)

                    # feature_vis(cond_n2, 'feature_ours_2.png')

                feat_prop = torch.cat([cond_n1, feat_current, cond_n2], dim=1)  #b, 64*3, 64,64
                feat_prop = self.fusion[module_name](feat_prop)   #conv2d 64*3-->64
                # print(feats[module_name][0].shape)


            # concatenate and residual blocks
            feat = [feat_current] + [
                feats[k][idx]
                for k in feats if k not in ['branch_L', 'branch_M', 'branch_S', module_name]
            ] + [feat_prop]

            # print(len(feat))   #2,...5
            # print(module_name, len(feat))
            feat = torch.cat(feat, dim=1)
            feat_prop = feat_prop + self.backbone[module_name](feat)
            feats[module_name].append(feat_prop)
            # print("feat_" + module_name, len(feats[module_name]))

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def upsample(self, lqs, feats):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propgation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).

        """

        outputs = []
        num_outputs = len(feats['branch_L'])
        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]



        for i in range(0, lqs.size(1)):
            hr = [feats[k].pop(0) for k in feats if k != 'branch_L' and k != 'branch_M' and k != 'branch_S']  #backward_1, forward_1, backward_2, forward_2
            hr.insert(0, feats['branch_L'][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)
            hr = self.reconstruction(hr)
            hr = self.lrelu(self.upsample1(hr))  #PixelShuffle
            hr = self.lrelu(self.upsample2(hr))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            hr += self.img_upsample(lqs[:, i, :, :, :])

            outputs.append(hr)


        return torch.stack(outputs, dim=1)


class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with a convolution in front.
    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(N.RCAGroup(out_channels, out_channels, nb=num_blocks))
        # main.append(
        #     make_layer(
        #         ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """
        Forward function for ResidualBlocksWithInputConv.
        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)
        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)


class FLOW(nn.Module):
    def __init__(self, opt, net):
        super(FLOW, self).__init__()
        self.opt = opt
        self.netFlow = net
        self.n_frame = opt.n_frame
        self.center_frame_idx = self.n_frame // 2  # index of key frame
        self.n_flow = opt.n_flow

    def forward(self, lr_seq):
        off_f = []
        times = lr_seq.shape[1] // self.n_flow
        for time in range(times):
            curr = lr_seq[:, time * self.n_flow + 1, ...]  # align7
            last = lr_seq[:, time * self.n_flow, ...]
            off_f.append(self.netFlow(curr, last))
        return off_f

import argparse
def str2bool(v):
    return v.lower() in ('yes', 'y', 'true', 't', '1')


def random_num(size, end):
    range_ls = [i for i in range(end)]
    num_ls = []
    for i in range(size):
        num = random.choice(range_ls)
        range_ls.remove(num)
        num_ls.append(num)
    return num_ls


def feature_vis_init(feat, path):
    b, c, h, w = feat.shape
    feat_vis = feat[0, : , :, :].detach().cpu().numpy()
    print(feat_vis.shape)   #64, 64, 64
    channel_num = random_num(25, feat_vis.shape[0])
    plt.figure(figsize=(10, 10))
    for index, channel in enumerate(channel_num):
        ax = plt.subplot(5, 5, index + 1, )
        plt.imshow(feat_vis[channel, :, :])
    plt.savefig(path, dpi=300)

def feature_vis(feat, path):
    b, c, h, w = feat.shape
    feat_vis = feat[0, : , :, :].detach().cpu().numpy()
    #print(feat_vis.shape)   #64, 64, 64
    channel = 25
    # plt.figure(figsize=(10, 10))
    # for index, channel in enumerate(channel_num):
    #     ax = plt.subplot(5, 5, index + 1, )
    plt.imshow(feat_vis[channel, :, :])
    plt.savefig(path, dpi=300)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', type=str2bool, default=True)
    parser.add_argument('--n_frame', type=int, default=5, help='Number of frames processed at once.')
    parser.add_argument('--n_flow', type=int, default=5)
    args = parser.parse_args()
    input = torch.rand(2, 7, 3, 64, 64).cuda()
    net = CEAVSR(args).cuda()
    print("CEAVSR have {:.3f}M paramerters in total".format(sum(x.numel() for x in net.parameters()) / 1000000.0))
    out = net(input)
    print(out.shape)
