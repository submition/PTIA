
import torch.nn as nn

from util.util import *



class GuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GuidedFilter, self).__init__()
        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)

    def forward(self, input, ref):
        n_x, t_x, c_x, h_x, w_x = input.size()
        n_y, t_y, c_y, h_y, w_y = ref.size()
        output = torch.zeros_like(input).to(input.device)

        assert n_x == n_y
        assert c_x == 1 or c_x == c_y
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1
        for index in range(t_x):
            # N
            x = input[:, index, ...]
            y = input[:, index, ...]
            N = self.boxfilter(torch.tensor(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0)))

            mean_x = self.boxfilter(x) / N
            mean_y = self.boxfilter(y) / N
            # cov_xy
            cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
            # var_x
            var_x = self.boxfilter(x * x) / N - mean_x * mean_x

            # A
            A = cov_xy / (var_x + self.eps)
            # b
            b = mean_y - A * mean_x

            mean_A = self.boxfilter(A) / N
            mean_b = self.boxfilter(b) / N

            output[:, index, ...] = mean_A * x + mean_b

        return output


class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()
        self.r = r

    def forward(self, x):
        assert x.dim() == 4
        return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)



def diff_y(input, r):
    assert input.dim() == 4

    left = input[:, :, :, r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:] - input[:, :, :, :-2 * r - 1]
    right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1: -r - 1]

    output = torch.cat([left, middle, right], dim=3)

    return output


def diff_x(input, r):
    assert input.dim() == 4

    left = input[:, :, r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:] - input[:, :, :-2 * r - 1]
    right = input[:, :, -1:] - input[:, :, -2 * r - 1: -r - 1]

    output = torch.cat([left, middle, right], dim=2)

    return output