import torch
import torch.nn as nn
from .common import loss_digamma

class InferNet(nn.Module):
    def __init__(self, sample_shape, num_classes, dropout=0.5):
        super().__init__()
        if len(sample_shape) == 1:
            self.conv = nn.Sequential()
            fc_in = sample_shape[0]
        else:  # 3
            dims = [sample_shape[0], 20, 50]
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=dims[0], out_channels=dims[1], kernel_size=5, stride=1, padding=2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=dims[1], out_channels=dims[2], kernel_size=5, stride=1, padding=2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(),
            )
            fc_in = sample_shape[1] // 4 * sample_shape[2] // 4 * dims[2]

        fc_dims = [fc_in, min(fc_in, 500), num_classes]
        self.fc = nn.Sequential(
            nn.Linear(fc_dims[0], fc_dims[1]),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(fc_dims[1], fc_dims[2]),
            nn.ReLU(),
        )

    def forward(self, x):
        out_conv = self.conv(x).view(x.shape[0], -1)
        evidence = self.fc(out_conv)
        return evidence


class TMC(nn.Module):
    def __init__(self, sample_shapes: list, num_classes: int, annealing=50):
        assert len(sample_shapes[0]) in [1, 3], '`sample_shape` is 1 for vector or 3 for image.'
        super().__init__()
        self.annealing = annealing
        self.num_views = len(sample_shapes)
        self.classes = num_classes
        self.inferences = nn.ModuleList([InferNet(shape, num_classes) for shape in sample_shapes])

    def forward(self, x, target=None, epoch=0):
        view_e = dict()
        view_a = dict()
        for v in x.keys():
            view_e[v] = self.inferences[v](x[v])
            view_a[v] = view_e[v] + 1

        fusion_a = self.DS_Combine(view_a)
        fusion_e = fusion_a - 1

        return {
            'evidence': view_e,
            'probability': fusion_e,
            'loss': self.loss(target, view_a, fusion_a, epoch)
        }

    # Objective function
    def loss(self, target, view_a: dict, fusion_a: dict, epoch):
        if target is None:
            return None
        loss = loss_digamma(fusion_a, target, kl_penalty=min(1., epoch / self.annealing))
        for v in view_a.keys():
            loss += loss_digamma(view_a[v], target, kl_penalty=min(1., epoch / self.annealing))
        return loss # shape=(batch_size,)

    # Other functions...
    def DS_Combine(self, alpha):
        def DS_Combine_two(alpha1, alpha2):
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v] - 1
                b[v] = E[v] / (S[v].expand(E[v].shape))
                u[v] = self.classes / S[v]

            # b^0 @ b^(0+1)
            bb = torch.bmm(b[0].view(-1, self.classes, 1), b[1].view(-1, 1, self.classes))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            # calculate C
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            C = bb_sum - bb_diag

            # calculate b^a
            b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - C).view(-1, 1).expand(b[0].shape))
            # calculate u^a
            u_a = torch.mul(u[0], u[1]) / ((1 - C).view(-1, 1).expand(u[0].shape))

            # calculate new S
            S_a = self.classes / u_a
            # calculate new e_k
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a

        alpha_a = alpha[0]
        for v in range(len(alpha) - 1):
            if v == 0:
                alpha_a = DS_Combine_two(alpha[0], alpha[1])
            else:
                alpha_a = DS_Combine_two(alpha_a, alpha[v + 1])
        return alpha_a
