import torch
import torch.nn as nn
from .common import loss_digamma
import torch.nn.functional as F

# KL Divergence calculator. alpha shape(batch_size, num_classes)
def KL(alpha):
    ones = torch.ones([1, alpha.shape[-1]], dtype=torch.float32, device=alpha.device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (alpha - ones).mul(torch.digamma(alpha) - torch.digamma(sum_alpha)).sum(dim=1, keepdim=True)
    kl = first_term + second_term
    return kl.reshape(-1)


def loss_log(alpha, labels, kl_penalty):
    y = F.one_hot(labels.long(), alpha.shape[-1])
    log_likelihood = torch.sum(y * (torch.log(alpha.sum(dim=-1, keepdim=True)) - torch.log(alpha)), dim=-1)
    loss = log_likelihood + kl_penalty * KL((alpha - 1) * (1 - y) + 1)
    return loss


def loss_digamma(alpha, labels, kl_penalty):
    y = F.one_hot(labels.long(), alpha.shape[-1])
    log_likelihood = torch.sum(y * (torch.digamma(alpha.sum(dim=-1, keepdim=True)) - torch.digamma(alpha)), dim=-1)
    loss = log_likelihood + kl_penalty * KL((alpha - 1) * (1 - y) + 1)
    return loss


def loss_mse(alpha, labels, kl_penalty):
    y = F.one_hot(labels.long(), alpha.shape[-1])
    sum_alpha = torch.sum(alpha, dim=-1, keepdim=True)
    err = (y - alpha / sum_alpha) ** 2
    var = alpha * (sum_alpha - alpha) / (sum_alpha ** 2 * (sum_alpha + 1))
    loss = torch.sum(err + var, dim=-1)
    loss = loss + kl_penalty * KL((alpha - 1) * (1 - y) + 1)
    return loss

class ViewSpecificDNN(nn.Module):
    def __init__(self, sample_shape, dropout=0.5):
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

        fc_dims = [fc_in, min(fc_in, 500)]
        self.fc = nn.Sequential(
            nn.Linear(fc_dims[0], fc_dims[1]),
            nn.Dropout(dropout),
            nn.ReLU(),
        )

    def forward(self, x):
        out_conv = self.conv(x).view(x.shape[0], -1)
        # z: view-specific representation
        z = self.fc(out_conv)
        # z.shape=(batch_size, min(fc_in, 500))
        return z
    

class ViewSpecificEDN(nn.Module):
    def __init__(self, sample_shape, num_classes, dropout=0.5):
        super().__init__()
        if len(sample_shape) == 1:
            fc_in = sample_shape[0]
        else:  # 3
            dims = [sample_shape[0], 20, 50]
            fc_in = sample_shape[1] // 4 * sample_shape[2] // 4 * dims[2]

        fc_dims = [min(fc_in, 500), num_classes]
        self.fc = nn.Sequential(
            nn.Linear(fc_dims[0], fc_dims[1]),
            nn.ReLU(),
        )

        self.C = num_classes

    def forward(self, x):
        # e: evidence
        e = self.fc(x)
        # e.shape=(batch_size, num_classes)
        alpha = e + 1
        S = torch.sum(alpha, dim=-1, keepdim=True)
        # S.shape=(batch_size, 1)

        # u: uncertainty
        u = self.C / S
        # u.shape=(batch_size, 1)

        # r: reliability
        r = 1 - u
        # r.shape=(batch_size, 1)
        return r, alpha

class DegradationLayer(nn.Module):
    def __init__(self, sample_shapes, global_dim):
        super(DegradationLayer, self).__init__()
        self.num_views = len(sample_shapes)
        # 初始化视角特定的处理层
        self.view_layers = nn.ModuleList()
        self.reconstruction_layers = nn.ModuleList()

        # 假设每个视角都使用简单的线性变换处理
        for shape in sample_shapes:
            if len(shape) == 1:
                fc_in = shape[0]
            else:  # 3
                dims = [shape[0], 20, 50]
                fc_in = shape[1] // 4 * shape[2] // 4 * dims[2]

            self.view_layers.append(nn.Linear(fc_in, global_dim))
            self.reconstruction_layers.append(nn.Linear(global_dim, fc_in))

        # 全局表示的线性层
        self.global_layer = nn.Linear(global_dim, global_dim)

    def forward(self, view_z, view_r):
        # 使用可靠性作为权重加权视角表示
        weighted_views = [r * layer(z) for r, layer, z in zip(view_r.values(), self.view_layers, view_z.values())]
        # 计算加权平均，对堆叠后的张量沿视角维度加权平均
        total_reliability = torch.stack(list(view_r.values())).sum(dim=0, keepdim=True).clamp(min=1e-5)
        weighted_sum = torch.stack(weighted_views).sum(dim=0)
        global_representation = weighted_sum / total_reliability

        # 对全局表示进行进一步处理
        global_representation = self.global_layer(global_representation).squeeze(0)

        # 尝试重构每个视角
        reconstructed_view_z = {k: recon_layer(global_representation).squeeze(0) for k, recon_layer in zip(view_z.keys(), self.reconstruction_layers)}
        
        return global_representation, reconstructed_view_z
    

class TaskSpecificLayer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TaskSpecificLayer, self).__init__()
        # 添加一些全连接层进行任务特定的处理
        self.fc1 = nn.Linear(input_dim, 100)  # 首先将输入维度降低或升高到一个中间维度
        self.relu = nn.ReLU()                # 非线性激活函数
        self.fc2 = nn.Linear(100, num_classes) # 最后一个全连接层输出类别数

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TMF(nn.Module):
    def __init__(self, sample_shapes: list, num_classes: int, annealing=20):
        assert len(sample_shapes[0]) in [1, 3], '`sample_shape` is 1 for vector or 3 for image.'
        super().__init__()
        self.annealing = annealing
        self.num_views = len(sample_shapes)
        self.classes = num_classes
        self.viewSpecificDNNs = nn.ModuleList([ViewSpecificDNN(shape) for shape in sample_shapes])
        self.viewSpecificEDNs = nn.ModuleList([ViewSpecificEDN(shape, num_classes) for shape in sample_shapes])
        # 我也不知道global_dim设多少好
        global_dim = num_classes
        self.degradation_layer = DegradationLayer(sample_shapes, global_dim)
        self.task_layer = TaskSpecificLayer(global_dim, num_classes)

    def forward(self, x, target=None, epoch=0):
        view_z = dict()
        view_r = dict()
        view_a = dict()

        for v in x.keys():
            # view-specific representation
            view_z[v] = self.viewSpecificDNNs[v](x[v])
            # reliability, alpha of view v
            view_r[v], view_a[v] = self.viewSpecificEDNs[v](view_z[v])

        # print(f'view_z: {view_z}')
        # h: the trusted multi-view common representation
        h, reconstructed_view_z = self.degradation_layer(view_z, view_r)
        
        # 应用任务特定层
        probability = F.softmax(self.task_layer(h), dim=1)
        
        return {
            'probability': probability, # 概率分布得出预测类别
            'loss': self.loss(probability, target,
                reconstructed_view_z, view_z, view_r,
                view_a, epoch)
        }
    
    def loss(self, 
             probability, target, 
             reconstructed_view_z, view_z, view_r,
             view_a: dict, epoch):
        # task-specific loss
        taskSpecificLoss = 0
        if target is not None:
            taskSpecificLoss = F.cross_entropy(probability, target)

        # reconstruction loss
        beta = [r ** 2 / sum(r ** 2 for r in view_r) for r in view_r]
        reconstructionLoss = sum(beta[k] * F.mse_loss(recon, view_z[k]) for k, recon in reconstructed_view_z.items())
        
        # view-specific evidential loss
        viewSpecificEvidentialLoss = 0
        if target is not None:
            # print(f'target.shape: {target.shape}')
            for v in view_a.keys():
                viewSpecificEvidentialLoss += loss_digamma(view_a[v], target, kl_penalty=min(1., epoch / self.annealing))
            
        epsilon = 1
        delta = 1
        totalLoss = taskSpecificLoss + epsilon * reconstructionLoss + delta * viewSpecificEvidentialLoss
        return totalLoss
