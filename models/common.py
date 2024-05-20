import torch
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