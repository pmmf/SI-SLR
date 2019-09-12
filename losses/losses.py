import torch
from torch.nn import functional as F
import itertools

cross_entropy_loss = F.cross_entropy


def pairwise_loss(signers_combo_list, h_list):
    """Computes pairwise signer-transfer loss"""
    loss = 0
    for s1, s2 in signers_combo_list:
        for l_s1, l_s2 in zip(h_list[s1], h_list[s2]):
            loss += torch.sum((l_s1 - l_s2)**2)
    return loss


def signer_transfer_loss(h_conv_split, h_dense_split, signers_on_batch):
    """Computes signer-transfer loss"""
    # generate pairwise signer combo list
    signers_combo_list = list(itertools.combinations(signers_on_batch, 2))

    # pairwise loss for conv and dense layers
    conv_loss = pairwise_loss(signers_combo_list, h_conv_split)
    dense_loss = pairwise_loss(signers_combo_list, h_dense_split)

    loss = (conv_loss + dense_loss) * 1/len(signers_combo_list)

    return loss


if __name__ == '__main__':
    pass
