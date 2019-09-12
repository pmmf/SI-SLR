import torch
from torch.nn.functional import log_softmax


def softCrossEntropy(outputs, targets):
    r"""
    Args:
    outputs: output of the last layer before softmax, tensor of size (N, C)
    targets: ground-truth probabilities of each class, tensor of size (N, C)

    Outputs:
    loss: empirical cross-entropy between targets and outputs, tensor of size (1,)
    """

    log_predictions = log_softmax(outputs, dim=1)
    loss = -(targets.unsqueeze(1) @ log_predictions.unsqueeze(2)).mean()
    return loss


def softCrossEntropyUniform(outputs):
    r"""
    Args:
    outputs: output of the last layer before softmax, tensor of size (N, C)

    Outputs:
    loss: empirical cross-entropy between the uniform distribution and outputs,
        tensor of size (1,)
    """
    return -log_softmax(outputs, dim=1).mean()


def main():
    preds = torch.Tensor([-1, 3, 0]).reshape(1, -1)
    probs = torch.Tensor([0.2, 0.5, 0.3]).reshape(1, -1)

    xen = softCrossEntropy(preds, probs)
    print('xen', xen)

    n_classes = 7
    n_examples = 100
    probs = torch.ones(n_examples, n_classes)/n_classes
    preds = torch.zeros(n_examples, n_classes).float().random_(-10, 10)

    # xen1 = softCrossEntropy(preds, probs)
    # print('xen1', xen1)
    xen2 = softCrossEntropyUniform(probs)
    print('xen2', xen2)

    # print('err', (xen1 - xen2).abs().sum())

main()