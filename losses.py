from torchkge.utils import MarginLoss, BinaryCrossEntropyLoss, LogisticLoss

def get_loss(loss_name, args):
    if loss_name == 'MarginLoss':
        return MarginLoss(args.margin)
    elif loss_name == 'BinaryCrossEntropyLoss':
        return BinaryCrossEntropyLoss()