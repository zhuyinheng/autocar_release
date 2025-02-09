import torch


class DiceLossLogit(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        """
        pred: N, torch.float32, before sigmoid
        gt: N, torch.float32
        """
        pred = torch.sigmoid(pred)
        inter = (pred * gt).sum()
        denom = (pred**2 + gt**2).sum()

        return 1 - (2.0 * inter + 1) / (denom.clamp(min=1) + 1)


class DiceScoreLogit(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        """
        pred: N, torch.float32, before sigmoid
        gt: N, torch.float32
        """
        pred = (torch.sigmoid(pred) > 0.5) * 1.0
        inter = (pred * gt).sum()
        denom = (pred + gt).sum()

        return (2.0 * inter + 1e-5) / (denom + 1e-5)
