"""
This file is released under  CC BY-NC-ND 4.0 license.
If you have any question, please contact zhuyh19@mails.tsinghua.edu.cn
"""

import kornia
import rootutils
import torch
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.dataset.imagecas_enchanced import ImageCASEnhancedDataset
from src.modules.encoder2d import StackedHourGlassEncoder
from src.modules.minkunet import MinkUNet18A
from src.modules.ray_casting import SparseBackwardProjection


class AutoCAR(torch.nn.Module):
    """module for AutoCAR in PyTorch.
    - Used for both training and inference.
    - Without dependency on PyTorch Lightning.
    - Only include reconstrution, no rendering or loss calculation.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        cfg = OmegaConf.create(cfg)
        self.encoder2d = StackedHourGlassEncoder(cfg.encoder2d.out_ch)
        self.ray_casting = SparseBackwardProjection(
            cfg.ray_casting.bbox_min,
            cfg.ray_casting.bbox_max,
            cfg.ray_casting.LODs,
        )
        self.unet3d = MinkUNet18A(
            cfg.unet3d.in_channels, cfg.unet3d.out_channels
        )

    def forward(self, masks, world2pix4x4):
        """
        masks: B x V x H x W, pytorch tensor
        world2pix4x4: B x V x 4 x 4, pytorch tensor
        """
        # B = masks.shape[0]
        # V = masks.shape[1]
        # H, W = masks.shape[2:]

        # sdf2d: B x V x H x W
        sdf2d = kornia.contrib.distance_transform(masks.type(torch.float32), 3)

        feature = self.encoder2d(torch.exp(-sdf2d))
        # feature: B x V x C x H x W

        sparse_volume, world_coords = self.ray_casting(
            sdf2d, feature, world2pix4x4
        )
        pred = self.unet3d(sparse_volume)
        return pred, world_coords


if __name__ == "__main__":
    import tqdm

    dataset = ImageCASEnhancedDataset(
        "data/imagecas_left_surface", "data/filelist.txt"
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, collate_fn=lambda x: x
    )
    model = AutoCAR().cuda()
    for i, data in enumerate(tqdm.tqdm(dataloader)):
        # print(data)
        model(data)
        # if i > 10:
        #     break
