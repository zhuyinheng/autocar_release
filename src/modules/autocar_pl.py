"""
This file is released under  CC BY-NC-ND 4.0 license.
If you have any question, please contact zhuyh19@mails.tsinghua.edu.cn
"""

from typing import Any, Dict, Tuple

import rootutils
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.dataset.camera_sampler import CameraPairSampler
from src.dataset.deform_augmentor import SinDeformAugmentor
from src.dataset.imagecas_enchanced import ImageCASEnhancedDataset
from src.modules.autocar import AutoCAR
from src.modules.loss import DiceLossLogit, DiceScoreLogit
from src.modules.renderer import MeshRenderer
from src.utils.camera import cam_list_to_tensor


class AutoCARLit(LightningModule):
    def __init__(
        self,
        recon_net,
        deform_augmentor: SinDeformAugmentor,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        render_batch_size: int = 2,
        render_view_num: int = 2,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.recon_net = AutoCAR(recon_net)
        self.deform_augmentor = deform_augmentor
        self.camera_pair_sampler = CameraPairSampler()
        self.mesh_renderer = MeshRenderer()
        # loss function
        self.criterion_bce = torch.nn.BCEWithLogitsLoss()
        self.criterion_dice = DiceLossLogit()
        self.criterion_centerness = torch.nn.MSELoss()
        # metric objects for calculating and averaging accuracy across batches
        self.metrics_DICE = DiceScoreLogit()
        # self.train_acc = Accuracy(task="multiclass", num_classes=10)
        # self.val_acc = Accuracy(task="multiclass", num_classes=10)
        # self.test_acc = Accuracy(task="multiclass", num_classes=10)

        # Epoch-wise logging
        self.train_loss = MeanMetric()
        self.train_loss_DICE = MeanMetric()
        self.train_loss_bce = MeanMetric()
        self.train_loss_centerness = MeanMetric()

        self.val_loss = MeanMetric()
        self.val_loss_DICE = MeanMetric()
        self.val_loss_bce = MeanMetric()
        self.val_loss_centerness = MeanMetric()
        self.val_metric_dice = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_dice_best = MaxMetric()

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        pass

    def model_step(self, batch):
        if self.global_step % 10 == 0:
            torch.cuda.empty_cache()
        mesh = batch[0][0]
        sdf_func = batch[0][1]
        B = self.hparams.render_batch_size
        cams = self.camera_pair_sampler.sample_batch(B)
        # cams: [[a_0,b_0], [a_1,b_1], ..., [a_B,b_B]]
        cams_a = [cam[0] for cam in cams]
        cams_b = [cam[1] for cam in cams]
        meshes, deformed_meshes = self.deform_augmentor([mesh])
        mask_a, _ = self.mesh_renderer.render_mask_depth(meshes, cams_a)
        mask_b, _ = self.mesh_renderer.render_mask_depth(
            deformed_meshes, cams_b
        )
        device = mask_a.device
        # mask_a: 1 x B x H x W
        # mask_b: 1 x B x H x W
        masks = torch.stack([mask_a[0], mask_b[0]], 1)
        # masks: B x V(2) x H x W
        world2pix4x4 = torch.stack(
            [
                cam_list_to_tensor(cam_list=cams_a, device="cpu"),
                cam_list_to_tensor(cam_list=cams_b, device="cpu"),
            ],
            1,
        ).to(device)

        # world2pix4x4: B x V x 4 x 4
        pred, world_coords = self.recon_net(masks, world2pix4x4)
        self.log(
            "N_pts", pred.shape[0], on_step=True, on_epoch=False, prog_bar=True
        )
        if pred.shape[0] > 320000:
            print(f"Too many points({pred.shape[0]}), skip")
            return (None, None, None, None, None, None)
        # pred: sparse tensor, C: N_b x 4, F: N_b x 2
        # world_coords: N_b x 3, batched with same coords_manager with pred

        # calculating loss
        gt_sdf = sdf_func(world_coords.cpu().numpy())
        gt_sdf = torch.tensor(gt_sdf, dtype=torch.float32, device=device)
        # gt_sdf: >0: inside surface, <0: outside surface
        pred_mask_logit = pred.F[:, 0]
        pred_centerness = pred.F[:, 1]
        sparse_volume_mask = (gt_sdf > 0).type(torch.float32)
        sparse_volume_psudo_centerness = torch.exp(gt_sdf)
        loss_bce = self.criterion_bce(pred_mask_logit, sparse_volume_mask)
        loss_dice = self.criterion_dice(pred_mask_logit, sparse_volume_mask)
        loss_centerness = self.criterion_centerness(
            pred_centerness, sparse_volume_psudo_centerness
        )
        loss = loss_bce + loss_dice + loss_centerness
        return (
            loss,
            loss_dice,
            loss_bce,
            loss_centerness,
            pred_mask_logit,
            sparse_volume_mask,
        )

    def training_step(self, batch, batch_idx):

        (
            loss,
            loss_dice,
            loss_bce,
            loss_centerness,
            pred_mask_logit,
            sparse_volume_mask,
        ) = self.model_step(batch)
        if loss is None:
            return None
        self.log(
            "train/step/loss",
            loss.detach().item(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        # epoch-wise log
        self.train_loss(loss)
        self.train_loss_DICE(loss_dice)
        self.train_loss_bce(loss_bce)
        self.train_loss_centerness(loss_centerness)
        self.log(
            "train/loss",
            self.train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/loss_DICE",
            self.train_loss_DICE,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/loss_bce",
            self.train_loss_bce,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/loss_centerness",
            self.train_loss_centerness,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def on_validation_start(self) -> None:

        self.val_loss.reset()
        self.val_loss_DICE.reset()
        self.val_loss_bce.reset()
        self.val_loss_centerness.reset()
        self.val_metric_dice.reset()

    def validation_step(self, batch, batch_idx: int) -> None:

        (
            loss,
            loss_dice,
            loss_bce,
            loss_centerness,
            pred_mask_logit,
            sparse_volume_mask,
        ) = self.model_step(batch)
        if loss is None:
            return None
        dice_score = self.metrics_DICE(pred_mask_logit, sparse_volume_mask)
        self.val_metric_dice(dice_score)
        self.val_loss(loss)
        self.val_loss_DICE(loss_dice)
        self.val_loss_bce(loss_bce)
        self.val_loss_centerness(loss_centerness)
        # step-wise log
        self.log(
            "val/step/loss",
            loss.detach().item(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        self.log(
            "val/step/dice_score",
            dice_score.detach().item(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        # epoch-wise log
        self.log(
            "val/dice_score",
            self.val_metric_dice,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/loss",
            self.val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            "val/loss_DICE",
            self.val_loss_DICE,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/loss_bce",
            self.val_loss_bce,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/loss_centerness",
            self.val_loss_centerness,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        dice = self.val_metric_dice.compute()  # get current val dice
        self.val_dice_best(dice)  # update best so far val dice

        self.log(
            "val/dice_best",
            self.val_dice_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        # test is defined on real world dataset
        pass

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(
            params=self.trainer.model.parameters()
        )
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/dice_score",
                    "interval": "epoch",
                    "frequency": 1,
                    "strict": False,
                    "verbose": True,
                },
            }
        return {"optimizer": optimizer}


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
