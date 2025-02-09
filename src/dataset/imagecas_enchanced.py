"""
This file is released under  CC BY-NC-ND 4.0 license.
If you have any question, please contact zhuyh19@mails.tsinghua.edu.cn
"""

from typing import Any, Dict, Optional

import pysdf
import torch

# import torch.utils
import torch.utils.data
import trimesh
from lightning import LightningDataModule
from torch.utils.data import DataLoader


class ImageCASEnhancedDataset(torch.utils.data.Dataset):
    """
    ImageCASEnhancedDataset is a class that loads the ImageCASEnhanced dataset with SDF function.
    [ImageCAS](https://arxiv.org/abs/2211.01607) is a public dataset containing HU/vesselmask volume of CTA scans.
    ImageCAS Enhanced is derived from ImageCAS.
    """

    def __init__(
        self,
        surface_folder: str,
        filelist_path=None,
        filelist=None,
    ) -> None:

        self.surface_folder = surface_folder
        if filelist is None:
            with open(filelist_path) as f:
                self.filelist = [_.replace("\n", "") for _ in f.readlines()]
        else:
            filelist = filelist

        self.cache = {}
        # cache will consumming ~25 GB memory, saving the loading and building SDF time

    def load_file(self, idx: int):

        mesh = trimesh.load(f"{self.surface_folder}/{self.filelist[idx]}.ply")
        center = mesh.vertices.mean(axis=0)
        mesh.vertices -= center

        sdf_func = pysdf.SDF(mesh.vertices, mesh.faces)
        return (mesh, sdf_func)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx: int):

        if self.cache.get(idx) is None:
            self.cache[idx] = self.load_file(idx)

        return self.cache[idx]


class ImageCASEnhancedDataModule(LightningDataModule):
    """pytorch lightning data module for ImageCASEnhanced dataset."""

    def __init__(
        self,
        surface_folder="data/imagecas_left_surface",
        train_filelist_path="data/filelist_train.txt",
        val_filelist_path="data/filelist_filelist.txt",
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.surface_folder = surface_folder
        self.train_filelist_path = train_filelist_path
        self.val_filelist_path = val_filelist_path

        self.data_train = ImageCASEnhancedDataset(
            surface_folder, train_filelist_path
        )
        self.data_val = ImageCASEnhancedDataset(
            surface_folder, val_filelist_path
        )
        self.data_test = self.data_val

        self.batch_size_per_device = 1

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            collate_fn=lambda x: x,
            num_workers=0,
            pin_memory=False,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            collate_fn=lambda x: x,
            num_workers=0,
            pin_memory=False,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            collate_fn=lambda x: x,
            num_workers=0,
            pin_memory=False,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass
