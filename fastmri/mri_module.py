import pathlib
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning import _logger as log
from torch.utils.data import DataLoader, DistributedSampler
from .math import complex_abs_numpy
import os
import sys
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fastmri
# from fastmri import evaluate1
from fastmri.data import SliceDataset
from fastmri.data.volume_sampler import VolumeSampler
# from fastmri.evaluate import nmse, psnr, ssim
from torchmetrics.metric import Metric

import fastmri.evaluate2 as evaluate
class DistributedMetricSum(Metric):
    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("quantity", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch: torch.Tensor):  # type: ignore
        self.quantity += batch

    def compute(self):
        return self.quantity

class MriModule(pl.LightningModule):
    """
    Abstract super class for deep larning reconstruction models.
    
    This is a subclass of the LightningModule class from pytorch_lightning,
    with some additional functionality specific to fastMRI:
        - fastMRI data loaders
        - Evaluating reconstructions
        - Visualization
        - Saving test reconstructions

    To implement a new reconstruction model, inherit from this class and
    implement the following methods:
        - train_data_transform, val_data_transform, test_data_transform:
            Create and return data transformer objects for each data split
        - training_step, validation_step, test_step:
            Define what happens in one step of training, validation, and
            testing, respectively
        - configure_optimizers:
            Create and return the optimizers

    Other methods from LightningModule can be overridden as needed.
    """

    def __init__(
        self,
        data_path,
        challenge,
        exp_dir,
        exp_name,
        dataset_name,
        train_csv,
        val_csv,
        test_csv,
        test_split="test",
        num_log_images=4,
        sample_rate=1.0,
        batch_size=1,
        num_workers=4,
        use_ddp=False,
        **kwargs,
    ):
        """
        Args:
            data_path (pathlib.Path): Path to root data directory. For example, if
                knee/path is the root directory with subdirectories
                multicoil_train and multicoil_val, you would input knee/path for
                data_path.
            challenge (str): Name of challenge from ('multicoil', 'singlecoil').
            exp_dir (pathlib.Path): Top directory for where you want to store log
                files.
            exp_name (str): Name of this experiment - this will store logs in
                exp_dir / {exp_name}.
            test_split (str): Name of test split from ("test", "challenge").
            sample_rate (float, default=1.0): Fraction of models from the
                dataset to use.
            batch_size (int, default=1): Batch size.
            num_workers (int, default=4): Number of workers for PyTorch dataloader.
        """
        super().__init__()

        self.data_path = data_path
        
        self.dataset_name = dataset_name
        self.train_csv=train_csv
        self.val_csv=val_csv
        self.test_csv = test_csv
        
        self.challenge = challenge
        self.exp_dir = exp_dir
        self.exp_name = exp_name
        self.test_split = test_split
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.num_workers = num_workers        
        self.val_log_indices = None

        self.use_ddp = use_ddp
        self.num_log_images = num_log_images
        
        self.NMSE = DistributedMetricSum()
        self.SSIM = DistributedMetricSum()
        self.PSNR = DistributedMetricSum()
        self.ValLoss = DistributedMetricSum()
        self.TotExamples = DistributedMetricSum()
        self.TotSliceExamples = DistributedMetricSum()
        # self.NMSE = DistributedMetricSum(name="NMSE")
        # self.SSIM = DistributedMetricSum(name="SSIM")
        # self.PSNR = DistributedMetricSum(name="PSNR")
        # self.ValLoss = DistributedMetricSum(name="ValLoss")
        # self.TotExamples = DistributedMetricSum(name="TotExamples")
        
        # self.NMSE = nmse()
        # self.SSIM = ssim()
        # self.PSNR = psnr()
        # self.ValLoss = VaLloss()
        # self.TotExamples = DistributedMetricSum(name="TotExamples")
        
        self.best_psnr = 0

    def _create_data_loader(self, data_transform, data_partition, sample_rate=None):
        sample_rate = sample_rate or self.sample_rate
        if data_partition == 'train':
            csv_file = self.train_csv
        elif data_partition == 'val':
            csv_file = self.val_csv
        else:
            csv_file = self.test_csv
        
        dataset = SliceDataset(
            root=self.data_path / f"{self.challenge}_{data_partition}",
            transform=data_transform,
            sample_rate=sample_rate,
            csv_file=csv_file,
            challenge=self.challenge,
            mode=data_partition
        )

        is_train = data_partition == "train"

        # ensure that entire volumes go to the same GPU in the ddp setting
        sampler = None
        if self.use_ddp:
            if is_train:
                sampler = DistributedSampler(dataset)
            else:
                sampler = VolumeSampler(dataset)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=is_train,
            sampler=sampler,
        )

        return dataloader

    def train_data_transform(self):
        pass

    def val_data_transform(self):
        pass
    
    def test_data_transform(self):
        pass
    
    def train_dataloader(self):
        return self._create_data_loader(self.train_data_transform(), data_partition="train")

    def val_dataloader(self):
        return self._create_data_loader(self.val_data_transform(), data_partition="val")

    def test_dataloader(self):
        return self._create_data_loader(self.test_data_transform(),data_partition="val")

    def _visualize(self, val_outputs, val_targets):
        def _normalize(image):
            image = image[np.newaxis]
            image = image - image.min()
            return image / image.max()

        def _save_image(image, tag):
            grid = torchvision.utils.make_grid(torch.Tensor(image), nrow=4, pad_value=1)
            self.logger.experiment.add_image(tag, grid, self.global_step)

        # only process first size to simplify visualization.
        visualize_size = val_outputs[0].shape
        val_outputs = [x[0] for x in val_outputs if x.shape == visualize_size]
        val_targets = [x[0] for x in val_targets if x.shape == visualize_size]

        num_logs = len(val_outputs)
        assert num_logs == len(val_targets)

        num_viz_images = 16
        step = (num_logs + num_viz_images - 1) // num_viz_images
        outputs, targets = [], []

        for i in range(0, num_logs, step):
            outputs.append(_normalize(val_outputs[i]))
            targets.append(_normalize(val_targets[i]))

        outputs = np.stack(outputs)
        targets = np.stack(targets)
        _save_image(targets, "Target")
        _save_image(outputs, "Reconstruction")
        _save_image(np.abs(targets - outputs), "Error")

    def _visualize_val(self, val_outputs, val_targets, val_inputs,val_refs):
        def _normalize(image):
            image = image[np.newaxis]
            image = image - image.min()
            return image / image.max()

        def _save_image(image, tag):
            grid = torchvision.utils.make_grid(torch.Tensor(image), nrow=4, pad_value=1)
            self.logger.experiment.add_image(tag, grid, self.global_step)

        # only process first size to simplify visualization.
        visualize_size = val_outputs[0].shape
        visualize_size_inputs = val_inputs[0].shape
        val_outputs = [x[0] for x in val_outputs if x.shape == visualize_size]
        val_targets = [x[0] for x in val_targets if x.shape == visualize_size]
        val_refs = [x[0] for x in val_refs if x.shape == visualize_size]
        val_inputs = [x[0] for x in val_inputs if x.shape == visualize_size_inputs]#????

        num_logs = len(val_outputs)
        num_logs = len(val_inputs)
        assert num_logs == len(val_targets)

        num_viz_images = 16
        step = (num_logs + num_viz_images - 1) // num_viz_images
        outputs, targets, inputs,refs = [], [], [], []

        for i in range(0, num_logs, step):
            outputs.append(_normalize(val_outputs[i]))
            targets.append(_normalize(val_targets[i]))
            inputs.append(_normalize(val_inputs[i]))
            refs.append(_normalize(val_refs[i]))

        outputs = np.stack(outputs)#(2, 1, 1, 256, 256)
        targets = np.stack(targets)#(2, 1, 1, 256, 256)
        inputs = np.stack(inputs)#(2, 1, 1, 256, 256)
        refs = np.stack(refs)
        
        _save_image(targets, "Target")
        _save_image(refs, "Reference")
        _save_image(outputs, "Reconstruction")
        _save_image(inputs, "Input")
        _save_image(np.abs(targets - outputs), "Error")

    # def validation_step_end(self, val_logs):
    #     device = val_logs["output"].device
    #     # device = val_logs["output_k"].device    #kspace branch
    #     # move to CPU to save GPU memory
    #     val_logs = {key: value.cpu() for key, value in val_logs.items()}
    #     val_logs["device"] = device

    #     return val_logs
    
    def validation_step_end(self, val_logs):
        # check inputs
        for k in (
            "batch_idx",
            "fname",
            "slice_num",
            "output",
            "target",
            "val_loss",
        ):
            if k not in val_logs.keys():
                raise RuntimeError(
                    f"Expected key {k} in dict returned by validation_step."
                )
        if val_logs["output"].ndim == 2:
            val_logs["output"] = val_logs["output"].unsqueeze(0)
        elif val_logs["output"].ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")
        if val_logs["target"].ndim == 2:
            val_logs["target"] = val_logs["target"].unsqueeze(0)
        elif val_logs["target"].ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")

        # pick a set of images to log if we don't have one already
        if self.val_log_indices is None:
            self.val_log_indices = list(
                np.random.permutation(len(self.trainer.val_dataloaders[0]))[
                    : self.num_log_images
                ]
            )

        # log images to tensorboard
        if isinstance(val_logs["batch_idx"], int):
            batch_indices = [val_logs["batch_idx"]]
        else:
            batch_indices = val_logs["batch_idx"]
        for i, batch_idx in enumerate(batch_indices):
            if batch_idx in self.val_log_indices:
                key = f"val_images_idx_{batch_idx}"
                target = val_logs["target"][i].unsqueeze(0)
                output = val_logs["output"][i].unsqueeze(0)
                error = torch.abs(target - output)
                output = output / output.max()
                target = target / target.max()
                error = error / error.max()
                self.log_image(f"{key}/target", target)
                self.log_image(f"{key}/reconstruction", output)
                self.log_image(f"{key}/error", error)

        # compute evaluation metrics
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()
        for i, fname in enumerate(val_logs["fname"]):
            slice_num = int(val_logs["slice_num"][i].cpu())
            maxval = val_logs["max_value"][i].cpu().numpy()
            output = val_logs["output"][i].cpu().numpy()
            target = val_logs["target"][i].cpu().numpy()

            mse_vals[fname][slice_num] = torch.tensor(
                evaluate.mse(target, output)
            ).view(1)
            target_norms[fname][slice_num] = torch.tensor(
                evaluate.mse(target, np.zeros_like(target))
            ).view(1)
            ssim_vals[fname][slice_num] = torch.tensor(
                evaluate.ssim(target[None, ...], output[None, ...], maxval=maxval)
            ).view(1)
            max_vals[fname] = maxval

        return {
            "val_loss": val_logs["val_loss"],
            "mse_vals": dict(mse_vals),
            "target_norms": dict(target_norms),
            "ssim_vals": dict(ssim_vals),
            "max_vals": max_vals,
        }

    def validation_epoch_end(self, val_logs):
        # aggregate losses
        losses = []
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()

        # use dict updates to handle duplicate slices
        for val_log in val_logs:
            losses.append(val_log["val_loss"].view(-1))

            for k in val_log["mse_vals"].keys():
                mse_vals[k].update(val_log["mse_vals"][k])
            for k in val_log["target_norms"].keys():
                target_norms[k].update(val_log["target_norms"][k])
            for k in val_log["ssim_vals"].keys():
                ssim_vals[k].update(val_log["ssim_vals"][k])
            for k in val_log["max_vals"]:
                max_vals[k] = val_log["max_vals"][k]

        # check to make sure we have all files in all metrics
        assert (
            mse_vals.keys()
            == target_norms.keys()
            == ssim_vals.keys()
            == max_vals.keys()
        )

        # apply means across image volumes
        metrics = {"nmse": 0, "ssim": 0, "psnr": 0}
        local_examples = 0
        for fname in mse_vals.keys():
            local_examples = local_examples + 1
            mse_val = torch.mean(
                torch.cat([v.view(-1) for _, v in mse_vals[fname].items()])
            )
            target_norm = torch.mean(
                torch.cat([v.view(-1) for _, v in target_norms[fname].items()])
            )
            metrics["nmse"] = metrics["nmse"] + mse_val / target_norm
            print()
            metrics["psnr"] = (
                metrics["psnr"]
                + 20
                * torch.log10(
                    torch.tensor(
                        max_vals[fname], dtype=mse_val.dtype, device=mse_val.device
                    )
                )
                - 10 * torch.log10(mse_val)
            )
            metrics["ssim"] = metrics["ssim"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()])
            )

        # reduce across ddp via sum
        metrics["nmse"] = self.NMSE(metrics["nmse"])
        metrics["ssim"] = self.SSIM(metrics["ssim"])
        metrics["psnr"] = self.PSNR(metrics["psnr"])
        tot_examples = self.TotExamples(torch.tensor(local_examples))
        val_loss = self.ValLoss(torch.sum(torch.cat(losses)))
        tot_slice_examples = self.TotSliceExamples(
            torch.tensor(len(losses), dtype=torch.float)
        )

        self.log("validation_loss", val_loss / tot_slice_examples, prog_bar=True)
        for metric, value in metrics.items():
            self.log(f"val_metrics/{metric}", value / tot_examples)
            
    def log_image(self, name, image):
        self.logger.experiment.add_image(name, image, global_step=self.global_step)
    def test_epoch_end(self, test_logs):
        outputs = defaultdict(dict)

        # use dicts for aggregation to handle duplicate slices in ddp mode
        for log in test_logs:
            for i, (fname, slice_num) in enumerate(zip(log["fname"], log["slice"])):
                outputs[fname][int(slice_num.cpu())] = log["output"][i]

        # stack all the slices for each file
        for fname in outputs:
            outputs[fname] = np.stack(
                [out for _, out in sorted(outputs[fname].items())]
            )

        # pull the default_root_dir if we have a trainer, otherwise save to cwd
        if hasattr(self, "trainer"):
            save_path = pathlib.Path(self.trainer.default_root_dir) / "reconstructions"
        else:
            save_path = pathlib.Path.cwd() / "reconstructions"
        self.print(f"Saving reconstructions to {save_path}")

        fastmri.save_reconstructions(outputs, save_path)

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser])

        # data arguments
        parser.add_argument(
            "--data_path", default=pathlib.Path("Datasets/"), type=pathlib.Path
        )
        parser.add_argument(
            "--challenge",
            choices=["singlecoil", "multicoil"],
            default="singlecoil",
            type=str,
        )
        parser.add_argument(
            "--sample_rate", default=1.0, type=float,
        )
        parser.add_argument(
            "--batch_size", default=1, type=int,
        )
        parser.add_argument(
            "--num_workers", default=4, type=float,
        )
        parser.add_argument(
            "--seed", default=42, type=int,
        )

        # logging params
        parser.add_argument(
            "--exp_dir", default=pathlib.Path("logs/"), type=pathlib.Path
        )
        parser.add_argument(
            "--exp_name", default="my_experiment", type=str,
        )
        parser.add_argument(
            "--test_split", default="test", type=str,
        )

        return parser
