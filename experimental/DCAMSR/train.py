import pathlib
import sys
from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything

import warnings
warnings.filterwarnings("ignore")
import os
# sys.path.append('../../')
import sys
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from fastmri.data.mri_data import fetch_dir
from experimental.DCAMSR.DCAMSR import SRModule 
# from MINet import SRModule 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
def main(args):
    """Main training routine."""
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    checkpoint_callback = ModelCheckpoint(monitor='psnr',mode='max',verbose=True)
    args.checkpoint_callback = checkpoint_callback
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # seed_everything(args.seed)
    torch.use_deterministic_algorithms(False)
    # torch.random.seed(1)
    model = SRModule(**vars(args))
    print('batch_size: ', args.batch_size)
    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    # trainer = Trainer(accelerator="gpu", gpus=args.num_gpusdevices=[0,1]).from_argparse_args(args)
    # trainer = Trainer(gpus=args.num_gpus).from_argparse_args(args)
    trainer = Trainer().from_argparse_args(args)

    # ------------------------
    # 3 START TRAINING OR TEST
    # ------------------------
    if args.mode == "train":
        trainer.fit(model)
    elif args.mode == "test":
        assert args.resume_from_checkpoint is not None
        trainer.test(model)
    else:
        raise ValueError(f"unrecognized mode {args.mode}")

def get_dataset(exp_name):
    print('path: ', pathlib.Path.cwd())
    
    path_config = pathlib.Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mriSR_dirs.yaml"))
    knee_path = fetch_dir("knee_path", path_config)
    knee_train_csv_path = fetch_dir("knee_train_csv_path", path_config)
    knee_val_csv_path = fetch_dir("knee_val_csv_path", path_config)
    knee_test_csv_path = fetch_dir("knee_val_csv_path", path_config)
    knee_challenge = 'singlecoil'
    
    # M4raw_path = fetch_dir("M4Raw_path", path_config)
    # M4RAW_train_csv_path = fetch_dir("M4Raw_train_csv_path", path_config)
    # M4RAW_val_csv_path = fetch_dir("M4Raw_val_csv_path", path_config)
    # M4RAW_test_csv_path = fetch_dir("M4Raw_test_csv_path", path_config)
    # M4RAW_challenge = 'multicoil'
    
    if exp_name == 'knee':
        return knee_path,knee_train_csv_path,knee_val_csv_path,knee_test_csv_path,knee_challenge
    # elif exp_name == 'M4Raw':
        # return M4raw_path,M4RAW_train_csv_path,M4RAW_val_csv_path,M4RAW_test_csv_path,M4RAW_challenge
        
def build_args():
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    dataset_name = 'knee'
    # dataset_name = 'M4Raw'
    data_path,train_csv,val_csv,test_csv,challenge = get_dataset(dataset_name)
    # print('path: ', pathlib.Path.cwd())
    path_config = pathlib.Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mriSR_dirs.yaml"))
    
    # path_config = pathlib.Path.cwd() / "mriSR_dirs.yaml"
    upscale = 4
    net_name = 'DCAMSR'
    try:
        logdir = fetch_dir("log_path", path_config) 
    except:
        os.makedirs(logdir)
    logdir = logdir / net_name / dataset_name / f"{upscale}x_SR"
    os.makedirs(logdir, exist_ok=True)

    parent_parser = ArgumentParser(add_help=False)

    parser = SRModule.add_model_specific_args(parent_parser)
    parser = Trainer.add_argparse_args(parser)

    num_gpus = 1
    backend = "ddp"
    batch_size = 4 if backend == "ddp" else num_gpus
        
    # module config
    config = dict(
        
        mask_type="random",
        center_fractions=[0.08],
        accelerations=[4],
        lr=2*1e-4,
        upscale=upscale,
        net_name=net_name,
        lr_step_size=40,
        lr_gamma=0.1,
        weight_decay=0.0,
        dataset_name=dataset_name,
        data_path=data_path,
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        challenge=challenge,
        exp_dir=logdir,
        exp_name="unet_demo",
        test_split="test",
        batch_size=batch_size,
    )
    config['batch_size'] = batch_size
    if num_gpus > 1:
        config["use_ddp"] = True
        
    config["num_gpus"] = num_gpus
    
    parser.set_defaults(**config)
    # trainer config
    parser.set_defaults(
        gpus=num_gpus,
        max_epochs=50,
        default_root_dir=logdir,
        replace_sampler_ddp=(backend != "ddp"),
        distributed_backend=backend,
        seed=42,
        deterministic=True,
    )
    
    parser.add_argument("--mode", default="train", type=str)
    args = parser.parse_args()
    

    return args


def run_cli():
    args = build_args()
    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(args)


if __name__ == "__main__":
    run_cli()
