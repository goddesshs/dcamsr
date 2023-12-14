nohup: ignoring input
GPU available: True (cuda), used: False
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
You are using a CUDA device ('GeForce RTX 3090') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2]

  | Name             | Type                 | Params
----------------------------------------------------------
0 | NMSE             | DistributedMetricSum | 0     
1 | SSIM             | DistributedMetricSum | 0     
2 | PSNR             | DistributedMetricSum | 0     
3 | ValLoss          | DistributedMetricSum | 0     
4 | TotExamples      | DistributedMetricSum | 0     
5 | TotSliceExamples | DistributedMetricSum | 0     
6 | network          | DCAMSR               | 12.4 M
----------------------------------------------------------
12.4 M    Trainable params
0         Non-trainable params
12.4 M    Total params
49.533    Total estimated model params size (MB)
/home3/huangshan/superResolution/DCAMSR
/home3/huangshan/superResolution/DCAMSR
/home3/huangshan/superResolution/DCAMSR/experimental
['/home3/huangshan/superResolution/DCAMSR/experimental/DCAMSR', '/home3/huangshan/anaconda/envs/open-mmlabnew/lib/python310.zip', '/home3/huangshan/anaconda/envs/open-mmlabnew/lib/python3.10', '/home3/huangshan/anaconda/envs/open-mmlabnew/lib/python3.10/lib-dynload', '/home3/huangshan/anaconda/envs/open-mmlabnew/lib/python3.10/site-packages', '/home3/huangshan/open-mmlab/mmediting', '/home/huangshan/superResolution/multi_modal/MDT', '/home/huangshan/superResolution/multi_modal/improved-diffusion', '/tmp/tmpn4kxq93l', '/home3/huangshan/superResolution/DCAMSR', '/home3/huangshan/superResolution/DCAMSR', '/home3/huangshan/superResolution/DCAMSR/experimental']
path:  /home3/huangshan/superResolution/DCAMSR
Adjusting learning rate of group 0 to 2.0000e-04.
Sanity Checking: 0it [00:00, ?it/s]Sanity Checking:   0%|          | 0/2 [00:00<?, ?it/s]Sanity Checking DataLoader 0:   0%|          | 0/2 [00:00<?, ?it/s]Traceback (most recent call last):
  File "/home3/huangshan/superResolution/DCAMSR/experimental/DCAMSR/train.py", line 159, in <module>
    run_cli()
  File "/home3/huangshan/superResolution/DCAMSR/experimental/DCAMSR/train.py", line 155, in run_cli
    main(args)
  File "/home3/huangshan/superResolution/DCAMSR/experimental/DCAMSR/train.py", line 47, in main
    trainer.fit(model)
  File "/home3/huangshan/anaconda/envs/open-mmlabnew/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py", line 608, in fit
    call._call_and_handle_interrupt(
  File "/home3/huangshan/anaconda/envs/open-mmlabnew/lib/python3.10/site-packages/pytorch_lightning/trainer/call.py", line 38, in _call_and_handle_interrupt
    return trainer_fn(*args, **kwargs)
  File "/home3/huangshan/anaconda/envs/open-mmlabnew/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py", line 650, in _fit_impl
    self._run(model, ckpt_path=self.ckpt_path)
  File "/home3/huangshan/anaconda/envs/open-mmlabnew/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py", line 1103, in _run
    results = self._run_stage()
  File "/home3/huangshan/anaconda/envs/open-mmlabnew/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py", line 1182, in _run_stage
    self._run_train()
  File "/home3/huangshan/anaconda/envs/open-mmlabnew/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py", line 1195, in _run_train
    self._run_sanity_check()
  File "/home3/huangshan/anaconda/envs/open-mmlabnew/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py", line 1267, in _run_sanity_check
    val_loop.run()
  File "/home3/huangshan/anaconda/envs/open-mmlabnew/lib/python3.10/site-packages/pytorch_lightning/loops/loop.py", line 199, in run
    self.advance(*args, **kwargs)
  File "/home3/huangshan/anaconda/envs/open-mmlabnew/lib/python3.10/site-packages/pytorch_lightning/loops/dataloader/evaluation_loop.py", line 152, in advance
    dl_outputs = self.epoch_loop.run(self._data_fetcher, dl_max_batches, kwargs)
  File "/home3/huangshan/anaconda/envs/open-mmlabnew/lib/python3.10/site-packages/pytorch_lightning/loops/loop.py", line 199, in run
    self.advance(*args, **kwargs)
  File "/home3/huangshan/anaconda/envs/open-mmlabnew/lib/python3.10/site-packages/pytorch_lightning/loops/epoch/evaluation_epoch_loop.py", line 137, in advance
    output = self._evaluation_step(**kwargs)
  File "/home3/huangshan/anaconda/envs/open-mmlabnew/lib/python3.10/site-packages/pytorch_lightning/loops/epoch/evaluation_epoch_loop.py", line 234, in _evaluation_step
    output = self.trainer._call_strategy_hook(hook_name, *kwargs.values())
  File "/home3/huangshan/anaconda/envs/open-mmlabnew/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py", line 1485, in _call_strategy_hook
    output = fn(*args, **kwargs)
  File "/home3/huangshan/anaconda/envs/open-mmlabnew/lib/python3.10/site-packages/pytorch_lightning/strategies/strategy.py", line 390, in validation_step
    return self.model.validation_step(*args, **kwargs)
  File "/home3/huangshan/superResolution/DCAMSR/experimental/DCAMSR/DCAMSR.py", line 99, in validation_step
    pdfs = self(targetpd,imagepd,image)
  File "/home3/huangshan/anaconda/envs/open-mmlabnew/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1130, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home3/huangshan/superResolution/DCAMSR/experimental/DCAMSR/DCAMSR.py", line 80, in forward
    pdfs = self.network(LR.unsqueeze(1),Ref.unsqueeze(1),Ref_SR.unsqueeze(1))
  File "/home3/huangshan/anaconda/envs/open-mmlabnew/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1130, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home3/huangshan/superResolution/DCAMSR/fastmri/models/archs/DCAMSR_arch.py", line 342, in forward
    fea_ref_l = self.enc(ref)
  File "/home3/huangshan/anaconda/envs/open-mmlabnew/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1130, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home3/huangshan/superResolution/DCAMSR/fastmri/models/archs/DCAMSR_arch.py", line 243, in forward
    fea_L1 = self.blk_L1(self.act(self.conv_L1(x)))
  File "/home3/huangshan/anaconda/envs/open-mmlabnew/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1130, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home3/huangshan/anaconda/envs/open-mmlabnew/lib/python3.10/site-packages/torch/nn/modules/container.py", line 139, in forward
    input = module(input)
  File "/home3/huangshan/anaconda/envs/open-mmlabnew/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1130, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home3/huangshan/superResolution/DCAMSR/fastmri/models/archs/DCAMSR_arch.py", line 169, in forward
    out = self.conv2(self.act(self.conv1(x)))
  File "/home3/huangshan/anaconda/envs/open-mmlabnew/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1130, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home3/huangshan/anaconda/envs/open-mmlabnew/lib/python3.10/site-packages/torch/nn/modules/conv.py", line 457, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/home3/huangshan/anaconda/envs/open-mmlabnew/lib/python3.10/site-packages/torch/nn/modules/conv.py", line 453, in _conv_forward
    return F.conv2d(input, weight, bias, self.stride,
RuntimeError: Unable to find a valid cuDNN algorithm to run convolution
                                                                   