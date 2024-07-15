# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc

#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    ema_halflife_kimg   = 500,      # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = 0.05,     # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg      = 10000,    # Learning rate ramp-up duration.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,     # Start from the given training state, None = reset training state.
    resume_kimg         = 0,        # Start from the given training progress.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))

    # Construct network.
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)
    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.zeros([batch_gpu, net.label_dim], device=device)
            misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # training.loss.(VP|VE|EDM)Loss
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device])
    
    # load existing parameters into the checkpoint

    net = restore_song_checkpoint("/storage/ect/checkpoint_48.pth", net, device = device)

    
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    
    
    resume_pkl = "/storage/ect/celebahq_song.pkl"
    
    # dump checkpoint 
    with open(resume_pkl, "wb") as f:
        data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
        pickle.dump(data, f)

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data # conserve memory
    if False:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        del data # conserve memory

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    while True:

        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                images = images.to(device).to(torch.float32) / 127.5 - 1
                labels = labels.to(device)
                loss = loss_fn(net=ddp, images=images, labels=labels, augment_pipe=augment_pipe)
                training_stats.report('Loss/loss', loss)
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()

        # Update weights.
        for g in optimizer.param_groups:
            g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()

        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            del data # conserve memory

        # Save full dump of the training state.
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------




def restore_song_checkpoint(ckpt_dir, net, device):
  """function loads the checkpoint from the yang song model and copies the weights into the model in EDM format. 

  Args:
      ckpt_dir (str): path to the checkpoint
      net (_type_): Model in EDM format
      device (_type_): device to load the model on

  Returns:
      _type_: Model in EDM format
  """
  loaded_state = torch.load(ckpt_dir, map_location=device)
  shadow_ema_parameters = loaded_state['ema']["shadow_params"]
  src_tensors = loaded_state["model"]
  
  # omit the first parameter - 'module.all_modules.0.W' and copy the rest
  parameters_to_copy = [param for key, param in src_tensors.items() if "module.all_modules.0.W" not in key]
  
  
  print("dzien dobry")
  
  for s_param, param in zip(shadow_ema_parameters, parameters_to_copy):
      param.data.copy_(s_param.data)
  
  dst_tensors = net.state_dict()
  
  
  selected_dst_keys = list(dst_tensors.keys())
  selected_src_keys = list(src_tensors.keys())
  
  # remove some keys from the loaded state
  keys_to_remove = ['model.map_augment.weight'] + [og_string for og_string in selected_dst_keys if "resample" in og_string]
  for key in keys_to_remove:
      if key in selected_dst_keys:
          selected_dst_keys.remove(key)
  
  # match attention blocks
  attention_blocks_src = [og_string for og_string in src_tensors.keys() if "NIN" in og_string]
  attention_blocks_qkv_dst = [og_string for og_string in dst_tensors.keys() if "qkv" in og_string]
  attention_blocks_proj_dst = [og_string for og_string in dst_tensors.keys() if "proj" in og_string]
  
  attention_blocks_num = len(attention_blocks_qkv_dst) // 2 
  
  for attention_block in range(attention_blocks_num):
      # attention block in yang song are organized in order [NNI0.W, NNI0.b, NNI1.W, NNI1.b, NNI2.W, NNI2.b, NNI3.W, NNI3.b]
      # where NNI3 corresponds to projection the remaining linear blocks corresponds to qkv
      # match qkv
      qkv_id_src_weight = attention_block * 2
      qkv_id_src_bias = attention_block * 2 + 1
      
      NIN_weights = [attention_block * 8, attention_block * 8 + 2, attention_block * 8 + 4]
      NIN_bias = [attention_block * 8 + 1, attention_block * 8 + 3, attention_block * 8 + 5]
      
      NIN_keys_weight = [attention_blocks_src[src_key] for src_key in NIN_weights]
      NIN_keys_bias = [attention_blocks_src[src_key] for src_key in NIN_bias]
      
      num_heads = src_tensors[NIN_keys_weight[0]].shape[0]
      batch_size = src_tensors[NIN_keys_weight[0]].shape[1]
      
      NIN_tensor_weight = torch.cat([src_tensors[src_key].T.unsqueeze(2) for src_key in NIN_keys_weight], dim=2).permute(0, 2, 1).reshape(num_heads * 3, batch_size)
      NIN_tensor_bias = torch.stack([src_tensors[src_key] for src_key in NIN_keys_bias], dim=1).reshape(-1)
      
      
      try:
          dst_tensors[attention_blocks_qkv_dst[qkv_id_src_weight]].copy_(NIN_tensor_weight.unsqueeze(2).unsqueeze(2))
      except Exception as e:
          print(f"Error copying {NIN_keys_weight} to {attention_blocks_qkv_dst[qkv_id_src_weight]}: {e}")
      
      
      try:
          dst_tensors[attention_blocks_qkv_dst[qkv_id_src_bias]].copy_(NIN_tensor_bias)
      except Exception as e:
          print(f"Error copying {NIN_keys_bias} to {attention_blocks_qkv_dst[qkv_id_src_bias]}: {e}")
      
      
      # match projection
      projection_id_src_weight = attention_block * 8 + 6
      projection_id_src_bias = attention_block * 8 + 7
      
      projection_id_dst_weight = attention_block * 2
      projection_id_dst_bias = attention_block * 2 + 1
      
      
      
      dst_key = attention_blocks_proj_dst[projection_id_dst_weight]
      src_key = attention_blocks_src[projection_id_src_weight]                
      try:
          dst_tensors[dst_key].copy_(src_tensors[src_key].T.unsqueeze(2).unsqueeze(2))
      except Exception as e:
          print(f"Error copying {src_key} to {dst_key}: {e}")
          
      
      dst_key = attention_blocks_proj_dst[projection_id_dst_bias]
      src_key = attention_blocks_src[projection_id_src_bias]                
      try:
          dst_tensors[dst_key].copy_(src_tensors[src_key])
      except Exception as e:
          print(f"Error copying {src_key} to {dst_key}: {e}")
  
  for key in attention_blocks_src:
      selected_src_keys.remove(key)
  for key in attention_blocks_qkv_dst + attention_blocks_proj_dst:
      selected_dst_keys.remove(key)          
  
  assert len(selected_src_keys) == len(selected_dst_keys)
  # iterate over all the keys in the loaded state
  for src_key, dst_key in zip(selected_src_keys, selected_dst_keys):
      try:
          dst_tensors[dst_key].copy_(src_tensors[src_key])
      except Exception as e:
          raise ValueError(f"Error copying {src_key} to {dst_key}: {e}")
          
  return net
