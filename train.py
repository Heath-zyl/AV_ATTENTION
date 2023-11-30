import torch
from torch.nn.parallel import DistributedDataParallel as DDP
# from dataset import DistributedGroupSampler
from data.train_data import AVData
from data.collater import collater
from torch.utils.data import DataLoader
from model.carTrackTransformerEncoder import CarTrackTransformerEncoder
import torch.optim as optim
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
import os
import time
from master_ops import print_log, make_dirs, master_only, TB, save_model
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from inference import test
from torch.nn.utils import clip_grad_norm_


def parse(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a network.')
    parser.add_argument('--workdir', help='Dir to save log and checkpoint', default='workdir/')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser = parser.parse_args(args)

    return parser


def main():
    parser = parse()
    
    # Distribution
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend='nccl')
    assert rank == dist.get_rank()
    world_size = dist.get_world_size()  

    # Create Workdir
    time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    parser.workdir = os.path.join(parser.workdir, time_stamp)
    make_dirs(parser.workdir)
    print_log(f'pytorch version: {torch.__version__}')
    print_log(f'created workdir {parser.workdir}.')
    
    # Create Tensorboard
    if rank == 0:
        tb_loss = TB(workdir=parser.workdir, title='loss')
        tb_lr = TB(workdir=parser.workdir, title='lr')
        tb_rmse = TB(workdir=parser.workdir, title='rmse')
    print_log(f'created tensorboard.')
    
    # Create Data
    dataset_train = AVData(path='/face/ylzhang/tirl_data/3/*.npy', collision_file_path='/face/ylzhang/tirl_data/3/collision_res_for_data000.txt')
    distributedSampler = DistributedSampler(dataset_train, num_replicas=world_size, rank=rank, seed=18813173471)
    BS = 1
    dataloader_train = DataLoader(dataset_train, num_workers=1, batch_size=BS, sampler=distributedSampler, pin_memory=True, collate_fn=collater)
    print_log('created data.')

    # print([(k, v.shape) for k, v in dataset_train[0].items()])
    # [('ego_veh', torch.Size([5])), ('traffic_veh_list', torch.Size([11, 5])), ('ego_future_path', torch.Size([100, 3])), ('ego_action', torch.Size([]))]
    
    # Create Model
    d_model = 16
    nhead = 4
    num_layers = 1
    model = CarTrackTransformerEncoder(d_model=d_model, nhead=nhead, num_layers=num_layers).cuda()
    model = DDP(model, device_ids=[int(os.environ['LOCAL_RANK'])], output_device=[int(os.environ['LOCAL_RANK'])], broadcast_buffers=False, find_unused_parameters=True)
    print_log(f'created model d_model={d_model} nhead={nhead} num_layer={num_layers}.')

    # Create Optimizer
    # optimizer = optim.SGD(model.parameters(), lr=3e-2, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[600,], gamma=0.1)
    
    # optimizer = optim.AdamW(model.parameters(), lr=1e-2)
    # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[900, 950], gamma=0.1)
    
    print_log('created optimizer.')

    # Create Criterion
    criterion = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    # criterion = torch.nn.L1Loss()
    print_log('created criterion.')

    for epoch in range(1, 1000+1):
        if hasattr(dataloader_train.sampler, 'set_epoch'):
            print_log(f'setting epoch number: {epoch}')
            dataloader_train.sampler.set_epoch(epoch)
        
        model.train()
        if epoch == 1:
            print_log('about to start training...')
            
        for i, batch in enumerate(dataloader_train):
            
            ego_veh_data = batch['ego_veh_data'].cuda()
            traffic_veh_data = batch['traffic_veh_data'].cuda()
            ego_future_track_data = batch['ego_future_track_data'].cuda()
            ego_history_track_data = batch['ego_history_track_data'].cuda()
            ego_action_data = batch['ego_action_data'].cuda()
            traffic_veh_key_padding = batch['traffic_veh_key_padding'].cuda()
            
            output = model(ego_veh_data, ego_future_track_data, ego_history_track_data, traffic_veh_data, traffic_veh_key_padding)
            ego_action_data = torch.unsqueeze(ego_action_data, 1)
                        
            loss = criterion(output, ego_action_data)
            
            dist.all_reduce(loss.div_(BS*world_size))
            
            optimizer.zero_grad()
            loss.backward()
            
            clip_grad_norm_(model.parameters(), max_norm=100, norm_type=2)
            
            optimizer.step()

            # dist.all_reduce(ratio_sum.div_(world_size * BS))
            # dist.all_reduce(ratio_std.div_(world_size))
            
            current_lr = optimizer.param_groups[0]['lr']
            global_iter = (epoch - 1) * len(dataloader_train) + i
            if rank == 0 and global_iter % 50 == 0:
                tb_loss.write(loss.data, global_iter)
                tb_lr.write(current_lr, global_iter)
            
            if i % 10 == 0:
                print_log(f'epoch:{epoch} | iter:{i}/{len(dataloader_train)} | lr:{"%.4e"%current_lr} | loss:{"%.4f"%loss.data}')
        
            
        lr_scheduler.step()
        
        saved_path = save_model(parser.workdir, epoch, model)
        if rank == 0 and epoch % 50 == 0:
            rmse = test(d_model=d_model, nhead=nhead, num_layers=num_layers, model_path=saved_path)
            print_log(f'rmse: {rmse}')
            tb_rmse.write(rmse, epoch-1)
        
    if rank == 0:
        tb_loss.close()
        tb_lr.close()
        tb_rmse.close()

def expand_dim_0(sz, tensor):
    dst_shape = tensor.shape[1:]
    tensor = tensor.expand(sz, *dst_shape)
    return tensor

    
if __name__ == '__main__':
    main()