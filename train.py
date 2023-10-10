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
    tb_loss = TB(workdir=parser.workdir, title='loss')
    tb_lr = TB(workdir=parser.workdir, title='lr')
    print_log(f'created tensorboard.')
    
    # Create Data
    dataset_train = AVData('process_data/npy_data/*.npy')
    distributedSampler = DistributedSampler(dataset_train, num_replicas=world_size, rank=rank, seed=18813173471)
    dataloader_train = DataLoader(dataset_train, num_workers=2, batch_size=32, sampler=distributedSampler, pin_memory=True, collate_fn=collater)
    print_log('created data.')

    # print([(k, v.shape) for k, v in dataset_train[0].items()])
    # [('ego_veh', torch.Size([5])), ('traffic_veh_list', torch.Size([11, 5])), ('ego_future_path', torch.Size([100, 3])), ('ego_action', torch.Size([]))]
    
    # Create Model
    d_model = 64
    nhead = 8
    num_layers = 6
    model = CarTrackTransformerEncoder(d_model=d_model, nhead=nhead, num_layers=num_layers).cuda()
    model = DDP(model, device_ids=[rank], output_device=rank, broadcast_buffers=False, find_unused_parameters=True)
    print_log(f'created model d_model={d_model} nhead={nhead} num_layer={num_layers}.')

    # Create Optimizer
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.0001)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80, 90, 95], gamma=0.1)
    print_log('created optimizer.')

    # Create Criterion
    # criterion = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    # criterion = torch.nn.L1Loss()
    # print_log('created criterion.')

    for epoch in range(1, 101):
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
            ego_action_data = batch['ego_action_data'].cuda()
            traffic_veh_key_padding = batch['traffic_veh_key_padding'].cuda()
            
            output = model(ego_veh_data, traffic_veh_data, ego_future_track_data, ego_action_data, traffic_veh_key_padding)
            
            candidates_BS = 32
            loss = torch.zeros(1,).type_as(ego_veh_data)
            for j in range(len(output)):
                expert_output = output[j]
                
                candidates_action_list = []
                while len(candidates_action_list) < candidates_BS:
                    candidates_action = np.random.uniform(low=-5, high=3, size=(1,))[0]
                    if np.abs(candidates_action - ego_action_data[j].detach().cpu().numpy()) < 0.2:
                        continue
                
                    candidates_action_list.append(candidates_action)
                
                # print(expert_action, candidates_action_list)
                
                candidates_action = torch.Tensor(candidates_action_list).type_as(ego_action_data)
                # print(candidates_action.shape)
                                
                single_ego_veh_data = expand_dim_0(candidates_BS, torch.unsqueeze(ego_veh_data[j], 0))
                single_traffic_veh_data = expand_dim_0(candidates_BS, torch.unsqueeze(traffic_veh_data[j], 0))
                single_ego_future_track_data = expand_dim_0(candidates_BS, torch.unsqueeze(ego_future_track_data[j], 0))
                single_traffic_veh_key_padding = expand_dim_0(candidates_BS, torch.unsqueeze(traffic_veh_key_padding[j], 0))
                                
                candidates_output = model(single_ego_veh_data, single_traffic_veh_data, single_ego_future_track_data, candidates_action, single_traffic_veh_key_padding)

                prob = torch.exp(expert_output) / (torch.sum(torch.exp(candidates_output)) + torch.exp(expert_output))

                loss += -torch.log(prob)

            loss = loss / candidates_BS / len(batch)
            avg_prob = torch.exp((-loss))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            dist.all_reduce(loss.div_(world_size))
            dist.all_reduce(avg_prob.div_(world_size))
            
            current_lr = optimizer.param_groups[0]['lr']
            tb_loss.write(loss.data, (epoch - 1) * len(dataloader_train) + i)
            tb_lr.write(current_lr, (epoch - 1) * len(dataloader_train) + i)
            
            if i % 10 == 0:
                print_log(f'epoch:{epoch} | iter:{i}/{len(dataloader_train)} | lr:{"%.4e"%current_lr} | loss:{"%.4f"%loss.data} | avg_prob: {"%.4f"%avg_prob}')
            
        lr_scheduler.step()
        
        save_model(parser.workdir, epoch, model)
 
    tb_loss.close()


def expand_dim_0(sz, tensor):
    dst_shape = tensor.shape[1:]
    tensor = tensor.expand(sz, *dst_shape)
    return tensor

    
if __name__ == '__main__':
    main()