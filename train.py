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


def parse(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a network.')
    parser.add_argument('--workdir', help='Dir to save log and checkpoint', default='workdir/')
    parser.add_argument('--lbd', type=float, default=600., help='weight for loss2')
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
        tb_loss1 = TB(workdir=parser.workdir, title='loss1')
        tb_loss2 = TB(workdir=parser.workdir, title='loss2')
        tb_lr = TB(workdir=parser.workdir, title='lr')
        tb_avg_prob = TB(workdir=parser.workdir, title='avg_prob')
        tb_avg_safe_prob = TB(workdir=parser.workdir, title='avg_safe_prob')
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
    # criterion = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    # criterion = torch.nn.L1Loss()
    # print_log('created criterion.')

    candidates_BS = 81
    print_log(f'the num of candidates: {candidates_BS}')
    print_log(f'the weight of loss_2: {parser.lbd}')

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
            negative_action_data = batch['negative_action_data'] # .cuda()
            
            output = model(ego_veh_data, ego_future_track_data, ego_history_track_data, traffic_veh_data, ego_action_data, traffic_veh_key_padding)
               
            if candidates_BS == 801:
                candidates_action_list = (np.arange(-5, 3.01, 0.01) + 1) / 4
            elif candidates_BS == 81:
                candidates_action_list = (np.arange(-5, 3.1, 0.1) + 1) / 4
                
            candidates_action = torch.Tensor(candidates_action_list).type_as(ego_action_data)
            loss1 = torch.zeros(1,).type_as(ego_veh_data)
            loss2 = torch.zeros(1,).type_as(ego_veh_data)
            ratio_list = []
            prob_safe_list = []
            for j in range(len(output)):
                expert_output = output[j]

                single_ego_veh_data = expand_dim_0(candidates_BS, torch.unsqueeze(ego_veh_data[j], 0))         
                single_traffic_veh_data = expand_dim_0(candidates_BS, torch.unsqueeze(traffic_veh_data[j], 0))
                single_ego_future_track_data = expand_dim_0(candidates_BS, torch.unsqueeze(ego_future_track_data[j], 0))
                single_ego_history_track_data = expand_dim_0(candidates_BS, torch.unsqueeze(ego_history_track_data[j], 0))
                single_traffic_veh_key_padding = expand_dim_0(candidates_BS, torch.unsqueeze(traffic_veh_key_padding[j], 0))
                                
                candidates_output = model(single_ego_veh_data, single_ego_future_track_data, single_ego_history_track_data, single_traffic_veh_data, candidates_action, single_traffic_veh_key_padding)
                
                ratio = torch.exp(expert_output) / (torch.sum(torch.exp(candidates_output)))
                ratio_list.append(ratio)
                loss1 += -torch.log(ratio + 1e-9)
            
                # 负面演示的概率
                negative_action = negative_action_data[j].cuda()
                negative_BS = len(negative_action)
                
                if negative_BS == 0:
                    prob_safe = torch.tensor(1., requires_grad=True).type_as(candidates_output)
                    loss2 += -torch.log(prob_safe)
                    prob_safe_list.append(prob_safe)
                
                else:
                    
                    negative_ego_veh_data = expand_dim_0(negative_BS, torch.unsqueeze(ego_veh_data[j], 0))
                    negative_traffic_veh_data = expand_dim_0(negative_BS, torch.unsqueeze(traffic_veh_data[j], 0))
                    negative_ego_future_track_data = expand_dim_0(negative_BS, torch.unsqueeze(ego_future_track_data[j], 0))
                    negative_ego_history_track_data = expand_dim_0(negative_BS, torch.unsqueeze(ego_history_track_data[j], 0))
                    negative_traffic_veh_key_padding = expand_dim_0(negative_BS, torch.unsqueeze(traffic_veh_key_padding[j], 0))


                    negative_output = model(negative_ego_veh_data, negative_ego_future_track_data,
                                            negative_ego_history_track_data, negative_traffic_veh_data, negative_action,
                                            negative_traffic_veh_key_padding)

                    prob_safe = 1 - torch.sum(torch.exp(negative_output)) / (torch.sum(torch.exp(candidates_output)))

                    if torch.isnan(prob_safe).sum() > 0:
                        prob_safe = torch.tensor(1.).type_as(candidates_output)
                    
                    loss2 += -torch.log(prob_safe + 1e-9)
                    prob_safe_list.append(prob_safe)
            
            dist.all_reduce(loss1.div_(BS*world_size))
            dist.all_reduce(loss2.div_(BS*world_size))

            loss = loss1 + parser.lbd * loss2
            
            if torch.isnan(loss).sum() > 0:
                # input = torch.tensor(1.0, requires_grad=True).type_as(candidates_output)
                # loss = -torch.log(input)
                print('Loss NaN warning.')
                loss = torch.tensor(0., requires_grad=True).type_as(candidates_output)

            # dist.all_reduce(loss.div_(world_size))
            
            ratio_avg = torch.sum(torch.tensor(ratio_list)).cuda()
            dist.all_reduce(ratio_avg.div_(world_size * BS))
            
            prob_safe_avg = torch.sum(torch.tensor(prob_safe_list)).cuda()
            dist.all_reduce(prob_safe_avg.div_(world_size * BS))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # dist.all_reduce(ratio_sum.div_(world_size * BS))
            # dist.all_reduce(ratio_std.div_(world_size))
            
            current_lr = optimizer.param_groups[0]['lr']
            global_iter = (epoch - 1) * len(dataloader_train) + i
            if rank == 0 and global_iter % 50 == 0:
                tb_loss.write(loss.data, global_iter)
                tb_loss1.write(loss1.data, global_iter)
                tb_loss2.write(loss2.data, global_iter)
                tb_avg_prob.write(ratio_avg.data, global_iter)
                tb_avg_safe_prob.write(prob_safe_avg.data, global_iter)
                tb_lr.write(current_lr, global_iter)
            
            if i % 10 == 0:
                print_log(f'epoch:{epoch} | iter:{i}/{len(dataloader_train)} | lr:{"%.4e"%current_lr} | loss1:{"%.4f"%loss1.data} | loss2:{"%.4f"%loss2.data} | loss:{"%.4f"%loss.data} | ratio_avg: {"%.4f"%ratio_avg} | safe_avg: {"%.4f"%prob_safe_avg}')
        
            
        lr_scheduler.step()
        
        saved_path = save_model(parser.workdir, epoch, model)
        if rank == 0: # and epoch % 20 == 0:
            rmse = test(d_model=d_model, nhead=nhead, num_layers=num_layers, model_path=saved_path, candidates_num=candidates_BS)
            print_log(f'rmse: {rmse}')
            tb_rmse.write(rmse, epoch-1)
        
    if rank == 0:
        tb_loss.close()
        tb_lr.close()
        tb_avg_prob.close()
        tb_rmse.close()

def expand_dim_0(sz, tensor):
    dst_shape = tensor.shape[1:]
    tensor = tensor.expand(sz, *dst_shape)
    return tensor

    
if __name__ == '__main__':
    main()