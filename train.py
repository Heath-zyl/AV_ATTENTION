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
    print_log(f'created tensorboard.')
    
    # Create Data
    dataset_train = AVData('process_data/npy_data/*.npy')
    distributedSampler = DistributedSampler(dataset_train, num_replicas=world_size, rank=rank, seed=18813173471)
    dataloader_train = DataLoader(dataset_train, num_workers=2, batch_size=16, sampler=distributedSampler, pin_memory=True, collate_fn=collater)
    print_log('created data.')
    
    # Create Model
    d_model = 64
    nhead = 8
    num_layers = 6
    model = CarTrackTransformerEncoder(d_model=d_model, nhead=nhead, num_layers=num_layers).cuda()
    model = DDP(model, device_ids=[rank], output_device=rank, broadcast_buffers=False, find_unused_parameters=True)
    print_log(f'created model d_model={d_model} nhead={nhead} num_layer={num_layers}.')

    # Create Optimizer
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=0.0001)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 11], gamma=0.1)
    print_log('created optimizer.')

    # Create Criterion
    # criterion = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    criterion = torch.nn.L1Loss()
    print_log('created criterion.')

    for epoch in range(1, 13):
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
            
            # print(ego_action_data.shape)
            
            try:
                output = model(ego_veh_data, traffic_veh_data, ego_future_track_data, traffic_veh_key_padding)
            except Exception as e:
                print(e)
                # print(ego_veh.shape, traffic_veh.shape, ego_future_path.shape)

            output = torch.squeeze(output)
        
            loss = criterion(output, ego_action_data)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            dist.all_reduce(loss.div_(world_size))
            
            tb_loss.write(loss.data, (epoch - 1) * len(dataloader_train) + i)
            
            if i % 10 == 0:
                print_log(f'epoch:{epoch} | iter:{i}/{len(dataloader_train)} | loss:{"%.4f"%loss.data}')
            
        lr_scheduler.step()
        
        save_model(parser.workdir, epoch, model)
 
    tb_loss.close()
        
    
if __name__ == '__main__':
    main()