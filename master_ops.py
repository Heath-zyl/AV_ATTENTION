import torch.distributed as dist
import functools
import os
import time
import sys
import torch
from torch.utils.tensorboard import SummaryWriter


class TB():
    def __init__(self, workdir, title):
        self.rank = dist.get_rank()
        if self.rank == 0:
            self.writer = SummaryWriter(os.path.join(workdir, 'tb_log'))
            self.title = title
    
    def write(self, y, x):
        if self.rank == 0:
            self.writer.add_scalar(self.title, y, x)
    
    def close(self,):
        if self.rank == 0:
            self.writer.close()


def master_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank = dist.get_rank()
        if rank == 0:
            return func(*args, **kwargs)
    return wrapper

LOG_PATH = None

@master_only
def make_dirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        raise NameError('Sth is wrong with workdir name.')
    
    log_path = os.path.join(dir_path, 'log.txt')
    f = open(log_path, 'w')
    f.close()
        
    global LOG_PATH
    LOG_PATH = log_path

@master_only
def print_log(*msg):
    time_stamp = time.strftime("%m%d-%H:%M:%S", time.localtime())
    message = f'{time_stamp} | ' + ''.join(msg)
    print(message)
    with open(LOG_PATH, 'a') as f:
        f.write(message + '\n')
        
        
@master_only
def save_model(dst_dir, epoch, model):
    dst_path = os.path.join(dst_dir, f'epoch_{epoch}.pth')
    print_log(f'save model to {dst_path}.')
    torch.save(model.state_dict(), dst_path)
    return dst_path