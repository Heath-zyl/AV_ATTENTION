from torch.utils.data import Dataset
import numpy as np
import torch
from glob import glob
from master_ops import print_log
import sys
import random


def transform(sample):
    new_sample = {}
    nor_x = 100
    nor_y = 100
    nor_vx = 30
    nor_vy = 30
    nor_yaw = np.pi

    ego_x = sample['ego_veh'][1]
    ego_y = sample['ego_veh'][2]
    ego_yaw = sample['ego_veh'][5]
    
    # 转换主车
    x = sample['ego_veh'][1]
    y = sample['ego_veh'][2]
    vx = sample['ego_veh'][3]
    vy = sample['ego_veh'][4]
    yaw = sample['ego_veh'][5]

    x_rel = x - ego_x
    y_rel = y - ego_y
    relative_x = x_rel * np.cos(ego_yaw) + y_rel * np.sin(ego_yaw)
    relative_y = -x_rel * np.sin(ego_yaw) + y_rel * np.cos(ego_yaw)
    relative_vx = vx * np.cos(ego_yaw) + vy * np.sin(ego_yaw)
    relative_vy = -vx * np.sin(ego_yaw) + vy * np.cos(ego_yaw)
    relative_yaw = yaw - ego_yaw

    x_min = 972.5
    x_max = 1089.4
    y_min = 965.3
    y_max = 1034.6
    new_sample['ego_veh'] = [(ego_x - x_min)/(x_max - x_min), (ego_y - y_min)/(y_max - y_min), relative_vx / nor_vx, 
                         relative_vy / nor_vy, ego_yaw/nor_yaw]


    # 转换交通车
    new_sample['traffic_veh_list'] = []
    for traffic_veh in sample['traffic_veh_list']:
        x = traffic_veh[1]
        y = traffic_veh[2]
        vx = traffic_veh[3]
        vy = traffic_veh[4]
        yaw = traffic_veh[5]

        x_rel = x - ego_x
        y_rel = y - ego_y
        relative_x = x_rel * np.cos(ego_yaw) + y_rel * np.sin(ego_yaw)
        relative_y = -x_rel * np.sin(ego_yaw) + y_rel * np.cos(ego_yaw)
        relative_vx = vx * np.cos(ego_yaw) + vy * np.sin(ego_yaw)
        relative_vy = -vx * np.sin(ego_yaw) + vy * np.cos(ego_yaw)
        relative_yaw = yaw - ego_yaw

        new_traffic_veh = [relative_x / nor_x, relative_y / nor_y, relative_vx / nor_vx, relative_vy / nor_vy,
                           relative_yaw / nor_yaw]

        new_sample['traffic_veh_list'].append(new_traffic_veh)


    # 转换未来轨迹
    new_sample['ego_future_path'] = []
    for points in sample['ego_future_path']:
        x = points[0]
        y = points[1]
        yaw = points[2]

        x_rel = x - ego_x
        y_rel = y - ego_y
        relative_x = x_rel * np.cos(ego_yaw) + y_rel * np.sin(ego_yaw)
        relative_y = -x_rel * np.sin(ego_yaw) + y_rel * np.cos(ego_yaw)
        relative_yaw = yaw - ego_yaw

        new_point = [relative_x / nor_x, relative_y / nor_y, relative_yaw / nor_yaw]

        new_sample['ego_future_path'].append(new_point)


    # 转换历史轨迹
    new_sample['ego_history_path'] = []
    for points in sample['ego_history_path']:
        x = points[0]
        y = points[1]
        yaw = points[2]

        x_rel = x - ego_x
        y_rel = y - ego_y
        relative_x = x_rel * np.cos(ego_yaw) + y_rel * np.sin(ego_yaw)
        relative_y = -x_rel * np.sin(ego_yaw) + y_rel * np.cos(ego_yaw)
        relative_yaw = yaw - ego_yaw

        new_point = [relative_x / nor_x, relative_y / nor_y, relative_yaw / nor_yaw]

        new_sample['ego_history_path'].append(new_point)


    # 转换加速度
    new_sample['ego_action'] = ((sample['ego_action'] + 1) / 4).astype(np.float32)

    
    # ToTensor
    for key in new_sample.keys():
        if isinstance(new_sample[key], (np.float64, np.float32)):
            new_sample[key] = torch.from_numpy(np.array(new_sample[key]))
        else:
            new_sample[key] = torch.Tensor(new_sample[key])


    # 主车未来轨迹， 取前100个轨迹点，如果不够100，则以最后一个实际轨迹点补充到100
    if new_sample['ego_future_path'].shape[0] >= 100:
        new_sample['ego_future_path'] = new_sample['ego_future_path'][:100]
    else:
        repeat_times = 100 - new_sample['ego_future_path'].shape[0]
        append_value = new_sample['ego_future_path'][-1][None, :].repeat(repeat_times, 1)
        new_sample['ego_future_path'] = torch.cat((new_sample['ego_future_path'], append_value), axis=0)
    
    
    return new_sample


class AVData(Dataset):
    def __init__(self, path, transform=transform, test_mode=False):
        files = glob(path)
        
        data_all = np.zeros(0)
        for file in files:
            data_temp = np.load(file, allow_pickle=True)
            data_all = np.concatenate((data_all, data_temp))
        
        # index: 662~693
        for i in range(len(data_all)-1, -1, -1):
            if data_all[i]['frame_id'] in list(range(1, 33)) and data_all[i]['ego_veh'][0] == 9:
                continue
            data_all = np.delete(data_all, i)
        
        self.data_all = data_all
        
        self.transform = transform
        self.test_mode = test_mode
    
    def __len__(self):
        return len(self.data_all)

    def __getitem__(self, idx):

        sample = self.data_all[idx]
        
        if self.test_mode:
            frame_id = sample['frame_id']
            ego_veh_id = sample['ego_veh'][0]
            # print(f'frame_id: {frame_id}')
            # print(f'ego_veh_id: {ego_veh_id}')
            
            vec_traffic_id_list = []
            # print('traffic veh id: ', end='')
            for traffic_vec in sample['traffic_veh_list']:
                # print(traffic_vec[0], end=',')
                vec_traffic_id_list.append(traffic_vec[0])
            # print()
            return self.transform(sample), frame_id, ego_veh_id, vec_traffic_id_list
            
        
        if len(sample['traffic_veh_list']) == 0:
            random_index = np.random.randint(0, self.__len__()+1)
            return self.__getitem__(random_index)
        
        return self.transform(sample)
    

if __name__ == '__main__':
    data = AVData('process_data/train_data_000.npy')
    
    for i in range(len(data)):
        for key, value in data[i].items():
            print(key, value.shape, end=', ')
        print()

    