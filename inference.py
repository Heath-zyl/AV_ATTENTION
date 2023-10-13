import torch
from data.train_data import AVData
from model.carTrackTransformerEncoder import CarTrackTransformerEncoder
from collections import OrderedDict
import sys
import numpy as np


def mapIdx(ts_num, traffic_id_list):
    return traffic_id_list[ts_num]


def main(data_idx):
    
    # Create Data
    dataset_train = AVData('process_data/npy_data/*.npy', test_mode=True)
    print(f'total num: {len(dataset_train)}')
    
    data_temp, frame_id, ego_veh_id, vec_traffic_id_list = dataset_train[data_idx]
    ego_veh, traffic_veh, ego_future_path, ego_action = data_temp['ego_veh'], data_temp['traffic_veh_list'], data_temp['ego_future_path'], data_temp['ego_action']
    
    print('************************************')
    print(f'frame_id: {frame_id}')
    print(f'ego_veg_id: {ego_veh_id}')
    print(f'vec_traffic_id_list({len(vec_traffic_id_list)}): {",".join(map(str, vec_traffic_id_list))}')
    
    print('************************************')
    ego_veh = torch.unsqueeze(ego_veh, 0)
    traffic_veh = torch.unsqueeze(traffic_veh, 0)
    ego_future_path = torch.unsqueeze(ego_future_path, 0)
    ego_action = torch.unsqueeze(ego_action, 0)
    
    d_model = 64
    nhead = 8
    num_layers = 6
    model = CarTrackTransformerEncoder(num_layers=num_layers, nhead=nhead, d_model=d_model)
    weights = torch.load('workdir/20231012_192806/epoch_12.pth', map_location='cpu')
    
    delete_module_weight = OrderedDict()
    for k, v in weights.items():
        delete_module_weight[k[7:]] = weights[k]
    model.load_state_dict(delete_module_weight, strict=True)
    model.eval()
    
    outs = model(ego_veh, traffic_veh, ego_future_path, ego_action)
    logit = torch.squeeze(outs[0], 1)
    print(logit.shape)
    weight_attention_list = outs[1]
    
    # print(logit)
    # print(weight_attention_list)
    # print([weight_attention.shape for weight_attention in weight_attention_list])
    
    # candidates
    candidates_action_list = np.arange(-5, 3.01, 0.01)
    candidates_action = torch.Tensor(candidates_action_list).type_as(ego_action)
    candidates_BS = 801
    
    single_ego_veh_data = expand_dim_0(candidates_BS, torch.unsqueeze(ego_veh[0], 0))         
    single_traffic_veh_data = expand_dim_0(candidates_BS, torch.unsqueeze(traffic_veh[0], 0))
    single_ego_future_track_data = expand_dim_0(candidates_BS, torch.unsqueeze(ego_future_path[0], 0))
    
    candidates_output = model(single_ego_veh_data, single_traffic_veh_data, single_ego_future_track_data, candidates_action)
    candidates_logit, _weights = torch.squeeze(candidates_output[0], 1), candidates_output[1]
    
    max_idx = torch.argmax(candidates_logit)
    
    print('='*20, 'res', '='*20)
    print(f'gt: {ego_action.data} | predict: {candidates_action_list[max_idx]}')
    
    diff = torch.abs(ego_action.data - candidates_action_list[max_idx]) * 8
    
    print(f'diff: {torch.abs(ego_action.data - candidates_action_list[max_idx])} | {torch.abs(ego_action.data - candidates_action_list[max_idx]) * 8}')

    return diff

def expand_dim_0(sz, tensor):
    dst_shape = tensor.shape[1:]
    tensor = tensor.expand(sz, *dst_shape)
    return tensor
    

if __name__ == '__main__':
    
    # main(int(sys.argv[1]))
    
    res = []
    for i in range(0, 1000, 502695):
        res = 