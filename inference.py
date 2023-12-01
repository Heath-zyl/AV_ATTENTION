import torch
from data.train_data import AVData
from model.carTrackTransformerEncoder import CarTrackTransformerEncoder
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
import sys
torch.set_printoptions(16)

# MODEL_PATH = '/face/ylzhang/tirl_workdir/v5.0/20231116_151506/epoch_999.pth'
# MODEL_PATH = '/face/ylzhang/tirl_workdir/v6.0/20231123_104956/epoch_119.pth'
MODEL_PATH = '/face/ylzhang/tirl_workdir/action_clone/v1.0/20231129_113456/epoch_1.pth'

# DATA_PATH = '/face/ylzhang/tirl_data/test/*.npy'
DATA_PATH = '/face/ylzhang/tirl_data/3/TIRL_train_data_000.npy'

D_MODEL = 16
NHEAD = 4
NUM_LAYERS = 1


def mapIdx(ts_num, traffic_id_list):
    return traffic_id_list[ts_num]

def expand_dim_0(sz, tensor):
    dst_shape = tensor.shape[1:]
    tensor = tensor.expand(sz, *dst_shape)
    return tensor


def main(data_idx):
    
    # Create Data
    dataset_train = AVData(DATA_PATH, test_mode=True)
    print(f'total num: {len(dataset_train)}')
    
    data_temp, frame_id, ego_veh_id, vec_traffic_id_list = dataset_train[data_idx]
    # ego_veh, traffic_veh, ego_future_path, ego_action = data_temp['ego_veh'], data_temp['traffic_veh_list'], data_temp['ego_future_path'], data_temp['ego_action']
    ego_veh, traffic_veh, ego_future_path, ego_history_path, ego_action = data_temp['ego_veh'], data_temp['traffic_veh_list'], data_temp['ego_future_path'], data_temp['ego_history_path'], data_temp['ego_action']

    
    print('************************************')
    print(f'frame_id: {frame_id}')
    print(f'ego_veg_id: {ego_veh_id}')
    print(f'vec_traffic_id_list({len(vec_traffic_id_list)}): {",".join(map(str, vec_traffic_id_list))}')
    
    print('************************************')
    ego_veh = torch.unsqueeze(ego_veh, 0)
    traffic_veh = torch.unsqueeze(traffic_veh, 0)
    ego_future_path = torch.unsqueeze(ego_future_path, 0)
    ego_action = torch.unsqueeze(ego_action, 0)
    
    d_model = D_MODEL
    nhead = NHEAD
    num_layers = NUM_LAYERS
    model = CarTrackTransformerEncoder(num_layers=num_layers, nhead=nhead, d_model=d_model)
    weights = torch.load(MODEL_PATH, map_location='cpu')
    
    delete_module_weight = OrderedDict()
    for k, v in weights.items():
        delete_module_weight[k[7:]] = weights[k]
    model.load_state_dict(delete_module_weight, strict=True)
    model.eval()
    
    outs = model(ego_veh, traffic_veh, ego_future_path, ego_action)
    logit = torch.squeeze(outs[0], 1)
    
    trans_layer = 1
    # outs[1] # (num_layers, 1, num_traffic+3)
    weight_attention_list = outs[1][trans_layer][0]
    
    # print(logit)
    # print(weight_attention_list[0])
    weight_attention = weight_attention_list[0]
    # print([torch.unique(weight_attention.data) for weight_attention in weight_attention_list])
    
    sort_idx = torch.argsort(weight_attention, descending=True)
    for si in sort_idx:
        if si == 0:
            print(f'cls({weight_attention[si]})', end=', ')
        elif si == 1:
            print(f'track({weight_attention[si]})', end=',' )
        elif si == 2:
            print(f'ego({weight_attention[si]})', end=', ')
        else:
            print(f'{vec_traffic_id_list[si-3]}({weight_attention[si]}))', end=', ')
    print()
    
    # candidates
    candidates_action_list = (np.arange(-5, 3.01, 0.01) + 1) / 4
    candidates_action = torch.Tensor(candidates_action_list).type_as(ego_action)
    candidates_BS = 801
    
    single_ego_veh_data = expand_dim_0(candidates_BS, torch.unsqueeze(ego_veh[0], 0))         
    single_traffic_veh_data = expand_dim_0(candidates_BS, torch.unsqueeze(traffic_veh[0], 0))
    single_ego_future_track_data = expand_dim_0(candidates_BS, torch.unsqueeze(ego_future_path[0], 0))
    single_ego_history_track_data = expand_dim_0(candidates_BS, torch.unsqueeze(ego_history_path)).cuda()
    
    # candidates_output = model(single_ego_veh_data, single_traffic_veh_data, single_ego_future_track_data, candidates_action)
    candidates_output = model(single_ego_veh_data, single_ego_future_track_data, single_ego_history_track_data, single_traffic_veh_data, candidates_action)
    candidates_logit, _weights = torch.squeeze(candidates_output[0], 1), candidates_output[1]
    
    max_idx = torch.argmax(candidates_logit)
    
    print('='*20, 'res', '='*20)
    print(f'gt: {ego_action.data} | predict: {candidates_action_list[max_idx]}')
    
    # diff = torch.abs(ego_action.data - candidates_action_list[max_idx])
    
    print(f'diff: {torch.abs(ego_action.data - candidates_action_list[max_idx])} | {torch.abs(ego_action.data - candidates_action_list[max_idx]) * 4}')



def test(d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, model_path=MODEL_PATH, data_path=DATA_PATH):
    
    # Create model
    model = CarTrackTransformerEncoder(num_layers=num_layers, nhead=nhead, d_model=d_model)
    weights = torch.load(model_path, map_location='cpu')
    
    delete_module_weight = OrderedDict()
    for k, v in weights.items():
        delete_module_weight[k[7:]] = weights[k]
    model.load_state_dict(delete_module_weight, strict=True)
    model.eval()
    model = model.cuda()
    
    # Create Data
    dataset_train = AVData(data_path, collision_file_path='/face/ylzhang/tirl_data/3/collision_res_for_data000.txt', test_mode=True)
    
    res, cnt = 0, 0
    for data_idx in tqdm(range(0, len(dataset_train)//5)):
        
        data_temp, frame_id, ego_veh_id, vec_traffic_id_list = dataset_train[data_idx]
        ego_veh, traffic_veh, ego_future_path, ego_history_path, ego_action = data_temp['ego_veh'], data_temp['traffic_veh_list'], data_temp['ego_future_path'], data_temp['ego_history_path'], data_temp['ego_action']
                
        if traffic_veh.numel() == 0:
            print(f'traffic_veh.numel() == 0: {data_idx}')
            continue
        
        ego_veh = torch.unsqueeze(ego_veh, 0).cuda()
        traffic_veh = torch.unsqueeze(traffic_veh, 0).cuda()
        ego_future_path = torch.unsqueeze(ego_future_path, 0).cuda()
        ego_history_path = torch.unsqueeze(ego_history_path, 0).cuda()
        ego_action = torch.unsqueeze(ego_action, 0).cuda()
        
        with torch.no_grad():
            candidates_output = model(ego_veh, ego_future_path, ego_history_path, traffic_veh)
        
        candidates_logit, _weights = torch.squeeze(candidates_output[0], 1), candidates_output[1]
        
        diff = torch.abs(ego_action.data - candidates_logit) * 4
        res += diff**2
        cnt += 1

    rmse = (res / cnt) ** 0.5
    # print(f'final res: {rmse}')
    return rmse


if __name__ == '__main__':
    
    # main(int(sys.argv[1]))
    
    print(f'final res: {test()}')