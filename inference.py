import torch
from data.train_data import AVData
from model.carTrackTransformerEncoder import CarTrackTransformerEncoder
from collections import OrderedDict
import sys


def mapIdx(ts_num, traffic_id_list):
    if ts_num == 0:
        return 'ego'
    else:
        return str(traffic_id_list[ts_num - 1])


def main():
    
    # Create Data
    dataset_train = AVData('process_data/npy_data/*.npy', test_mode=True)
    data_temp, frame_id, ego_veh_id, vec_traffic_id_list = dataset_train[int(sys.argv[1])]
    ego_veh, traffic_veh, ego_future_path = data_temp['ego_veh'], data_temp['traffic_veh_list'], data_temp['ego_future_path']
    
    print('************************************')
    print(f'frame_id: {frame_id}')
    print(f'ego_veg_id: {ego_veh_id}')
    print(f'vec_traffic_id_list: {",".join(map(str, vec_traffic_id_list))}')
    
    print('************************************')
    print(f'gt ego_action: {data_temp["ego_action"]}')
    ego_veh = torch.unsqueeze(ego_veh, 0)
    traffic_veh = torch.unsqueeze(traffic_veh, 0)
    ego_future_path = torch.unsqueeze(ego_future_path, 0)
    
    # distributedSampler = DistributedSampler(dataset_train, num_replicas=world_size, rank=rank, seed=18813173471)
    # dataloader_train = DataLoader(dataset_train, num_workers=2, batch_size=1, sampler=distributedSampler, pin_memory=True)
    # print_log('created data.')
    
    # Create Model
    # model = CarTrackTransformerEncoder(num_layers=6, nhead=8, d_model=64)
    # weights = torch.load('/home/ylzhang/AV_attention/workdir/20230922_160134/epoch_2.pth', map_location='cpu')
    
    d_model = 64
    nhead = 8
    num_layers = 6
    model = CarTrackTransformerEncoder(num_layers=num_layers, nhead=nhead, d_model=d_model)
    weights = torch.load('workdir/20230925_172524/epoch_4.pth', map_location='cpu')
    
    delete_module_weight = OrderedDict()
    for k, v in weights.items():
        delete_module_weight[k[7:]] = weights[k]
    model.load_state_dict(delete_module_weight, strict=True)
    model.eval()
    
    outs = model(ego_veh, traffic_veh, ego_future_path)
    print(f'predicted ego_action: {outs[0].data}')
    
    print(f'diff between prediction and gt: {outs[0].data - data_temp["ego_action"].data}')

    print('************************************')
    print(f'There are {len(outs[1])} groups of attention weights and each has shape of {outs[1][0].shape}')
    print('************************************')

    # for atten in outs[1]:
    #     atten_squeeze = torch.squeeze(atten)
    #     print(atten_squeeze.shape)
    #     print(atten_squeeze)
    #     print('====================================')
    
    layer_idx = -1
    print(f'The attention weights of layer: {layer_idx}:')
    atten_squeeze = torch.squeeze(outs[1][layer_idx])
    print('The attention weights of EGO:')
    cls_ebd_attention_weights = atten_squeeze[0]
    sort_idx = torch.argsort(cls_ebd_attention_weights, descending=True)
    for si in sort_idx:
        read_idx = mapIdx(si, vec_traffic_id_list)
        print(f'{read_idx}({"%.4f"%cls_ebd_attention_weights[si]})', end=', ')
    print()
    
    # print('************************************')
    # layer_idx = -1
    # print(f'The attention weights of layer: {layer_idx}:')
    # atten_squeeze = torch.squeeze(outs[1][layer_idx])
    # print('The attention weights of ego_veh:')
    # ego_attention_weights = atten_squeeze[1]
    # # print(ego_attention_weights)
    # sort_idx = torch.argsort(ego_attention_weights, descending=True)
    # for si in sort_idx:
    #     read_idx = mapIdx(si, vec_traffic_id_list)
    #     print(f'{read_idx}({"%.4f"%ego_attention_weights[si]})', end=', ')
    # print()
    
    
    # print('************************************')
    # layer_idx = -1
    # print(f'The attention future_track of layer: {layer_idx}:')
    # atten_squeeze = torch.squeeze(outs[1][layer_idx])
    # print('The attention weights of future_track:')
    # fp_attention_weights = atten_squeeze[-1]
    # # print(ego_attention_weights)
    # sort_idx = torch.argsort(fp_attention_weights, descending=True)
    # for si in sort_idx:
    #     read_idx = mapIdx(si, vec_traffic_id_list)
    #     print(f'{read_idx}({"%.4f"%fp_attention_weights[si]})', end=', ')
    # print()
    
    
if __name__ == '__main__':
    main()