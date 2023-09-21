import torch
from data.train_data import AVData
from model.carTrackTransformerEncoder import CarTrackTransformerEncoder
from collections import OrderedDict
import sys

sys

def main():
    
    # Create Data
    dataset_train = AVData('process_data/npy_data/*.npy')
    data_temp = dataset_train[int(sys.argv[1])]
    ego_veh, traffic_veh, ego_future_path = data_temp['ego_veh'], data_temp['traffic_veh_list'], data_temp['ego_future_path']
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
    
    d_model = 16
    nhead = 4
    num_layer = 4
    model = CarTrackTransformerEncoder(num_layers=num_layer, nhead=nhead, d_model=d_model)
    weights = torch.load('/workspace/workdir/20230923_113525/epoch_4.pth', map_location='cpu')
    
    delete_module_weight = OrderedDict()
    for k, v in weights.items():
        delete_module_weight[k[7:]] = weights[k]
    model.load_state_dict(delete_module_weight, strict=True)
    model.eval()
    print('created model.')
    print('************************************')
    
    # print(ego_veh.shape, traffic_veh.shape, ego_future_path.shape)
    outs = model(ego_veh, traffic_veh, ego_future_path)
    print(f'predicted ego_action: {outs[0].data}')
    
    print('************************************')
    
    print(f'diff between prediction and gt: {outs[0].data - data_temp["ego_action"].data}')

    print('************************************')

    print(f'There are {len(outs[1])} groups of attention weights and each has shape of {outs[1][0].shape}')

    for atten in outs[1]:
        atten_squeeze = torch.squeeze(atten)
        print(atten_squeeze.shape)
        print(atten_squeeze)
        print('====================================')
    
if __name__ == '__main__':
    main()