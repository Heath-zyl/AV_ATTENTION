import torch
import torch.distributed as dist


def collater(data):
    batch_size = len(data)
    
    ego_veh_data = torch.stack(tuple(d['ego_veh'] for d in data))
    ego_action_data = torch.stack(tuple(d['ego_action'] for d in data))
    ego_future_track_data = torch.stack(tuple(d['ego_future_path'] for d in data))    
    ego_history_track_data = torch.stack(tuple(d['ego_history_path'] for d in data))
    
    max_len_traffic_veh = max(tuple(d['traffic_veh_list'].shape[0] for d in data))  
    traffic_veh_data = torch.zeros(batch_size, max_len_traffic_veh, 5).type_as(data[0]['traffic_veh_list'])
    traffic_veh_key_padding = torch.zeros(traffic_veh_data.shape[:2]).type_as(data[0]['traffic_veh_list'])
    
    for idx_in_batch, d in enumerate(data):
        len_traffic_veh = d['traffic_veh_list'].shape[0]
        traffic_veh_data[idx_in_batch][:len_traffic_veh,:] = d['traffic_veh_list']
        traffic_veh_key_padding[idx_in_batch][len_traffic_veh:] = 1

    negative_action_data = [d['negative_action_data'] for d in data]

    sample = {}
    sample['ego_veh_data'] = ego_veh_data
    sample['ego_action_data'] = ego_action_data
    sample['ego_future_track_data'] = ego_future_track_data
    sample['ego_history_track_data'] = ego_history_track_data
    sample['traffic_veh_data'] = traffic_veh_data
    sample['traffic_veh_key_padding'] = traffic_veh_key_padding

    sample['negative_action_data'] = negative_action_data

    return sample