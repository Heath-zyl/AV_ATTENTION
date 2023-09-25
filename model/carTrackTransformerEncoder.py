import torch
from torch import nn
from .transformerEncoderLayer import TransformerEncoderLayer
from .transformerEncoder import TransformerEncoder
# from master_ops import print_log
import torch.distributed as dist


class CarTrackTransformerEncoder(nn.Module):
    def __init__(self, d_model=64, nhead=8, num_layers=1):
        super(CarTrackTransformerEncoder, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
                
        self.linear_ego_ft_1 = nn.Linear(300+1, 128)
        self.linear_ego_ft_2 = nn.Linear(128, 64)
        self.linear_ego_ft_3 = nn.Linear(64, 32)
        self.relu = nn.ReLU(inplace=True)
        
        self.linear_ego_veh_1 = nn.Linear(5+1, 16)
        self.linear_ego_veh_2 = nn.Linear(16, 32)
        
        self.linear_traffic_veh_1 = nn.Linear(5+1, 16)
        self.linear_traffic_veh_2 = nn.Linear(16, 32)
        self.linear_traffic_veh_3 = nn.Linear(32, 64)

        self.linear_final_fc1 = nn.Linear(64, 32)
        self.linear_final_fc2 = nn.Linear(32, 1)
        
    def forward(self, ego_veh_data, traffic_veh_data, ego_future_track_data, traffic_veh_key_padding=None):
        
        # process ego_future_track
        ego_future_track_pos = (torch.ones((ego_future_track_data.shape[0], 1)) * 0.4).type_as(ego_future_track_data)
        ego_future_track_data = ego_future_track_data.reshape(ego_future_track_data.shape[0],-1)
        ego_future_track_data = torch.cat((ego_future_track_data, ego_future_track_pos), dim=-1)
        ego_future_track_ebd = self.linear_ego_ft_3(self.relu(self.linear_ego_ft_2(self.relu(self.linear_ego_ft_1(ego_future_track_data)))))
        
        # process ego_veh
        ego_veh_pos = (torch.ones((ego_veh_data.shape[0], 1)) * (-0.2)).type_as(ego_veh_data)
        ego_veh_data = torch.cat((ego_veh_data, ego_veh_pos), dim=-1)
        ego_veh_ebd = self.linear_ego_veh_2(self.relu(self.linear_ego_veh_1(ego_veh_data)))
        
        # concat ego_future_track and ego_veh
        ego_ebd = torch.cat((ego_veh_ebd, ego_future_track_ebd), axis=1).unsqueeze(1)
        
        # process traffic_veh and corresponding mask
        traffic_veh_pos = (torch.ones(traffic_veh_data.shape[0], traffic_veh_data.shape[1], 1) * 0.1).type_as(traffic_veh_data)
        traffic_veh_data = torch.cat((traffic_veh_data, traffic_veh_pos), dim=-1)
        traffic_veh_ebd = self.linear_traffic_veh_3(self.relu(self.linear_traffic_veh_2(self.relu(self.linear_traffic_veh_1(traffic_veh_data)))))
        
        if traffic_veh_key_padding is not None:
            src_key_padding_mask = torch.zeros(traffic_veh_key_padding.shape[0], traffic_veh_key_padding.shape[1]+1).type_as(traffic_veh_key_padding)
            src_key_padding_mask[:, 1:] = traffic_veh_key_padding
            src_key_padding_mask = src_key_padding_mask.bool()
        else:
            print(traffic_veh_data.shape)
            src_key_padding_mask = torch.zeros(traffic_veh_data.shape[0], traffic_veh_data.shape[1]+1).bool()
        
        # concat all embdding to obtain final input to transformerEncoder
        cat_ebd = torch.cat((ego_ebd, traffic_veh_ebd), axis=1)
        cat_ebd = torch.transpose(cat_ebd, 0, 1)
        
        '''
        cat_ebd - src: :math:`(S, N, E)`.
        src_key_padding - src_key_padding_mask: :math:`(N, S)`.
        '''
                
        if self.training:
            memory = self.transformer_encoder(cat_ebd, src_key_padding_mask=src_key_padding_mask)            
            ego_fea = memory[0, ...]
            
            output = self.linear_final_fc2(self.relu(self.linear_final_fc1(ego_fea)))
            
            return output
        
        else:
            # print(input.shape) # S, BS, d_model 
            memory, attention_weights_list = self.transformer_encoder(cat_ebd, src_key_padding_mask=src_key_padding_mask)

            ego_fea = memory[0, ...]
            output = self.linear_final_fc2(self.relu(self.linear_final_fc1(ego_fea)))
        
            return output, attention_weights_list
        
        '''
        ego_veh_pos = (torch.ones((ego_veh.shape[0], 1)) * (-0.2)).type_as(ego_veh)
        traffic_veh_pos = (torch.ones((traffic_veh.shape[0], traffic_veh.shape[1], 1)) * 0.1).type_as(traffic_veh)
        
        ego_veh = torch.cat((ego_veh, ego_veh_pos), dim=-1)
        traffic_veh = torch.cat((traffic_veh, traffic_veh_pos), dim=-1)
        
                
        ebd_ego_veh = self.fc_ego_veh(ego_veh)
        ebd_traffic_veh = self.fc_traffic_veh(traffic_veh)
        
        ego_future_track = ego_future_track.reshape(ego_future_track.shape[0],-1)
        
        ego_future_track_pos = (torch.ones((ego_future_track.shape[0], 1)) * 0.4).type_as(ego_future_track)
        ego_future_track = torch.cat((ego_future_track, ego_future_track_pos), dim=-1)
        
        ebd_ego_future_track = self.fc_ego_future_track(ego_future_track)
        
        ebd_ego_veh = torch.unsqueeze(ebd_ego_veh, 1)
        ebd_ego_future_track = torch.unsqueeze(ebd_ego_future_track, 1)
                
        input = torch.cat((ebd_ego_veh, ebd_traffic_veh, ebd_ego_future_track), axis=1) 
        input = torch.cat((self.cls_ebd, input), axis=1)
        
        input = torch.transpose(input, 1, 0)
        
        if self.training:
            # print(input.shape) # S, BS, d_model 
            encoded_feature = self.transformer_encoder(input)
            
            assert encoded_feature.shape[0] == traffic_veh.shape[1] + 3
            
            # print(encoded_feature.shape) # S, BS, d_model 
            
            cls_ebd = encoded_feature[0,:,:]
            output = self.ebd_fc(cls_ebd)
        
            return output

        else:
            # print(input.shape) # S, BS, d_model 
            encoded_feature, attention_weights_list = self.transformer_encoder(input)
            
            assert encoded_feature.shape[0] == traffic_veh.shape[1] + 3
            
            # print(encoded_feature.shape) # S, BS, d_model 
            
            cls_ebd = encoded_feature[0,:,:]
            output = self.ebd_fc(cls_ebd)
        
            return output, attention_weights_list
            
        '''
    
    
if __name__ == '__main__':
	model = CarTrackTransformerEncoder(num_layers=4)
    
	ego_veh = torch.randn(32, 5)
	traffic_veh = torch.randn(32, 8, 5)
	ego_future_track = torch.randn(32, 300)

	out = model(ego_veh, traffic_veh, ego_future_track)
	print(out.shape)