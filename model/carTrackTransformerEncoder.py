import torch
from torch import nn
from .transformerEncoderLayer import TransformerEncoderLayer
from .transformerEncoder import TransformerEncoder
# from master_ops import print_log
import torch.distributed as dist
from torch.nn.init import xavier_uniform_


class CarTrackTransformerEncoder(nn.Module):
    def __init__(self, d_model=64, nhead=8, num_layers=1):
        super(CarTrackTransformerEncoder, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
                
        self.linear_ego_ft_1 = nn.Linear(300, 256)
        self.linear_ego_ft_2 = nn.Linear(256, 128)
        self.linear_ego_ft_3 = nn.Linear(128, d_model // 2)
        self.relu = nn.ReLU(inplace=True)
        
        self.linear_ego_veh_1 = nn.Linear(5, 16)
        self.linear_ego_veh_2 = nn.Linear(16, d_model // 2)
        
        self.linear_traffic_veh_1 = nn.Linear(5, 16)
        self.linear_traffic_veh_2 = nn.Linear(16, 64)
        self.linear_traffic_veh_3 = nn.Linear(64, d_model)

        self._reset_parameters()
        
    def _reset_parameters(self,):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

        
    def forward(self, ego_veh_data, traffic_veh_data, ego_future_track_data, traffic_veh_key_padding=None):
        
        # process ego_future_track
        # ego_future_track_pos = (torch.ones((ego_future_track_data.shape[0], 1)) * 0.4).type_as(ego_future_track_data)
        ego_future_track_data = ego_future_track_data.reshape(ego_future_track_data.shape[0], -1)
        # ego_future_track_data = torch.cat((ego_future_track_data, ego_future_track_pos), dim=-1)
        ego_future_track_ebd = self.linear_ego_ft_3(self.relu(self.linear_ego_ft_2(self.relu(self.linear_ego_ft_1(ego_future_track_data)))))
        
        # process ego_veh
        # ego_veh_pos = (torch.ones((ego_veh_data.shape[0], 1)) * (-0.2)).type_as(ego_veh_data)
        # ego_veh_data = torch.cat((ego_veh_data, ego_veh_pos), dim=-1)
        ego_veh_ebd = self.linear_ego_veh_2(self.relu(self.linear_ego_veh_1(ego_veh_data)))
        
        # concat ego_future_track and ego_veh
        ego_ebd = torch.cat((ego_veh_ebd, ego_future_track_ebd), axis=1).unsqueeze(1) - 0.2
        
        # process traffic_veh and corresponding mask
        # traffic_veh_pos = (torch.ones(traffic_veh_data.shape[0], traffic_veh_data.shape[1], 1) * 0.1).type_as(traffic_veh_data)
        # traffic_veh_data = torch.cat((traffic_veh_data, traffic_veh_pos), dim=-1)
        traffic_veh_ebd = self.linear_traffic_veh_3(self.relu(self.linear_traffic_veh_2(self.relu(self.linear_traffic_veh_1(traffic_veh_data))))) + 0.2
                
        if traffic_veh_key_padding is not None:
            src_key_padding_mask = torch.zeros(traffic_veh_key_padding.shape[0], traffic_veh_key_padding.shape[1]+1).type_as(traffic_veh_key_padding)
            src_key_padding_mask[:, 1:] = traffic_veh_key_padding
            src_key_padding_mask = src_key_padding_mask.bool()
        else:
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
            
            output = torch.mean(ego_fea, axis=1)
            
            return output
        
        else:
            # print(input.shape) # S, BS, d_model 
            memory, attention_weights_list = self.transformer_encoder(cat_ebd, src_key_padding_mask=src_key_padding_mask)

            ego_fea = memory[0, ...]
            output = torch.mean(ego_fea, axis=1)
        
            return output, attention_weights_list
    
    
if __name__ == '__main__':
	model = CarTrackTransformerEncoder(num_layers=4)
    
	ego_veh = torch.randn(32, 5)
	traffic_veh = torch.randn(32, 8, 5)
	ego_future_track = torch.randn(32, 300)

	out = model(ego_veh, traffic_veh, ego_future_track)
	print(out.shape)