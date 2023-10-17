import torch
from torch import nn
from .transformerEncoderLayer import TransformerEncoderLayer
from .transformerEncoder import TransformerEncoder
# from master_ops import print_log
import torch.distributed as dist
from torch.nn.init import xavier_uniform_
from torch.nn import functional as F


class CarTrackTransformerEncoder(nn.Module):
    def __init__(self, d_model=64, nhead=8, num_layers=1):
        super(CarTrackTransformerEncoder, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.cls_ebd = nn.Parameter(torch.zeros(1, 1, d_model), requires_grad=True)
        
        self.linear_ego_ft_1 = nn.Linear(300, 256)
        self.linear_ego_ft_2 = nn.Linear(256, 128)
        self.linear_ego_ft_3 = nn.Linear(128, d_model)
        
        self.linear_ego_veh_1 = nn.Linear(5, 16)
        self.linear_ego_veh_2 = nn.Linear(16, d_model)
        
        self.linear_traffic_veh = nn.Linear(5, d_model)

        # self.linear_action_1 = nn.Linear(1, d_model//2)
        # self.linear_action_2 = nn.Linear(d_model//2, d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, 100),
            nn.Sigmoid(),
            nn.Linear(100, 1),
        )

        self.relu = nn.ReLU(inplace=True)

        self._reset_parameters()
        
    def _reset_parameters(self,):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

        
    def forward(self, ego_veh_data, traffic_veh_data, ego_future_track_data, ego_action_data, traffic_veh_key_padding=None):
        
        # Process ego_future_track
        ego_future_track_data = ego_future_track_data.reshape(ego_future_track_data.shape[0], -1)
        ego_future_track_ebd = self.linear_ego_ft_3(self.relu(self.linear_ego_ft_2(self.relu(self.linear_ego_ft_1(ego_future_track_data)))))
                
        # Process ego_veh
        ego_veh_ebd = self.linear_ego_veh_2(self.relu(self.linear_ego_veh_1(ego_veh_data)))
        
        # Process traffic_veh and corresponding mask
        traffic_veh_ebd = self.linear_traffic_veh(traffic_veh_data)
                
        if traffic_veh_key_padding is not None:
            src_key_padding_mask = torch.zeros(traffic_veh_key_padding.shape[0], traffic_veh_key_padding.shape[1]+3).type_as(traffic_veh_key_padding)
            src_key_padding_mask[:, 3:] = traffic_veh_key_padding
            src_key_padding_mask = src_key_padding_mask.bool()
        else:
            src_key_padding_mask = torch.zeros(traffic_veh_data.shape[0], traffic_veh_data.shape[1] + 3).type_as(ego_veh_data).bool()
        
        cls_ebd = self.cls_ebd.expand(ego_veh_ebd.shape[0], 1, self.cls_ebd.shape[-1])
        ego_future_track_ebd = torch.unsqueeze(ego_future_track_ebd, dim=1)
        ego_veh_ebd = torch.unsqueeze(ego_veh_ebd, dim=1)
        
        # print(cls_ebd.shape, ego_future_track_ebd.shape, ego_veh_ebd.shape, traffic_veh_ebd.shape, src_key_padding_mask.shape)
        # torch.Size([2, 1, 64]) torch.Size([2, 1, 64]) torch.Size([2, 1, 64]) torch.Size([2, 14, 64]) torch.Size([2, 17])
        cat_ebd = torch.cat((cls_ebd, ego_future_track_ebd, ego_veh_ebd, traffic_veh_ebd), axis=1)
        # print(cat_ebd.shape)
        # torch.Size([2, 17, 64])
        cat_ebd = torch.transpose(cat_ebd, 0, 1)
        # print(cat_ebd.shape)
        # torch.Size([17, 2, 64])
        
        if self.training:
            memory = self.transformer_encoder(cat_ebd, src_key_padding_mask=src_key_padding_mask)
            cls_out = memory[0]
            ego_action_data = torch.unsqueeze(ego_action_data, dim=1)
            # mlp_input = torch.cat((cls_out, ego_action_data), dim=1)
            mlp_input = cls_out * ego_action_data
            out = self.mlp(mlp_input)
            return out
            
        else:
            memory, attentions_weights_list = self.transformer_encoder(cat_ebd, src_key_padding_mask=src_key_padding_mask)
            cls_out = memory[0]
            ego_action_data = torch.unsqueeze(ego_action_data, dim=1)
            # mlp_input = torch.cat((cls_out, ego_action_data), dim=1)
            mlp_input = cls_out * ego_action_data
            out = self.mlp(mlp_input)
            return out, attentions_weights_list
            
        
    
if __name__ == '__main__':
	model = CarTrackTransformerEncoder(num_layers=4)
    
	ego_veh = torch.randn(32, 5)
	traffic_veh = torch.randn(32, 8, 5)
	ego_future_track = torch.randn(32, 300)

	out = model(ego_veh, traffic_veh, ego_future_track)
	print(out.shape)