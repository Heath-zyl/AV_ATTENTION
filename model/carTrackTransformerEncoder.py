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

        self.mlp_ego = nn.Sequential(
            nn.Linear(5, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, d_model)
        )

        self.mlp_future = nn.Sequential(
            nn.Linear(300, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.mlp_history = nn.Sequential(
            nn.Linear(30, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, d_model)
        )

        self.mlp_traff = nn.Sequential(
            nn.Linear(5, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, d_model)
        )

        self.mlp_tail = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.Sigmoid(),
            nn.Linear(d_model//2, 1),
        )

        self._reset_parameters()
        
    def _reset_parameters(self,):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

        
    def forward(self, ego_veh_data, ego_future_track_data, ego_history_track_data, traffic_veh_data, ego_action_data, traffic_veh_key_padding=None):

        # mlp to ego
        ego_ebd = self.mlp_ego(ego_veh_data)
        
        # mlp to future_track
        ego_future_track_data = ego_future_track_data.reshape(ego_future_track_data.shape[0], -1)
        future_ebd = self.mlp_future(ego_future_track_data)
        
        # mlp to history_track
        ego_history_track_data = ego_history_track_data.reshape(ego_history_track_data.shape[0], -1)
        history_ebd = self.mlp_history(ego_history_track_data)
        
        # Process traffic_veh and corresponding mask
        traffic_ebd = self.mlp_traff(traffic_veh_data)
        
        # Input Ebd
        cls_ebd = self.cls_ebd.expand(ego_ebd.shape[0], 1, self.cls_ebd.shape[-1])
        ego_ebd = torch.unsqueeze(ego_ebd, dim=1)
        future_ebd = torch.unsqueeze(future_ebd, dim=1)
        history_ebd = torch.unsqueeze(history_ebd, dim=1)
        input_ebd = torch.cat([cls_ebd, ego_ebd, future_ebd, history_ebd, traffic_ebd], dim=1).transpose(1, 0)
        
        # print(input_ebd.shape) # torch.Size([15, 1, 16]) # torch.Size([15, 801, 16])
        # print(traffic_ebd.shape)

        # pos = [0.] + [0.05] + [0.1] + [0.15] + [0.2 for i in range(traffic_ebd.shape[1])]
        # pos = torch.tensor(pos).unsqueeze(1).unsqueeze(1).expand(input_ebd.shape).type_as(input_ebd)
        # input_ebd = input_ebd + pos
        
        if traffic_veh_key_padding is not None:
            src_key_padding_mask = torch.zeros(traffic_veh_key_padding.shape[0], traffic_veh_key_padding.shape[1]+4).type_as(traffic_veh_key_padding)
            src_key_padding_mask[:, 4:] = traffic_veh_key_padding
            src_key_padding_mask = src_key_padding_mask.bool()
        else:
            src_key_padding_mask = torch.zeros(traffic_veh_data.shape[0], traffic_veh_data.shape[1]+4).type_as(ego_veh_data).bool()
        
        if self.training:
            memory = self.transformer_encoder(input_ebd, src_key_padding_mask=src_key_padding_mask)
            cls_out = memory[0]
            ego_action_data = torch.unsqueeze(ego_action_data, dim=1)
            mlp_input = cls_out * ego_action_data
            out = self.mlp_tail(mlp_input)
            return out
            
        else:
            memory, attentions_weights_list = self.transformer_encoder(input_ebd, src_key_padding_mask=src_key_padding_mask)
            cls_out = memory[0]
            ego_action_data = torch.unsqueeze(ego_action_data, dim=1)
            mlp_input = cls_out * ego_action_data
            out = self.mlp_tail(mlp_input)
            return out, attentions_weights_list
            
        
    
if __name__ == '__main__':
	model = CarTrackTransformerEncoder(num_layers=4)
    
	ego_veh = torch.randn(32, 5)
	traffic_veh = torch.randn(32, 8, 5)
	ego_future_track = torch.randn(32, 300)

	out = model(ego_veh, traffic_veh, ego_future_track)
	print(out.shape)