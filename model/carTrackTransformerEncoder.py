import torch
from torch import nn
from .transformerEncoderLayer import TransformerEncoderLayer
from .transformerEncoder import TransformerEncoder
from master_ops import print_log
import torch.distributed as dist


class CarTrackTransformerEncoder(nn.Module):
    def __init__(self, d_model=64, nhead=8, num_layers=1):
        super(CarTrackTransformerEncoder, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.cls_ebd = nn.Parameter(torch.zeros(1, 1, d_model), requires_grad=True)
        
        self.fc_ego_veh = nn.Linear(5+1, d_model)
        self.fc_traffic_veh = nn.Linear(5+1, d_model)
        self.fc_ego_future_path = nn.Linear(300+1, d_model)
        
        self.ebd_fc = nn.Linear(d_model, 1)
        
        if dist.is_initialized():
            print_log(f'training model: {self.training}')
        
    def forward(self, ego_veh, traffic_veh, ego_future_path):
                
        ego_veh_pos = (torch.ones((ego_veh.shape[0], 1)) * (-0.2)).type_as(ego_veh)
        traffic_veh_pos = (torch.ones((traffic_veh.shape[0], traffic_veh.shape[1], 1)) * 0.1).type_as(traffic_veh)
        
        ego_veh = torch.cat((ego_veh, ego_veh_pos), dim=-1)
        traffic_veh = torch.cat((traffic_veh, traffic_veh_pos), dim=-1)
        
                
        ebd_ego_veh = self.fc_ego_veh(ego_veh)
        ebd_traffic_veh = self.fc_traffic_veh(traffic_veh)
        
        ego_future_path = ego_future_path.reshape(ego_future_path.shape[0],-1)
        
        ego_future_path_pos = (torch.ones((ego_future_path.shape[0], 1)) * 0.4).type_as(ego_future_path)
        ego_future_path = torch.cat((ego_future_path, ego_future_path_pos), dim=-1)
        
        ebd_ego_future_path = self.fc_ego_future_path(ego_future_path)
        
        ebd_ego_veh = torch.unsqueeze(ebd_ego_veh, 1)
        ebd_ego_future_path = torch.unsqueeze(ebd_ego_future_path, 1)
                
        input = torch.cat((ebd_ego_veh, ebd_traffic_veh, ebd_ego_future_path), axis=1) 
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
    
    
if __name__ == '__main__':
	model = CarTrackTransformerEncoder(num_layers=4)

	ego_veh = torch.randn(1, 5)
	traffic_veh = torch.randn(1, 8, 5)
	ego_future_path = torch.randn(1, 300)

	out = model(ego_veh, traffic_veh, ego_future_path)
	print(out.shape)