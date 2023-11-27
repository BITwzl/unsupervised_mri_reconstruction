from net.net_parts import *
import torch.nn as nn


class MoDL(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(MoDL, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.layers = Dw(self.in_channels, self.out_channels)
        self.lam = nn.Parameter(torch.FloatTensor([0.05]).to(tuple(self.layers.parameters())[0].device), requires_grad=True)
        self.CG = ConjugatedGrad()

    def forward(self, under_img, csm, under_mask):
        x = under_img
        for i in range(self.num_layers):
            x = self.layers(x)
            x = under_img + self.lam * x
            x = self.CG(x, csm, under_mask, self.lam)
            x = torch.view_as_real(x).permute(0, 3, 1, 2).contiguous()
        x_final = x
        return x_final
    
class ParallelNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(ParallelNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.up_network = MoDL(self.in_channels, self.out_channels, self.num_layers)
        self.down_network = MoDL(self.in_channels, self.out_channels, self.num_layers)

    def forward(self, under_image_up, mask_up, under_image_down, mask_down, csm):

        output_up = self.up_network(under_image_up, csm, mask_up)
        output_down = self.down_network(under_image_down, csm, mask_down)

        return output_up, output_down
    
class ParallelNetwork3(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(ParallelNetwork3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.up_network = MoDL(self.in_channels, self.out_channels, self.num_layers)
        self.mid_network = MoDL(self.in_channels, self.out_channels, self.num_layers)
        self.down_network = MoDL(self.in_channels, self.out_channels, self.num_layers)

    def forward(self, under_image_up, mask_up, under_image_mid, mask_mid, under_image_down, mask_down, csm):

        output_up = self.up_network(under_image_up, csm, mask_up)
        output_mid = self.mid_network(under_image_mid, csm, mask_mid)
        output_down = self.down_network(under_image_down, csm, mask_down)

        return output_up, output_mid, output_down

class MultipleNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, num_models=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_models = num_models
        self.models=nn.ModuleList([MoDL(self.in_channels, self.out_channels, self.num_layers) for _ in range(self.num_models)])
        
    def forward(self, under_images, under_masks, csms):
        outputs=[]
        for under_image,under_mask,csm,model in zip(under_images,under_masks,csms,self.models):
            outputs.append(model(under_image,csm,under_mask))
        return outputs