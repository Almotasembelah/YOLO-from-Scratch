import math
import torch.nn as nn
import torch
from tal import make_anchors, dist2bbox

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False,
                              **kwargs)
        self.norm = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()
    
    def forward(self, x):
        return self.silu(self.norm(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, in_out_channels, shortcut=True, **kwargs):
        super(Bottleneck, self).__init__()
        self.conv1 = ConvBlock(in_out_channels, 
                               in_out_channels, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1, **kwargs)
        
        self.conv2 = ConvBlock(in_out_channels, 
                               in_out_channels, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1, **kwargs)
        self.shortcut = shortcut
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut:
            return x + y
        else:
            return y

class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, **kwargs):
        super(C2f, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, **kwargs)
        self.conv2 = ConvBlock(n*out_channels//2 + out_channels, out_channels, kernel_size=1, stride=1, **kwargs)
        self.bottlenecks = nn.ModuleList([Bottleneck(out_channels//2, shortcut) for _ in range(n)])
    
    def forward(self, x):
        x = self.conv1(x)
        x1, x2 = x.chunk(2, dim=1)
        y = [x1, x2]
        for model in self.bottlenecks:
            x1 = model(x1)
            y.append(x1)
        y = torch.cat(y, dim=1)
        y = self.conv2(y)
        return y
    
class SPPFModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, block=ConvBlock):
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = block(in_channels, c_, 1, 1)
        self.cv2 = block(c_ * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))

class Concat(nn.Module):
    def __init__(self, dim=1):
        super(Concat, self).__init__()
        self.dim = dim
    def forward(self, x1, x2):
        return torch.cat([x1, x2], dim=self.dim)
    
class Upsample(nn.Module):
    def __init__(self, scale_factor=2, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

def get_block(name, **kwargs):
    if name == 'conv':
        return ConvBlock(**kwargs)
    elif name == 'c2f':
        return C2f(**kwargs)
    elif name == 'sppf':
        return SPPFModule(**kwargs)
    elif name == 'c':
        return Concat(**kwargs)
    elif name == 'u':
        return Upsample(**kwargs)
    elif name == 'conv2d':
        return nn.Conv2d(**kwargs)
    elif name == 'relu':
        return nn.ReLU(**kwargs)
    else:
        raise NotImplemented('Not Implemented yet!')

def make_layer(config):
    return nn.Sequential(*[get_block(name, **parameters) for name, parameters in config])

class Backbone(nn.Module):
    def __init__(self, config):
        super(Backbone, self).__init__()
        self.p1_net, self.p2_net, self.p3_net = self._make_layers(config)

    def forward(self, x):
        p1 = self.p1_net(x)
        p2 = self.p2_net(p1)
        p3 = self.p3_net(p2)
        return p1, p2, p3
    
    def _make_layers(self, config):
        layers = [make_layer(sub_config) for sub_config in config]
        return layers

class Neck(nn.Module):
    def __init__(self, config):
        super(Neck, self).__init__()
        # PAN
        self.p2_path_down, self.p1_path_down = self._FPN(config[0]) # FPN Down
        self.p2_path_up, self.p3_path_up = self._FPN(config[1]) # FPN Up

    def forward(self, p1, p2, p3):
        p2 = self._forward_path(p3, self.p2_path_down, p2)
        p1 = self._forward_path(p2, self.p1_path_down, p1)
        p2 = self._forward_path(p1, self.p2_path_up, p2)
        p3 = self._forward_path(p2, self.p3_path_up, p3)
        return p1, p2, p3

    def _FPN(self, config):
        layers = [make_layer(sub_config) for sub_config in config]
        return layers

    def _forward_path(self, x, path, p):
        for layer in path:
            if isinstance(layer, Concat):
                x = layer(x, p)
            else:
                x = layer(x)
        return x
    
class Head(nn.Module):
    def __init__(self, config):
        super(Head, self).__init__()
        self.config = config
        self.p1_path, self.p2_path, self.p3_path = self._make_layers()
        
    def forward(self, p1, p2, p3):
        p1_c, p1_r = self.p1_path[0](p1), self.p1_path[1](p1)
        p2_c, p2_r = self.p2_path[0](p2), self.p2_path[1](p2)
        p3_c, p3_r = self.p3_path[0](p3), self.p3_path[1](p3)
        preds = [torch.concat([p1_r, p1_c], dim=1), torch.concat([p2_r, p2_c], dim=1),torch.concat([p3_r, p3_c], dim=1)]
        return preds
    
    def _make_layers(self):
        return [nn.ModuleList([make_layer(sub_config) for sub_config in config]) for config in self.config]


class YOLOV8(nn.Module):
    def __init__(self, configs):
        super(YOLOV8, self).__init__()
        self.backbone = Backbone(configs[0])
        self.neck = Neck(configs[1])
        self.head = Head(configs[2])
        self.reg_max = 16
        self.nc = 2
        self.stride = torch.tensor([ 8., 16., 32.]) # model strides

        self.initialize_weights()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.proj = torch.arange(self.reg_max , dtype=torch.float, device=self.device)

    def forward(self, x):
        p1, p2, p3 = self.backbone(x)
        p1, p2, p3 = self.neck(p1, p2, p3)
        preds = self.head(p1, p2, p3) 
        if self.training:      
            return preds
        
        return (self._inference(preds), preds)
    
    def initialize_weights(self):
        """Initialize model weights to random values."""
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
                m.inplace = True

    def _inference(self, x):
        """
        Decode predicted bounding boxes and class probabilities based on multiple-level feature maps.

        Args:
            x (List[torch.Tensor]): List of feature maps from different detection layers.

        Returns:
            (torch.Tensor): Concatenated tensor of decoded bounding boxes and class probabilities.
        """
        # Inference path
        feats = x
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.nc + self.reg_max *4, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        dbox = self.bbox_decode(anchor_points, pred_distri) * stride_tensor
        return torch.cat((dbox.permute(0, 2, 1), pred_scores.sigmoid().permute(0, 2, 1)), 1)
    
    def decode_bboxes(self, bboxes, anchors, xywh=True):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=xywh, dim=1)
    
    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if True:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=True)
    