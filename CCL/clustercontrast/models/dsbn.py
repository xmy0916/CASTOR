import torch
import torch.nn as nn

# Domain-specific BatchNorm

class CBN2d(nn.Module):
    camera_index = None
    def __init__(self, planes, camera_num):
        super(CBN2d, self).__init__()
        self.num_features = planes
        self.BN_list = nn.ModuleList()
        self.camera_num = camera_num
        for i in range(camera_num):
            self.BN_list.append(nn.BatchNorm2d(planes).cuda())

    def forward(self, x):
        if self.training:
            self.BN_list[CBN2d.camera_index].train()
            return self.BN_list[CBN2d.camera_index](x)
        else:
            out = x.contiguous()
            for i in range(self.camera_num):
                camera_index_feature = x[CBN2d.camera_index == i]
                out_tmp = self.BN_list[i](camera_index_feature.contiguous())
                out[CBN2d.camera_index == i] = out_tmp
            return out


class CBN1d(nn.Module):
    camera_index = None
    def __init__(self, planes, camera_num):
        super(CBN1d, self).__init__()
        self.num_features = planes
        self.BN_list = nn.ModuleList()
        self.camera_num = camera_num
        for i in range(camera_num):
            self.BN_list.append(nn.BatchNorm1d(planes).cuda())

    def forward(self, x):
        if self.training:
            self.BN_list[CBN1d.camera_index].train()
            return self.BN_list[CBN1d.camera_index](x)
        else:
            out = x.clone()
            for i in range(self.camera_num):
                camera_index_feature = x[CBN1d.camera_index == i]
                out_tmp = self.BN_list[i](camera_index_feature.contiguous())
                out[CBN1d.camera_index == i] = out_tmp
            return out


def convert_cbn(model,camera_num):
    for _, (child_name, child) in enumerate(model.named_children()):
        assert(not next(model.parameters()).is_cuda)
        if isinstance(child, nn.BatchNorm2d):
            m = CBN2d(child.num_features, camera_num)
            for i in range(camera_num):
                m.BN_list[i].load_state_dict(child.state_dict())
            setattr(model, child_name, m)
        elif isinstance(child, nn.BatchNorm1d):
            m = CBN1d(child.num_features, camera_num)
            for i in range(camera_num):
                m.BN_list[i].load_state_dict(child.state_dict())
            setattr(model, child_name, m)
        else:
            convert_cbn(child,camera_num)
