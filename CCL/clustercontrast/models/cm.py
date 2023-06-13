import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd
from clustercontrast.utils.crossentropy import SoftEntropy


class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Teacher(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, inputs_teacher, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs_teacher, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs_teacher, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs_teacher, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm_teacher(inputs, inputs_teacher, indexes, features, momentum=0.5):
    return CM_Teacher.apply(inputs, inputs_teacher, indexes, features, torch.Tensor([momentum]).to(inputs.device))

class CM_Hard_Camera(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, cids, cam_memory_num_list,features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets, cids, cam_memory_num_list)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets, cids, cam_memory_num_list = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update

        for x, y, cid in zip(inputs, targets, cids):
            # np_y = y.cpu().numpy()
            # print(np.where(np_y>0))
            ctx.features[y>0][cid] = ctx.momentum * ctx.features[y>0][cid] + (1. - ctx.momentum) * x
            ctx.features[y>0][cid] /= ctx.features[y>0][cid].norm()

        return grad_inputs, None, None, None, None, None


def cm_hard_camera(inputs, indexes, cids, cam_memory_num_list,features, momentum=0.5):
    return CM_Hard_Camera.apply(inputs, indexes, cids, cam_memory_num_list, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets,features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))




class CM_Hard_Teacher(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, inputs_teacher, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs_teacher, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs_teacher, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs_teacher, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None, None


def cm_hard_teacher(inputs, inputs_teacher, indexes, features, momentum=0.5):
    return CM_Hard_Teacher.apply(inputs, inputs_teacher, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, loss_func='ce',temp=0.05, momentum=0.2, use_hard=False, use_teacher=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard
        self.loss_func = loss_func
        self.use_teacher = use_teacher

        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('labels', torch.zeros(num_samples).long())

    def forward(self, inputs, targets, inputs_teacher=None):


        inputs = F.normalize(inputs, dim=1).cuda()
        if self.use_hard:
            if self.use_teacher:
                inputs = cm_hard_teacher(inputs, inputs_teacher, targets, self.features, self.momentum)
            else:
                inputs = cm_hard(inputs, targets, self.features, self.momentum)
        else:
            if self.use_teacher:
                inputs = cm_teacher(inputs, inputs_teacher, targets, self.features, self.momentum)
            else:
                inputs = cm(inputs, targets, self.features, self.momentum)

        inputs /= self.temp
        if self.loss_func == "ce":
            loss = F.cross_entropy(inputs, targets)
            return loss
        else:
            def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
                exps = torch.exp(vec)
                masked_exps = exps * mask.float().clone()
                masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
                return (masked_exps/masked_sums)

            inputs /= self.temp
            B = inputs.size(0)
            labels = self.labels.clone()

            # sim:13619,64
            # labels:37778
            # inputs.t():37778,64

            sim = torch.zeros(labels.max() + 1, B).float().cuda()
            sim.index_add_(0, labels, inputs.t().contiguous())
            nums = torch.zeros(labels.max() + 1, 1).float().cuda()
            nums.index_add_(0, labels, torch.ones(self.num_samples, 1).float().cuda())
            mask = (nums > 0).float()
            sim /= (mask * nums + (1 - mask)).clone().expand_as(sim)
            mask = mask.expand_as(sim)
            masked_sim = masked_softmax(sim.t().contiguous(), mask.t().contiguous())
            return F.nll_loss(torch.log(masked_sim + 1e-6), targets)


class CamMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, camera_num_samples, temp=0.05, momentum=0.2, use_hard=False):
        super(CamMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.camera_num_samples = camera_num_samples
        self.camera_num_samples_tensor = torch.from_numpy(np.array(camera_num_samples))

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard

        self.ce_soft = SoftEntropy()


    def forward(self, inputs, targets, cid):
        inputs = F.normalize(inputs, dim=1).cuda()

        inputs = cm_hard_camera(inputs, targets, cid, self.camera_num_samples_tensor, self.features, self.momentum)
        inputs /= self.temp
        # tar = F.normalize(targets, dim=1, p=1).cuda()
        # inp = F.normalize(inputs, dim=1, p=1).cuda()
        loss = self.ce_soft(inputs, targets)
        return loss
