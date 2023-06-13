from __future__ import print_function, absolute_import
import time

import torch
import torch.nn.functional as F
from torch.nn import init

from .utils.meters import AverageMeter
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from clustercontrast.models.dsbn import CBN2d,CBN1d
from clustercontrast.utils.triplet import SoftTripletLoss_vallia

def  plot_embedding_2d(X, y,true_gt,cid, epoch,iter, title=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    color_dict  = {}
    for i in range(X.shape[0]):
        # plot colored number
        if y[i] not in color_dict.keys():
            color_dict[y[i]] = len(color_dict.keys()) * 1.0 / 10.0 + 0.1

        plt.text(X[i, 0], X[i, 1], "{}_{}".format(true_gt[i],cid[i]),
                 color=plt.cm.Set3(color_dict[y[i]]),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    # plt.show()
    plt.savefig("./view/cluster_result_{}_{}.png".format(epoch,iter))


def plot_embedding_3d(X, y, title=None):
    # 坐标缩放到[0,1]区间
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)
    # 降维后的坐标为（X[i, 0], X[i, 1],X[i,2]），在该位置画出对应的digits
    fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax = Axes3D(fig)
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1], X[i, 2], str(y[i]),
                color=plt.cm.Set3(y[i] / 10.),
                fontdict={'weight': 'bold', 'size': 9})
    if title is not None:
        plt.title(title)
    plt.show()

class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)

class Camera_classifier(torch.nn.Module):

    def __init__(self):
        super(Camera_classifier, self).__init__()
        # self.classifier = torch.nn.Linear(2048, 6, bias=False)
        # init.normal_(self.classifier.weight, std=0.001)
        self.fc1 = torch.nn.Linear(2048, 100)
        self.fc2 = torch.nn.Linear(100, 6)

    def forward(self, input, constant):
        input = GradReverse.grad_reverse(input, constant)
        # logits = self.classifier(input)
        # logits = F.sigmoid(self.classifier(input))
        logits = F.relu(self.fc1(input))
        logits = F.log_softmax(self.fc2(logits), 1)

        return logits

class ClusterContrastTrainer(object):
    def __init__(self, encoder, memory=None,use_teacher=False, use_bnn=False,alpha=0.999,norm_flag=False):
        super(ClusterContrastTrainer, self).__init__()
        self.encoder = encoder
        # self.ema_encoder = ema_encoder
        self.alpha = alpha
        self.memory = memory
        self.encoder_static = None
        self.use_teacher = use_teacher
        self.norm_flag = norm_flag
        self.use_bnn = use_bnn
        self.criterion_triple = SoftTripletLoss_vallia(margin=0.0).cuda()

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()
        # self.ema_encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        tri_losses = AverageMeter()

        sim_losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):


            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            if not self.norm_flag:
                inputs, labels, indexes, gt_label, cid = self._parse_data(inputs)
            else:
                inputs, norm_inputs, labels, indexes, gt_label, cid = self._parse_data(inputs)

            # forward
            if self.use_bnn:
                f_out,bnn_f_out = self._forward(inputs)
            else:
                f_out = self._forward(inputs)


            if self.use_teacher:
                f_out_t = self._forward_ema(inputs)
                loss = self.memory(f_out, labels, f_out_t)
            else:
                loss = self.memory(f_out, labels)

            if self.use_bnn:
                tri_loss = self.criterion_triple(bnn_f_out, bnn_f_out, labels)
                loss += tri_loss

            # if self.norm_flag:
                # norm_out = self._forward(norm_inputs)
                # sim_matrix = torch.cosine_similarity(f_out, norm_out, dim=1)
                # dist_matrix = 1 - sim_matrix
                # loss_sim = torch.sum(dist_matrix)
                # loss += 0.1 * loss_sim
                # exit()


            # with torch.no_grad():
            #     if i % 20 == 0:
            #         self.encoder_static.eval()
            #         gt = labels.clone()
            #         true_gt = gt_label.clone()
            #         feature = self.encoder_static(inputs)
            #         tsne2d = TSNE(n_components=2, init='pca', random_state=0)
            #         X_tsne_2d = tsne2d.fit_transform(feature.detach().cpu().numpy())
            #         plot_embedding_2d(X_tsne_2d[:, 0:2], gt.detach().cpu().numpy(),true_gt.detach().cpu().numpy(),cid.detach().cpu().numpy(),epoch,i ,"t-SNE 2D")
            # print("f_out shape: {}".format(f_out.shape))
            # compute loss with the hybrid memory
            # loss = self.memory(f_out, indexes)



            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新ema模型
            # self._update_ema_variables(self.encoder, self.ema_encoder, self.alpha, epoch * len(data_loader) + i)

            losses.update(loss.item())
            if self.use_bnn:
                tri_losses.update(tri_loss.item())
            else:
                tri_losses.update(0.0)

            if self.norm_flag:
                sim_losses.update(0.0)
            else:
                sim_losses.update(0.0)

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'TriL {:.3f} ({:.3f})\t'
                      'SimL {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,tri_losses.val,tri_losses.avg,
                              sim_losses.val,sim_losses.avg))

    def _parse_data(self, inputs):
        if not self.norm_flag:
            imgs, _, pids, cid, indexes, gt_label = inputs
            return imgs.cuda(), pids.cuda(), indexes.cuda(), gt_label.cuda(), cid.cuda()
        else:
            (imgs,wo_trans_imgs), _, pids, cid, indexes, gt_label = inputs
            return imgs.cuda(),wo_trans_imgs.cuda(), pids.cuda(), indexes.cuda(), gt_label.cuda(), cid.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)

    def _forward_static(self, inputs):
        return self.encoder_static(inputs)

    def _forward_ema(self,inputs):
        return self.ema_encoder(inputs)

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        # alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)



class DANNTrainer(object):
    def __init__(self, encoder, memory=None,use_teacher=False, use_bnn=False,alpha=0.999,norm_flag=False):
        super(DANNTrainer, self).__init__()
        self.encoder = encoder
        self.alpha = alpha
        self.memory = memory
        self.encoder_static = None
        self.use_teacher = use_teacher
        self.norm_flag = norm_flag
        self.use_bnn = use_bnn
        self.camera_cls_model = Camera_classifier().cuda()
        self.camera_loss = torch.nn.NLLLoss()
        self.criterion_triple = SoftTripletLoss_vallia(margin=0.0).cuda()
        self.epoch = 0

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        ce_losses = AverageMeter()
        cam_losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            p = float(i + self.epoch * train_iters) / 120 * train_iters
            constant = 2. / (1. + np.exp(-10 * p)) - 1
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, labels, indexes, gt_label, cid = self._parse_data(inputs)
            # forward
            # prob,f_out = self.encoder(inputs,constant)#self._forward(inputs)
            f_out = self.encoder(inputs)
            ce_loss = self.memory(f_out, labels)
            probs = self.camera_cls_model(f_out, constant)
            cam_loss = self.camera_loss(probs, cid)#F.cross_entropy(prob, cid)

            loss = cam_loss + ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())
            ce_losses.update(ce_loss.item())
            cam_losses.update(cam_loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'CELoss {:.3f} ({:.3f})\t'
                      'CAMLoss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              ce_losses.val,ce_losses.avg,
                              cam_losses.val,cam_losses.avg))
        self.epoch += 1

    def _parse_data(self, inputs):
        imgs, _, pids, cid, indexes, gt_label = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda(), gt_label.cuda(), cid.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)



class CBNTrainer(object):
    def __init__(self, encoder,alpha=0.999):
        super(CBNTrainer, self).__init__()
        self.encoder = encoder
        self.alpha = alpha
        self.memory = []


    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()


        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader[0].next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, labels, indexes, gt_label, cid = self._parse_data(inputs)

            # forward
            CBN1d.camera_index = 0
            CBN2d.camera_index = 0
            f_out = self._forward(inputs)
            loss = self.memory[0](f_out, labels)
            for camera_index in range(1,len(data_loader)):
                # load data
                inputs = data_loader[camera_index].next()
                data_time.update(time.time() - end)

                # process inputs
                inputs, labels, indexes, gt_label, cid = self._parse_data(inputs)

                # forward
                CBN1d.camera_index = camera_index
                CBN2d.camera_index = camera_index
                f_out = self._forward(inputs)

                loss += self.memory[camera_index](f_out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, cid, indexes, gt_label = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda(), gt_label.cuda(), cid.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)

#    def _forward_static(self, inputs):
#        return self.encoder_static(inputs)

    # def _forward_ema(self,inputs):
    #     return self.ema_encoder(inputs)

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


class CameraCCTrainer(object):
    def __init__(self, encoder,camera_encoder, memory=None,use_teacher=False, use_bnn=False,alpha=0.999):
        super(CameraCCTrainer, self).__init__()
        self.encoder = encoder
        self.camera_encoder = camera_encoder
        self.alpha = alpha
        self.memory = memory
        self.use_bnn = use_bnn
        self.criterion_triple = SoftTripletLoss_vallia(margin=0.0).cuda()

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()
        self.camera_encoder.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        tri_losses = AverageMeter()
        camera_losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, labels, indexes, gt_label, cid = self._parse_data(inputs)

            # forward
            if self.use_bnn:
                f_out,bnn_f_out = self._forward(inputs)
            else:
                f_out = self._forward(inputs)


            f_out_camera = self._forward_camera(inputs)
            f_out += f_out_camera

            loss = self.memory(f_out, labels)

            if self.use_bnn:
                bnn_f_out += f_out_camera
                tri_loss = self.criterion_triple(bnn_f_out, bnn_f_out, labels)
                loss += tri_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())
            if self.use_bnn:
                tri_losses.update(tri_loss.item())
            else:
                tri_losses.update(0.0)

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'TriL {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,tri_losses.val,tri_losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, cid, indexes, gt_label = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda(), gt_label.cuda(), cid.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)

    def _forward_camera(self,inputs):
        return self.camera_encoder(inputs)


class CamTrainer(object):
    def __init__(self, encoder, memory=None, cam_memory=None,use_bnn=False,alpha=0.999):
        super(CamTrainer, self).__init__()
        self.encoder = encoder
        self.alpha = alpha
        self.memory = memory
        self.cam_memory = cam_memory
        self.use_bnn = use_bnn
        self.criterion_triple = SoftTripletLoss_vallia(margin=0.0).cuda()

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        tri_losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, labels, indexes, gt_label, cid = self._parse_data(inputs)

            # forward
            if self.use_bnn:
                f_out,bnn_f_out = self._forward(inputs)
            else:
                f_out = self._forward(inputs)

            inputs = torch.clone(f_out)
            targets = torch.clone(labels)
            # label_muti = self.cam_memory.camera_num_samples[0]*cid+targets


            target_values = torch.ones(targets.shape[0], self.cam_memory.num_samples).cuda()
            label_muti = torch.zeros(targets.shape[0], self.cam_memory.num_samples).cuda()

            new_t = targets.reshape(-1, 1).cuda()
            label_muti.scatter_(1, new_t, target_values)
            for _i in range(len(self.cam_memory.camera_num_samples) - 1):
                new_t += self.cam_memory.camera_num_samples[_i]
                label_muti.scatter_(1, new_t, target_values)

            loss = self.cam_memory(inputs,label_muti,cid)
            # loss = self.memory(f_out, labels)
            #
            # loss = 0.5 * loss + 0.5 * loss_camera

            if self.use_bnn:
                tri_loss = self.criterion_triple(bnn_f_out, bnn_f_out, labels)
                loss += tri_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())
            if self.use_bnn:
                tri_losses.update(tri_loss.item())
            else:
                tri_losses.update(0.0)

            # print log
            batch_time.update(time.time() - end)
            end = time.time()


            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'TriL {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,tri_losses.val,tri_losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, cid, indexes, gt_label = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda(), gt_label.cuda(), cid.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)

