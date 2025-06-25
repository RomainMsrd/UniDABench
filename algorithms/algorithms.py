import copy

import ot
import torch
import torch.nn as nn
import numpy as np
import itertools
import skimage.filters as sfil

from easydl import TrainingModeManager, Accumulator
from sklearn.cluster import KMeans
from torch.nn import BCELoss
from tqdm import tqdm

from models.models import classifier, ReverseLayerF, \
    DiscriminatorUDA, classifierOVANet, classifierNoBias, LinearAverage, CLS, \
    ProtoCLS, MemoryQueue, ClassMemoryQueue
from models.loss import ConditionalEntropyLoss, Entropy
from utils import adaptive_filling, ubot_CCD, sinkhorn, ubot_CCD2, adaptive_filling2
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
import torch.nn. functional as F



def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs, backbone):
        super(Algorithm, self).__init__()
        self.configs = configs

        self.cross_entropy = nn.CrossEntropyLoss()
        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)
        self.is_uniDA = False
        self.uniDA = True

        self.optimizer = torch.optim.Adam(
            list(self.network.parameters()),
            lr=0.005,
        )


    # update function is common to all algorithms
    def update(self, src_loader, trg_loader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # training loop 
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())


            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        last_model = self.network.state_dict()

        return last_model, best_model
    def pretrain_epoch(self, src_loader, avg_meter):

        for src_x, src_y in src_loader:
            src_x, src_y = src_x.to(self.device), src_y.to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)

            loss = src_cls_loss

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            losses = {'Pr_Src_cls_loss': loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)
    def get_latent_features(self, dataloader):
        feature_set = []
        pred_set = []
        label_set = []
        self.feature_extractor.eval()
        self.classifier.eval()
        with torch.no_grad():
            for _, (data, label) in enumerate(dataloader):
                data = data.to(self.device)
                feature = self.feature_extractor(data)
                pred = F.softmax(self.classifier(feature))
                pred_set.append(pred.cpu())
                feature_set.append(feature.cpu())
                label_set.append(label.cpu())
            feature_set = torch.cat(feature_set, dim=0)
            pred_set = torch.cat(pred_set, dim=0)
            feature_set = F.normalize(feature_set, p=2, dim=-1)
            label_set = torch.cat(label_set, dim=0)
        return feature_set, label_set, pred_set

    def evaluate(self, test_loader, trg_private_class, src=False):
        feature_extractor = self.feature_extractor.to(self.device)
        classifier = self.classifier.to(self.device)

        feature_extractor.eval()
        classifier.eval()

        total_loss, preds_list, labels_list = [], [], []

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features = feature_extractor(data)
                predictions = classifier(features)

                # compute loss
                if self.uniDA:
                    if src and self.is_uniDA:
                        corr_preds = self.correct_predictions(predictions)
                        loss = F.cross_entropy(corr_preds, labels)
                        #loss = F.cross_entropy(predictions[m], labels[m])
                    else:
                        #m = torch.isin(labels, self.trg_private_class.view((-1)).long().to(self.device), invert=True)
                        m = torch.isin(labels.cpu(), trg_private_class, invert=True)
                        loss = F.cross_entropy(predictions[m], labels[m])
                else:
                    loss = F.cross_entropy(predictions, labels)
                total_loss.append(loss.detach().cpu().item())
                #predictions = self.algorithm.correct_predictions(predictions)
                pred = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)
        loss = torch.tensor(total_loss).mean()  # average loss
        full_preds = torch.cat((preds_list))
        full_labels = torch.cat((labels_list))
        return loss, full_preds, full_labels
    def correct_predictions(self, preds):
        return preds

    def decision_function(self, preds):
        confidence, pred = preds.max(dim=1)
        return pred
    # train loop vary from one method to another
    def training_epoch(self, *args, **kwargs):
        raise NotImplementedError


class NO_ADAPT(Algorithm):
    """
    Lower bound: train on source and test on target.
    """
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        #self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):
        for src_x, src_y in src_loader:

            src_x, src_y = src_x.to(self.device), src_y.to(self.device)
            print(src_x.shape)
            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)

            loss = src_cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses = {'Src_cls_loss': src_cls_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        #self.lr_scheduler.step()


class TARGET_ONLY(Algorithm):
    """
    Upper bound: train on target and test on target.
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):

        for trg_x, trg_y in trg_loader:

            trg_x, trg_y = trg_x.to(self.device), trg_y.to(self.device)

            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)

            trg_cls_loss = self.cross_entropy(trg_pred, trg_y)

            loss = trg_cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses = {'Trg_cls_loss': trg_cls_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()

class UDA(Algorithm):
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)


        # optimizer and scheduler

        #self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Domain Discriminator
        self.domain_classifier = DiscriminatorUDA(configs)
        self.adv_discriminator = DiscriminatorUDA(configs)

        self.conditional_entropy = ConditionalEntropyLoss()
        self.optimizer = torch.optim.Adam(
            list(self.network.parameters()) + list(self.adv_discriminator.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.w_0 = hparams["w0"]

        #self.bce = BCEWithLogitsLoss()
        self.bce = BCELoss()

    def normalize_weight(self, x):
        min_val = x.min()
        max_val = x.max()
        x = (x - min_val) / (max_val - min_val)
        #print(torch.mean(x))
        x = x/torch.mean(x)
        #assert (x > 1.0).any() or (x < 0.0).any()
        #x = (x-torch.mean(x))/torch.std(x)
        return x.detach()

    def reverse_sigmoid(self, y):
        return torch.log(y / (1.0 - y + 1e-10) + 1e-10)
    def get_src_weights(self, domain_out, before_softmax, domain_temperature=1.0, class_temperature=10.0):
        before_softmax = before_softmax / class_temperature
        after_softmax = nn.Softmax(-1)(before_softmax)
        domain_logit = self.reverse_sigmoid(domain_out)
        domain_logit = domain_logit / domain_temperature
        domain_out = nn.Sigmoid()(domain_logit)

        entropy = torch.sum(- after_softmax * torch.log(after_softmax + 1e-10), dim=1, keepdim=True)
        entropy_norm = entropy / np.log(after_softmax.size(1))
        #entropy_norm = self.normalize_weight(entropy_norm)
        #assert (entropy_norm > 1.0).any() or (entropy_norm < 0.0).any() == False
        weight = entropy_norm - domain_out
        #print(max(weight))
        weight = weight.detach()
        return weight
    def get_trg_weights(self, domain_out, before_softmax, domain_temperature=1.0, class_temperature=1.0):
        return -1*self.get_src_weights(domain_out, before_softmax, domain_temperature, class_temperature)

    def update(self, src_loader, trg_loader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        nb_pr_epochs = self.hparams["num_epochs_pr"]
        for epoch in range(1, nb_pr_epochs+1):
            self.pretrain_epoch(src_loader, avg_meter)

            logger.debug(f'[Pr Epoch : {epoch}/{nb_pr_epochs}]') #TODO : self.hparams["num_pr_epochs"]
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        with torch.no_grad():
            self.feature_extractor.eval()
            self.classifier.eval()
            X = src_loader.dataset.x_data.cuda()
            Y = src_loader.dataset.y_data.numpy()
            logits = self.classifier(self.feature_extractor(X))
            preds = logits.detach().cpu().argmax(axis=1).numpy()
            print("SRC Accuracy : ", (Y == preds).sum() / len(Y))
        self.feature_extractor.train()
        self.classifier.train()
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # source pretraining loop
            #self.pretrain_epoch(src_loader, avg_meter)

            # training loop
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        last_model = self.network.state_dict()

        return last_model, best_model

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        # Combine dataloaders
        # Method 1 (min len of both domains)
        # joint_loader = enumerate(zip(src_loader, trg_loader))

        # Method 2 (max len of both domains)
        # joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:

            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            p = float(step + epoch * num_batches) / self.hparams["num_epochs"] + 1 / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # zero grad
            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()

            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)

            # Task classification  Loss
            src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

            # Adv Domain Discriminator loss
            # source
            src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
            src_adv_pred = self.adv_discriminator(src_feat_reversed)

            # target
            trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
            trg_adv_pred = self.adv_discriminator(trg_feat_reversed)

            # Domain classifier and weights computation
            src_domain_pred = self.domain_classifier(src_feat)
            trg_domain_pred = self.domain_classifier(trg_feat)

            src_temp = 10
            '''w_s = self.normalize_weight(self.conditional_entropy(src_domain_pred/src_temp)/np.log(len(src_domain_pred)) - src_domain_pred/src_temp)
            w_t = self.normalize_weight(trg_domain_pred - self.conditional_entropy(trg_domain_pred)/np.log(len(trg_domain_pred)))'''
            w_s = self.normalize_weight(self.get_src_weights(src_domain_pred, src_pred))
            w_t = self.normalize_weight(self.get_trg_weights(trg_domain_pred, trg_pred))
            # print(w_t)

            src_domain_loss = self.bce(src_domain_pred.squeeze(), domain_label_src)
            trg_domain_loss = self.bce(trg_domain_pred.squeeze(), domain_label_trg)

            '''src_adv_loss = w_s * F.cross_entropy(src_adv_pred, domain_label_src.long(), reduction='none')
            src_adv_loss = src_adv_loss.mean()
            trg_adv_loss = w_t * F.cross_entropy(trg_adv_pred, domain_label_trg.long(), reduction='none')
            trg_adv_loss = trg_adv_loss.mean()'''

            src_adv_loss = w_s * F.binary_cross_entropy(src_adv_pred.squeeze(), domain_label_src, reduction='none')
            src_adv_loss = src_adv_loss.mean()
            trg_adv_loss = w_t * F.binary_cross_entropy(trg_adv_pred.squeeze(), domain_label_trg, reduction='none')
            trg_adv_loss = trg_adv_loss.mean()

            # Task classification  Loss
            """mask = w_s < self.w_0/2
            print("Wrong Src : ", mask.sum())
            w_s2 = w_s.clone()
            w_s2[mask] = 0
            src_cls_loss = (w_s2)*F.cross_entropy(src_pred.squeeze(), src_y, reduction='none')
            src_cls_loss = src_cls_loss.mean()"""

            # Total domain loss
            domain_loss = src_domain_loss + trg_domain_loss
            adv_loss = src_adv_loss + trg_adv_loss

            loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
                   self.hparams["domain_loss_wt"] * adv_loss

            loss.backward(retain_graph=True)
            domain_loss.backward()
            self.optimizer.step()
            self.optimizer_disc.step()

            losses = {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item(),
                      "Adv Loss": adv_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

    def evaluate(self, test_loader, trg_private_class, src=False):
        self.feature_extractor.eval()
        self.classifier.eval()

        total_loss, preds_list, labels_list = [], [], []

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features = self.feature_extractor(data)
                predictions = self.classifier(features)
                trg_domain_pred = self.domain_classifier(features)
                #w_t = trg_domain_pred - self.conditional_entropy(trg_domain_pred)/np.log(len(trg_domain_pred))
                w_t = self.normalize_weight(self.get_trg_weights(trg_domain_pred, predictions))
                #print(w_t)
                mask = w_t < self.w_0

                if not src:
                    predictions[mask.squeeze()] *= 0

                if self.is_uniDA:
                    mask = labels >= predictions.shape[-1]
                    labels[mask] = predictions.shape[-1]

                mask = labels < predictions.shape[-1]
                # z = torch.zeros((len(predictions), 1))
                # predictions = torch.cat((predictions, z.to(predictions.device)), dim=1)
                loss = F.cross_entropy(predictions[mask], labels[mask])
                total_loss.append(loss.detach().cpu().item())
                # predictions = self.algorithm.correct_predictions(predictions)
                pred = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)

        loss = torch.tensor(total_loss).mean()  # average loss
        full_preds = torch.cat((preds_list))
        full_labels = torch.cat((labels_list))
        return loss, full_preds, full_labels

    def decision_function(self, preds):
        mask = preds.sum(axis=1) == 0.0
        confidence, pred = preds.max(dim=1)
        pred[mask] = -1
        return pred


class OVANet(Algorithm):
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)


        # optimizer and scheduler

        #self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Domain Discriminator
        self.open_set_classifier = classifierOVANet(configs)

        self.optimizer_feature_gen = torch.optim.Adam(
            list(self.feature_extractor.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer_clasifier = torch.optim.Adam(
            list(self.open_set_classifier.parameters()) + list(self.classifier.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.entropy = Entropy()

    def ova_loss(self, open_preds, label):
        assert len(open_preds.size()) == 3
        assert open_preds.size(1) == 2

        out_open = F.softmax(open_preds, 1)
        label_p = torch.zeros((out_open.size(0),
                               out_open.size(2))).long().cuda()
        label_range = torch.range(0, out_open.size(0) - 1).long()
        label_p[label_range, label] = 1
        label_n = 1 - label_p
        open_loss_pos = torch.mean(torch.sum(-torch.log(out_open[:, 1, :]
                                                        + 1e-8) * label_p, 1))
        open_loss_neg = torch.mean(torch.max(-torch.log(out_open[:, 0, :] +
                                                        1e-8) * label_n, 1)[0])
        return open_loss_pos, open_loss_neg

    def open_entropy(self, open_preds):
        assert len(open_preds.size()) == 3
        assert open_preds.size(1) == 2
        out_open = F.softmax(open_preds, 1)
        ent_open = torch.mean(torch.mean(torch.sum(-out_open * torch.log(out_open + 1e-8), 1), 1))
        return ent_open

    def entropy(self, p, prob=True, mean=True):
        if prob:
            p = F.softmax(p)
        en = -torch.sum(p * torch.log(p + 1e-5), 1)
        if mean:
            return torch.mean(en)
        else:
            return en

    def update(self, src_loader, trg_loader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        nb_pr_epochs = self.hparams["num_epochs_pr"]
        for epoch in range(1, nb_pr_epochs+1):
            self.pretrain_epoch(src_loader, avg_meter)

            logger.debug(f'[Pr Epoch : {epoch}/{nb_pr_epochs}]') #TODO : self.hparams["num_pr_epochs"]
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        with torch.no_grad():
            self.network.eval()
            X = src_loader.dataset.x_data.cuda()
            Y = src_loader.dataset.y_data.numpy()
            logits = self.classifier(self.feature_extractor(X))
            preds = logits.detach().cpu().argmax(axis=1).numpy()
            print("SRC Accuracy : ", (Y == preds).sum() / len(Y))
        self.network.train()
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # source pretraining loop
            #self.pretrain_epoch(src_loader, avg_meter)

            # training loop
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        last_model = self.network.state_dict()

        return last_model, best_model

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        # Combine dataloaders
        # Method 1 (min len of both domains)
        # joint_loader = enumerate(zip(src_loader, trg_loader))

        # Method 2 (max len of both domains)
        # joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:

            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            p = float(step + epoch * num_batches) / self.hparams["num_epochs"] + 1 / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # zero grad
            self.optimizer_clasifier.zero_grad()
            self.optimizer_feature_gen.zero_grad()

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)
            src_open = self.open_set_classifier(src_feat)
            src_open = src_open.view(src_open.size(0), 2, -1)

            src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)
            open_loss_pos, open_loss_neg = self.ova_loss(src_open, src_y)
            ## b x 2 x C
            loss_open = 0.5 * (open_loss_pos + open_loss_neg)
            total_loss = loss_open + src_cls_loss

            trg_feat = self.feature_extractor(trg_x)
            trg_open_pred = self.open_set_classifier(trg_feat)
            trg_open_pred = trg_open_pred.view(trg_open_pred.size(0), 2, -1)

            out_open_t = trg_open_pred.view(trg_x.size(0), 2, -1)
            ent_open = self.open_entropy(out_open_t)
            total_loss += ent_open

            total_loss.backward()
            self.optimizer_feature_gen.step()
            self.optimizer_clasifier.step()
            self.optimizer_feature_gen.zero_grad()
            self.optimizer_clasifier.zero_grad()

            losses = {'Total_loss': total_loss.item(), 'Open Loss': loss_open.item(), 'Src_cls_loss': src_cls_loss.item(),
                      "Entropy Open": ent_open.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

    def evaluate(self, test_loader, trg_private_class, src=False):
        feature_extractor = self.feature_extractor.to(self.device)
        classifier = self.classifier.to(self.device)

        feature_extractor.eval()
        classifier.eval()

        total_loss, preds_list, labels_list = [], [], []

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features = self.feature_extractor(data)
                predictions = F.softmax(self.classifier(features))
                open_preds = self.open_set_classifier(features)

                # open_class = len(predictions.shape[0])

                conf, pred = predictions.max(dim=1)
                # entr = -1*torch.sum(predictions*torch.log(predictions), 1).data.cpu().numpy()

                open_preds = F.softmax(open_preds.view(predictions.size(0), 2, -1), 1)
                tmp_range = torch.range(0, predictions.size(0) - 1).long().cuda()
                pred_unk = open_preds[tmp_range, 0, pred]
                ind_unk = np.where(pred_unk.data.cpu().numpy() > 0.5)[0]
                pred[ind_unk] = predictions.shape[-1]
                #mask = labels >= predictions.shape[-1]
                mask = ind_unk

                if not src:
                    predictions[mask.squeeze()] *= 0

                if self.is_uniDA:
                    mask = labels >= predictions.shape[-1]
                    labels[mask] = predictions.shape[-1]

                mask = labels < predictions.shape[-1]
                # z = torch.zeros((len(predictions), 1))
                # predictions = torch.cat((predictions, z.to(predictions.device)), dim=1)
                loss = F.cross_entropy(predictions[mask], labels[mask])
                total_loss.append(loss.detach().cpu().item())
                # predictions = self.algorithm.correct_predictions(predictions)
                pred = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)
        loss = torch.tensor(total_loss).mean()  # average loss
        full_preds = torch.cat((preds_list))
        full_labels = torch.cat((labels_list))
        return loss, full_preds, full_labels
    def decision_function(self, preds):
        mask = preds.sum(axis=1) == 0.0
        confidence, pred = preds.max(dim=1)
        pred[mask] = -1
        return pred

class DANCE(Algorithm):
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler

        # self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device
        self.classifier = classifierNoBias(configs)
        self.rho = np.log(self.configs.num_classes)/2.0

        self.optimizer_feature_gen = torch.optim.Adam(
            list(self.feature_extractor.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer_clasifier = torch.optim.Adam(
            self.classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.entropy = Entropy()
        self.configs = configs

    def init_memory(self, trg_loader):
        self.ndata = len(trg_loader.dataset.y_data)
        '''for (trg_x, _, _) in trg_loader:
            self.ndata += len(trg_x)'''
        print(self.ndata)
        final_out_channels = self.configs.final_out_channels
        if self.configs.isFNO:
            final_out_channels = self.configs.final_out_channels * 2
        self.lemniscate = LinearAverage(final_out_channels, self.ndata).to(self.device)

    def entropy(self, p):
        p = F.softmax(p)
        return -torch.mean(torch.sum(p * torch.log(p + 1e-5), 1))

    def entropy_margin(self, p, value, margin=0.2, weight=None):
        p = F.softmax(p)
        return -torch.mean(self.hinge(torch.abs(-torch.sum(p * torch.log(p + 1e-5), 1) - value), margin))

    def hinge(self, input, margin=0.2):
        return torch.clamp(input, min=margin)

    def update(self, src_loader, trg_loader, avg_meter, logger):
        self.init_memory(trg_loader)
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        nb_pr_epochs = self.hparams["num_epochs_pr"]
        for epoch in range(1, nb_pr_epochs + 1):
            self.pretrain_epoch(src_loader, avg_meter)

            logger.debug(f'[Pr Epoch : {epoch}/{nb_pr_epochs}]')  # TODO : self.hparams["num_pr_epochs"]
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        with torch.no_grad():
            self.network.eval()
            X = src_loader.dataset.x_data.cuda()
            Y = src_loader.dataset.y_data.numpy()
            logits = self.classifier(self.feature_extractor(X))
            preds = logits.detach().cpu().argmax(axis=1).numpy()
            print("SRC Accuracy : ", (Y == preds).sum() / len(Y))
            self.network.train()
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # source pretraining loop
            # self.pretrain_epoch(src_loader, avg_meter)

            # training loop
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        last_model = self.network.state_dict()

        return last_model, best_model

    def pretrain_epoch(self, src_loader, avg_meter):

        for src_x, src_y, _ in src_loader:
            src_x, src_y = src_x.to(self.device), src_y.to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)

            loss = src_cls_loss

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            losses = {'Pr_Src_cls_loss': loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)
    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        # Combine dataloaders
        # Method 1 (min len of both domains)
        # joint_loader = enumerate(zip(src_loader, trg_loader))

        # Method 2 (max len of both domains)
        # joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))

        for step, ((src_x, src_y, _), (trg_x, _, trg_index)) in joint_loader:

            src_x, src_y, trg_x, trg_index = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device), trg_index.to(self.device)

            # zero grad
            self.optimizer_clasifier.zero_grad()
            self.optimizer_feature_gen.zero_grad()

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)
            trg_feat = F.normalize(trg_feat)

            feat_mat = self.lemniscate(trg_feat, trg_index)
            feat_mat[:, trg_index] = -1.0
            ### Calculate mini-batch x mini-batch similarity

            feat_mat2 = torch.matmul(trg_feat, trg_feat.t())
            mask = torch.eye(feat_mat2.size(0), feat_mat2.size(0)).bool().to(self.device)
            feat_mat2.masked_fill_(mask, -1)

            loss_nc = self.hparams["eta"] * self.entropy(torch.cat([trg_pred, feat_mat,feat_mat2], 1))
            loss_ent = self.hparams["eta"] * self.entropy_margin(trg_pred, self.rho, self.hparams["margin"])
            total_loss = loss_nc + src_cls_loss + loss_ent

            total_loss.backward()
            self.optimizer_feature_gen.step()
            self.optimizer_clasifier.step()
            self.optimizer_feature_gen.zero_grad()
            self.optimizer_clasifier.zero_grad()

            self.lemniscate.update_weight(trg_feat, trg_index)

            losses = {'Total_loss': total_loss.item(), 'Ent Loss': loss_ent.item(),
                      'Src_cls_loss': src_cls_loss.item(),
                      "Neighbors Clustering ": loss_nc.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

    def evaluate(self, test_loader, trg_private_class, src=False):
        feature_extractor = self.feature_extractor.to(self.device)
        classifier = self.classifier.to(self.device)

        feature_extractor.eval()
        classifier.eval()

        total_loss, preds_list, labels_list = [], [], []

        with torch.no_grad():
            for data, labels, _ in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features = self.feature_extractor(data)
                predictions = F.softmax(self.classifier(features))

                entr = -torch.sum(predictions * torch.log(predictions), 1).data.cpu().numpy()

                conf, pred = predictions.max(dim=1)

                pred_unk = np.where(entr > self.rho)
                pred[pred_unk] = predictions.shape[-1]
                mask = pred_unk
                #mask = labels >= predictions.shape[-1]

                if not src:
                    predictions[mask] *= 0

                if self.is_uniDA:
                    mask = labels >= predictions.shape[-1]
                    labels[mask] = predictions.shape[-1]
                mask = labels < predictions.shape[-1]
                #z = torch.zeros((len(predictions), 1))
                #predictions = torch.cat((predictions, z.to(predictions.device)), dim=1)
                loss = F.cross_entropy(predictions[mask], labels[mask])
                total_loss.append(loss.detach().cpu().item())
                # predictions = self.algorithm.correct_predictions(predictions)
                pred = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)
        loss = torch.tensor(total_loss).mean()  # average loss
        full_preds = torch.cat((preds_list))
        full_labels = torch.cat((labels_list))
        return loss, full_preds, full_labels

    def get_latent_features(self, dataloader):
        feature_set = []
        label_set = []
        logit_set = []
        self.feature_extractor.eval()
        self.classifier.eval()
        with torch.no_grad():
            for _, (data, label, ids) in enumerate(dataloader):
                data = data.to(self.device)
                feature = self.feature_extractor(data)
                logit = self.classifier(feature)
                logit_set.append(logit.cpu())
                feature_set.append(feature.cpu())
                label_set.append(label.cpu())
            feature_set = torch.cat(feature_set, dim=0)
            feature_set = F.normalize(feature_set, p=2, dim=-1)
            label_set = torch.cat(label_set, dim=0)
            logit_set = torch.cat(logit_set, dim=0)
        return feature_set, label_set, logit_set
    def decision_function(self, preds):
        mask = preds.sum(axis=1) == 0.0
        confidence, pred = preds.max(dim=1)
        pred[mask] = -1
        return pred

class PPOT(Algorithm):
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        '''self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )'''

        self.hparams = hparams
        self.configs = configs
        self.device = device

        self.src_prototype = None
        self.num_classes = self.configs.num_classes
        self.alpha = 0
        self.class_weight = 0
        self.beta = 0

        self.softmax = torch.nn.Softmax(dim=1)
        self.is_uniDA = True


    def get_features(self, dataloader):
        feature_set = []
        label_set = []
        self.feature_extractor.eval()
        with torch.no_grad():
            for _, (data, label) in enumerate(dataloader):
                data = data.to(self.device)
                feature = self.feature_extractor(data)
                feature_set.append(feature)
                label_set.append(label)
            feature_set = torch.cat(feature_set, dim=0)
            feature_set = F.normalize(feature_set, p=2, dim=-1)
            label_set = torch.cat(label_set, dim=0)
        return feature_set, label_set

    def get_prototypes(self, dataloader) -> torch.Tensor:
        feature_set, label_set = self.get_features(dataloader)
        class_set = [i for i in range(self.num_classes)]
        source_prototype = torch.zeros((len(class_set), feature_set[0].shape[0]))
        for i in class_set:
            source_prototype[i] = feature_set[label_set == i].sum(0) / feature_set[label_set == i].size(0)
        return source_prototype.to(self.device)

    def update_alpha(self, trg_loader) -> np.ndarray:
        num_conf, num_sample = 0, 0
        self.feature_extractor.eval()
        self.classifier.eval()
        with torch.no_grad():
            for _, (trg_x, _) in enumerate(trg_loader):
                trg_x = trg_x.to(self.device)
                output = self.classifier(self.feature_extractor(trg_x))
                output = self.softmax(output)
                conf, _ = output.max(dim=1)
                num_conf += torch.sum(conf > self.hparams["tau1"]).item()
                num_sample += output.shape[0]
            alpha = num_conf / num_sample
            alpha = np.around(alpha, decimals=2)
        return alpha

    def entropy_loss(self, prediction: torch.Tensor, weight=torch.zeros(1)):
        if weight.size(0) == 1:
            entropy = torch.sum(-prediction * torch.log(prediction + 1e-8), 1)
            entropy = torch.mean(entropy)
        else:
            entropy = torch.sum(-prediction * torch.log(prediction + 1e-8), 1)
            entropy = torch.mean(weight * entropy)
        return entropy

    def update(self, src_loader, trg_loader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        nb_pr_epochs = self.hparams['num_epochs_pr'] #20#+10+50
        for epoch in range(1, nb_pr_epochs + 1):
            self.pretrain_epoch(src_loader, avg_meter)

            logger.debug(f'[Pr Epoch : {epoch}/{nb_pr_epochs}]')  # TODO : self.hparams["num_pr_epochs"]
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        with torch.no_grad():
            self.network.eval()
            X = src_loader.dataset.x_data.cuda()
            Y = src_loader.dataset.y_data.numpy()
            logits = self.classifier(self.feature_extractor(X))
            preds = logits.detach().cpu().argmax(axis=1).numpy()
            print("SRC Accuracy : ", (Y == preds).sum() / len(Y))
        self.network.train()

        self.alpha = self.update_alpha(trg_loader)
        self.beta = self.alpha
        self.class_weight = torch.ones(self.num_classes).to(self.device)
        self.src_prototype = self.get_prototypes(src_loader)

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            print("Alpha : ", self.alpha)
            print("Beta : ", self.beta)

            # source pretraining loop
            # self.pretrain_epoch(src_loader, avg_meter)

            # training loop
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        last_model = self.network.state_dict()

        return last_model, best_model

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        self.feature_extractor.train()
        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            #for params in list(self.classifier.parameters()) + list(self.feature_extractor.parameters()):

            src_feat = self.feature_extractor(src_x)
            src_feat = F.normalize(src_feat, p=2, dim=-1)
            src_pred = self.classifier(src_feat)

            trg_feat = self.feature_extractor(trg_x)
            head = copy.deepcopy(self.classifier)
            for params in list(head.parameters()):
                params.requires_grad = False
            trg_pred = self.softmax(head(trg_feat))
            assert not (torch.isnan(src_feat).any() or torch.isnan(src_pred).any())
            assert not (torch.isnan(trg_feat).any() or torch.isnan(trg_pred).any())



            conf,_ = torch.max(trg_pred, dim=1)

            trg_feat = F.normalize(trg_feat, p=2, dim=-1)
            batch_size = trg_feat.shape[0]

            #update alpha by moving average
            self.alpha = (1 - self.hparams['alpha']) * self.alpha + self.hparams['alpha'] * (conf >= self.hparams['tau1']).sum().item() / conf.size(0)
            self.alpha = max(self.alpha, 1e-3)
            #self.alpha = min(self.alpha, 1-1e-3)
            # get alpha / beta
            match = self.alpha / self.beta
            assert not np.isnan(match)
            #match = max(match, self.alpha+0.001)
            #print("alpha/beta : ", match)

            # update source prototype by moving average
            self.src_prototype = self.src_prototype.detach().cpu() #Else try to re-backprog on previous value
            batch_source_prototype = torch.zeros_like(self.src_prototype)#.to(self.device)
            for i in range(self.num_classes):
                if (src_y == i).sum().item() > 0:
                    batch_source_prototype[i] = (src_feat[src_y == i].mean(dim=0))
                else:
                    batch_source_prototype[i] = (self.src_prototype[i])
            self.src_prototype = (1 - self.hparams["tau"]) * self.src_prototype + self.hparams["tau"] * batch_source_prototype
            self.src_prototype = F.normalize(self.src_prototype, p=2, dim=-1)
            self.src_prototype = self.src_prototype.to(self.device)


            #get ot loss
            #with torch.no_grad():
            a, b =  ot.unif(self.num_classes)+1e-3, ot.unif(batch_size)+1e-3 #TODO ot.unif(self.num_classes), ot.unif(batch_size) #for partial_wasserstein
            m = torch.cdist(self.src_prototype, trg_feat) ** 2 #self.src_prototype
            assert not torch.isnan(m).any()
            m_max = m.max().detach()
            m = m / m_max


            # change to ot.partial.entropic_partial_wasserstein, reg=self.hparams["reg"], m=self.alpha, stopThr=1e-10,log=True
            #pi= ot.partial.partial_wasserstein(a, b, m.detach().cpu().numpy(), self.alpha)
            #pi =  ot.unbalanced.mm_unbalanced(a, b, m.detach().cpu().numpy(), 5, div='l2')
            pi, log = ot.partial.entropic_partial_wasserstein(a, b, m.detach().cpu().numpy(), reg=0.02, m=self.alpha,
                                                                  stopThr=1e-10,log=True)
            pi = torch.from_numpy(pi).float().to(self.device)
            assert not torch.isnan(pi).any() #
            ot_loss = torch.sqrt(torch.sum(pi * m) * m_max)
            loss = self.hparams['ot'] * ot_loss

            '''self.feature_extractor.train()
            self.classifier.train()
            self.optimizer.zero_grad()
            for params in list(self.classifier.parameters()) + list(self.feature_extractor.parameters()):
                params.requires_grad = True'''

            '''src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.softmax(self.classifier(trg_feat))'''

            # update class weight and target weight by plan pi
            plan = pi * batch_size
            k = round(self.hparams['neg']*batch_size) #round(self.hparams['neg'] * batch_size)
            min_dist, _ = torch.min(m, dim=0)
            _, indicate = min_dist.topk(k=k, dim=0)
            batch_class_weight = torch.tensor([plan[i, :].sum() for i in range(self.num_classes)]).to(self.device)
            self.class_weight = self.hparams['tau'] * batch_class_weight + (1 - self.hparams['tau']) * self.class_weight
            self.class_weight = self.class_weight * self.num_classes / self.class_weight.sum()
            k_weight = torch.tensor([plan[:, i].sum() for i in range(batch_size)]).to(self.device)
            k_weight /= self.alpha
            u_weight = torch.zeros(batch_size).to(self.device)
            u_weight[indicate] = 1 - k_weight[indicate]

            # update beta
            self.beta = self.hparams['beta'] * (self.class_weight > self.hparams['tau2']).sum().item() / self.num_classes + (1 - self.hparams['beta']) * self.beta
            self.beta = max(self.beta, 1e-3)#1e-1)
            #self.beta = min(self.beta, 0.999)

            # get classification loss
            cls_loss = F.cross_entropy(src_pred, src_y, weight=self.class_weight.float())
            loss += cls_loss

            # get entropy loss
            p_ent_loss = self.hparams['p_entropy'] * self.entropy_loss(trg_pred, k_weight)
            n_ent_loss = self.hparams['n_entropy'] * self.entropy_loss(trg_pred, u_weight)
            ent_loss = p_ent_loss - n_ent_loss
            loss += ent_loss

            # compute gradient
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #self.lr_scheduler.step()
        #Update Protoypes and Alpha
        self.src_prototype = self.get_prototypes(src_loader)
        self.alpha = self.update_alpha(trg_loader)

        losses = {'Total_loss': loss.item(), 'OT Loss': ot_loss.item(),
                  'Entropic Loss': ent_loss.item(),
                  'Src_cls_loss': cls_loss.item()}

        for key, val in losses.items():
            avg_meter[key].update(val, 32)

    def evaluate(self, test_loader, trg_private_class, src=False):
        feature_extractor = self.feature_extractor.to(self.device)
        classifier = self.classifier.to(self.device)

        feature_extractor.eval()
        classifier.eval()

        total_loss, preds_list, labels_list = [], [], []

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features = self.feature_extractor(data)
                predictions = self.softmax(self.classifier(features))

                features = self.feature_extractor(data)
                predictions = self.softmax(self.classifier(features))
                confidence, pred = predictions.max(dim=1)
                mask = confidence < self.hparams["thresh"]
                if not src:
                    predictions[mask] *= 0

                if self.is_uniDA:
                    mask = labels >= predictions.shape[-1]
                    labels[mask] = predictions.shape[-1]

                mask = labels < predictions.shape[-1]
                # z = torch.zeros((len(predictions), 1))
                # predictions = torch.cat((predictions, z.to(predictions.device)), dim=1)
                loss = F.cross_entropy(predictions[mask], labels[mask])
                total_loss.append(loss.detach().cpu().item())
                # predictions = self.algorithm.correct_predictions(predictions)
                pred = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)
        loss = torch.tensor(total_loss).mean()  # average loss
        full_preds = torch.cat((preds_list))
        full_labels = torch.cat((labels_list))
        return loss, full_preds, full_labels
    def decision_function(self, preds):
        mask = preds.sum(axis=1) == 0.0
        confidence, pred = preds.max(dim=1)
        pred[mask] = -1
        return pred

class UniOT(Algorithm):
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        print(configs)
        # device
        self.configs = configs
        final_out_channels = self.configs.final_out_channels
        if self.configs.isFNO:
            final_out_channels = self.configs.final_out_channels * 2
        self.device = device
        self.feature_extractor = backbone(configs).to(self.device)
        self.classifier = CLS(configs, temp=hparams['temp']).to(self.device)

        self.cluster_head = ProtoCLS(final_out_channels, hparams['K'], temp=hparams['temp']).to(self.device)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # hparams
        self.hparams = hparams
        self.nb_classes = configs.num_classes


        # initialize the gamma (coupling in OT) with zeros
        self.gamma = torch.zeros(hparams["batch_size"],
                                 hparams["batch_size"])  # .dnn.K.zeros(shape=(self.batch_size, self.batch_size))
        self.gamma.to(self.device)


        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer_feat = torch.optim.Adam(
            self.feature_extractor.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer_cls = torch.optim.Adam(
            self.classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer_cluhead = torch.optim.Adam(
            self.cluster_head.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.n_batch = int(hparams['MQ_size']/hparams['batch_size'])
        feat_dim = configs.features_len * final_out_channels


        self.memqueue = MemoryQueue(feat_dim, hparams['batch_size'], self.n_batch, hparams['temp']).cuda()
        self.beta = None
        self.softmax = torch.nn.Softmax(dim=1)
        self.bce = BCELoss()
        self.is_uniDA = True
        self.t = 0.5


    def init_queue(self, dataloader):
        cnt_i = 0
        while cnt_i < self.n_batch:
            for x,y, id in dataloader:
                x, y, id = x.to(self.device), y.to(self.device), id.to(self.device)
                feats = self.feature_extractor(x)
                proto, preds = self.classifier(feats)
                self.memqueue.update_queue(F.normalize(proto), id)
                cnt_i += 1
                if cnt_i > self.n_batch - 1:
                    break

    def update(self, src_loader, trg_loader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        nb_pr_epochs = self.hparams["num_epochs_pr"]
        for epoch in range(1, nb_pr_epochs + 1):
            self.pretrain_epoch(src_loader, avg_meter)

            logger.debug(f'[Pr Epoch : {epoch}/{nb_pr_epochs}]')  # TODO : self.hparams["num_pr_epochs"]
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        with torch.no_grad():
            self.network.eval()
            X = src_loader.dataset.x_data.cuda()
            Y = src_loader.dataset.y_data.numpy()
            _, logits = self.network(X)
            preds = logits.detach().cpu().argmax(axis=1).numpy()
            print("SRC Accuracy : ", (Y == preds).sum() / len(Y))
        self.network.train()
        self.init_queue(trg_loader)
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # source pretraining loop
            # self.pretrain_epoch(src_loader, avg_meter)

            # training loop
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        last_model = self.network.state_dict()

        return last_model, best_model

    def pretrain_epoch(self, src_loader, avg_meter):

        for src_x, src_y, _ in src_loader:
            src_x, src_y = src_x.to(self.device), src_y.to(self.device)

            src_feat = self.feature_extractor(src_x)
            _, src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)

            loss = src_cls_loss

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            losses = {'Pr_Src_cls_loss': loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)


    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders
        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))
        temp = self.hparams['temp']
        #soft = nn.Softmax(dim=1)
        for step, ((src_x, src_y, id_source), (trg_x, _, id_target)) in joint_loader:
            """if src_x.shape[0] != trg_x.shape[0]:
                continue"""

            if src_x.shape[0] > trg_x.shape[0]:
                src_x = src_x[:trg_x.shape[0]]
                src_y = src_y[:trg_x.shape[0]]
            elif trg_x.shape[0] > src_x.shape[0]:
                trg_x = trg_x[:src_x.shape[0]]

            batch_size = len(src_x)
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(
                self.device)  # extract source features

            feature_ex_s = self.feature_extractor(src_x)
            feature_ex_t = self.feature_extractor(trg_x)

            before_lincls_feat_s, after_lincls_s = self.classifier(feature_ex_s)
            before_lincls_feat_t, after_lincls_t = self.classifier(feature_ex_t)

            #norm_feat_s = F.normalize(before_lincls_feat_s)
            norm_feat_t = F.normalize(before_lincls_feat_t)

            after_cluhead_t = self.cluster_head(before_lincls_feat_t)

            # =====Source Supervision=====
            criterion = nn.CrossEntropyLoss().cuda()
            loss_cls = criterion(after_lincls_s, src_y)

            # =====Private Class Discovery=====
            minibatch_size = norm_feat_t.size(0)

            # obtain nearest neighbor from memory queue and current mini-batch
            feat_mat2 = torch.matmul(norm_feat_t, norm_feat_t.t()) / temp
            mask = torch.eye(feat_mat2.size(0), feat_mat2.size(0)).bool().cuda()
            feat_mat2.masked_fill_(mask, -1 / temp)

            nb_value_tt, nb_feat_tt = self.memqueue.get_nearest_neighbor(norm_feat_t, id_target.cuda())
            neighbor_candidate_sim = torch.cat([nb_value_tt.reshape(-1, 1), feat_mat2], 1)
            values, indices = torch.max(neighbor_candidate_sim, 1)
            neighbor_norm_feat = torch.zeros((minibatch_size, norm_feat_t.shape[1])).cuda()
            for i in range(minibatch_size):
                neighbor_candidate_feat = torch.cat([nb_feat_tt[i].reshape(1, -1), norm_feat_t], 0)
                neighbor_norm_feat[i, :] = neighbor_candidate_feat[indices[i], :]

            neighbor_output = self.cluster_head(neighbor_norm_feat)

            # fill input features with memory queue
            fill_size_ot = self.hparams['K']
            mqfill_feat_t = self.memqueue.random_sample(fill_size_ot)
            mqfill_output_t = self.cluster_head(mqfill_feat_t)

            # OT process
            # mini-batch feat (anchor) | neighbor feat | filled feat (sampled from memory queue)
            S_tt = torch.cat([after_cluhead_t, neighbor_output, mqfill_output_t], 0)
            #print(mqfill_output_t.shape, after_cluhead_t.shape, neighbor_output.shape, S_tt.shape)
            S_tt *= temp
            Q_tt = sinkhorn(S_tt.detach(), epsilon=0.05, sinkhorn_iterations=3)
            Q_tt_tilde = Q_tt * Q_tt.size(0)
            anchor_Q = Q_tt_tilde[:minibatch_size, :]
            neighbor_Q = Q_tt_tilde[minibatch_size:2 * minibatch_size, :]

            # compute loss_PCD
            loss_local = 0
            for i in range(minibatch_size):
                sub_loss_local = 0
                sub_loss_local += -torch.sum(neighbor_Q[i, :] * F.log_softmax(after_cluhead_t[i, :]))
                sub_loss_local += -torch.sum(anchor_Q[i, :] * F.log_softmax(neighbor_output[i, :]))
                sub_loss_local /= 2
                loss_local += sub_loss_local
            loss_local /= minibatch_size
            loss_global = -torch.mean(torch.sum(anchor_Q * F.log_softmax(after_cluhead_t, dim=1), dim=1))
            loss_PCD = (loss_global + loss_local) / 2

            # =====Common Class Detection=====
            #if global_step > 100:
            source_prototype = self.classifier.ProtoCLS.fc.weight
            if self.beta is None:
                self.beta = ot.unif(source_prototype.size()[0])

            # fill input features with memory queue
            fill_size_uot = self.n_batch * batch_size
            mqfill_feat_t = self.memqueue.random_sample(fill_size_uot)
            ubot_feature_t = torch.cat([mqfill_feat_t, norm_feat_t], 0)
            #full_size = ubot_feature_t.size(0)

            # Adaptive filling
            newsim, fake_size = adaptive_filling(ubot_feature_t, source_prototype, self.hparams['gamma'], self.beta, fill_size_uot)
            #newsim = torch.matmul(ubot_feature_t, source_prototype.t())
            #fake_size = 0

            # UOT-based CCD
            high_conf_label_id, high_conf_label, _, new_beta = ubot_CCD(newsim, self.beta, fake_size=fake_size,
                                                                        fill_size=fill_size_uot, mode='minibatch')
            # adaptive update for marginal probability vector
            self.beta = self.hparams['mu'] * self.beta + (1 - self.hparams['mu']) * new_beta

            # fix the bug raised in https://github.com/changwxx/UniOT-for-UniDA/issues/1
            # Due to mini-batch sampling, current mini-batch samples might be all target-private.
            # (especially when target-private samples dominate target domain, e.g. OfficeHome)
            if high_conf_label_id.size(0) > 0:
                loss_CCD = criterion(after_lincls_t[high_conf_label_id, :], high_conf_label[high_conf_label_id])
            else:
                loss_CCD = 0

            loss_all = loss_cls + self.hparams['lam'] * (loss_CCD)

            self.optimizer_feat.zero_grad()
            self.optimizer_cls.zero_grad()
            self.optimizer_cluhead.zero_grad()
            loss_all.backward()
            self.optimizer_feat.step()
            self.optimizer_cls.step()
            self.optimizer_cluhead.step()

            self.classifier.ProtoCLS.weight_norm()  # very important for proto-classifier
            self.cluster_head.weight_norm()  # very important for proto-classifier
            self.memqueue.update_queue(norm_feat_t, id_target.cuda())

            losses = {'Total_loss': loss_all.item(), 'loss_cls': loss_cls.item(),
                      'loss_PCD': loss_PCD.item(),
                      'loss_CCD': loss_CCD}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

    def evaluate(self, test_loader, trg_private_class, src=False):
        feature_extractor = self.feature_extractor.to(self.device)
        classifier = self.classifier.to(self.device)

        feature_extractor.eval()
        classifier.eval()

        total_loss, preds_list, labels_list = [], [], []
        norm_feat_t_list = []

        with torch.no_grad():
            for data, labels, id in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features = feature_extractor(data)
                before_lincls_feat_t, predictions = classifier(features)
                norm_feat_t = F.normalize(before_lincls_feat_t)

                if self.is_uniDA and not src:
                    mask = labels < predictions.shape[-1]
                    loss = F.cross_entropy(predictions[mask], labels[mask])
                    total_loss.append(loss.detach().cpu().item())
                else:
                    loss = F.cross_entropy(predictions, labels)
                    total_loss.append(loss.detach().cpu().item())
                #predictions = self.algorithm.correct_predictions(predictions)
                pred = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)
                norm_feat_t_list.append(norm_feat_t)
        loss = torch.tensor(total_loss).mean()  # average loss
        full_preds = torch.cat((preds_list))
        full_labels = torch.cat((labels_list))
        norm_feat_t = torch.cat((norm_feat_t_list))

        source_prototype = classifier.ProtoCLS.fc.weight

        stopThr = 1e-6
        # Adaptive filling
        newsim, fake_size = adaptive_filling(norm_feat_t.cuda(),
                                             source_prototype, self.hparams['gamma'], self.beta, 0, stopThr=stopThr)

        # obtain predict label
        _, __, pred_label, ___ = ubot_CCD(newsim, self.beta, fake_size=fake_size, fill_size=0, mode='minibatch',
                                          stopThr=stopThr)
        pred_label = pred_label.cpu().data.numpy()
        mask = pred_label == self.nb_classes
        full_preds[mask] *= 0

        return loss, full_preds, full_labels

    def get_latent_features(self, dataloader):
        feature_set = []
        label_set = []
        logits_set = []
        self.feature_extractor.eval()
        self.classifier.eval()
        with torch.no_grad():
            for _, (data, label, id) in enumerate(dataloader):
                data = data.to(self.device)
                feature = self.feature_extractor(data)
                _, logit = self.classifier(feature)
                feature_set.append(feature.cpu())
                label_set.append(label.cpu())
                logits_set.append(logit.cpu())
            feature_set = torch.cat(feature_set, dim=0)
            feature_set = F.normalize(feature_set, p=2, dim=-1)
            label_set = torch.cat(label_set, dim=0)
            logits_set = torch.cat(logits_set, dim=0)
        return feature_set, label_set, logits_set

    def decision_function(self, preds):
        mask = preds.sum(axis=1) == 0.0
        confidence, pred = preds.max(dim=1)
        pred[mask] = -1
        return pred

class UniJDOT(Algorithm):
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        print(configs)
        # device
        self.device = device
        self.feature_extractor = backbone(configs).to(self.device)
        #self.classifier = CLS(configs, temp=hparams['temp']).to(self.device)
        self.classifier = CLS(configs).to(self.device)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # hparams
        self.hparams = hparams
        print("Batch Size : ", hparams['batch_size'])
        self.nb_classes = configs.num_classes

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer_feat = torch.optim.Adam(
            self.feature_extractor.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer_cls = torch.optim.Adam(
            self.classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.beta = None
        self.softmax = torch.nn.Softmax(dim=1)
        self.bce = BCELoss()
        self.is_uniDA = True
        self.src_latent_cluster = None
        self.register_buffer("final_threshold", torch.tensor(0.0))

        #self.final_threshold = None
        self.final_threshold = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)

        self.threshold_method = self.get_thresholding_method() #sfil.threshold_yen #sfil.threshold_triangle #sfil.threshold_yen
        #self.threshold_method = self.v
        #self.threshold_method = sfil.threshold_yen  # sfil.threshold_triangle #sfil.threshold_yen
        if configs.isFNO:
            feat_dim = configs.final_out_channels + 2*configs.fourier_modes
        else:
            feat_dim = configs.final_out_channels #+ 2 * configs.fourier_modes
        self.memqueue_feat = ClassMemoryQueue(feat_dim, self.nb_classes, hparams['n_batch']).cuda()
        #self.memqueue_preds = MemoryQueue(configs.num_classes, hparams['batch_size'], hparams['n_batch']).cuda()

    '''def v(self, x):
        return 0.5 * sfil.threshold_yen(x) + 0.5 * sfil.threshold_otsu(x)'''
    def init_queue(self, dataloader):
        cnt_i = 0
        for x,y, id in dataloader:
            x, y, id = x.to(self.device), y.to(self.device), id.to(self.device)
            feature_ex_s = self.feature_extractor(x)
            before_lincls_feat_s, after_lincls_s = self.classifier(feature_ex_s)
            self.memqueue_feat.update_queue(F.normalize(before_lincls_feat_s), y)
            #self.memqueue_preds.update_queue(y_src, id.clone())
            cnt_i += 1
            if self.memqueue_feat.is_memory_full().all():
                break
        print('Memory after init : ', self.memqueue_feat.is_memory_full())

    def infomax_loss(self, cluster_assignments, eps=1e-10):
        """
        InfoMax loss function to maximize mutual information between inputs and cluster assignments.

        Args:
            cluster_assignments (torch.Tensor): Tensor of shape (N, K) representing the probability distribution
                                                over K clusters for each sample.
            eps (float): Small constant to avoid numerical issues with log.

        Returns:
            torch.Tensor: Scalar loss value (InfoMax loss).
        """
        # Batch size and number of clusters
        N, K = cluster_assignments.shape

        # Step 1: Compute the marginal distribution p(z) over clusters (averaging over samples)
        marginal_prob = cluster_assignments.mean(dim=0)  # Shape (K,)

        # Step 2: Compute H(Z) - Entropy of the marginal distribution (cluster assignments)
        H_Z = -torch.sum(marginal_prob * torch.log(marginal_prob + eps))

        # Step 3: Compute H(Z|X) - Conditional entropy of cluster assignment given input
        H_Z_given_X = -torch.sum(cluster_assignments * torch.log(cluster_assignments + eps)) / N

        # Step 4: InfoMax loss is H(Z) - H(Z|X)
        infomax_loss_value = H_Z - H_Z_given_X

        return infomax_loss_value

    def get_thresholding_method(self):
        return getattr(sfil, self.hparams['threshold_method'])

    '''def threshold_method(self, x):
        if sfil.threshold_yen(x) < 1-1/self.nb_classes:
            return sfil.threshold_yen(x[x>1-1/self.nb_classes])
        return max(sfil.threshold_yen(x), 1-1/self.nb_classes)'''

    def class_centroids(self, x, y):
        # Get the number of classes

        # Reduce the sum of each feature across samples within a class
        class_sums = torch.einsum("ji,jk->ki", x, y.float().cuda())
        # return class_sums
        # Count the number of samples in each class (sum along the sample dimension)
        class_counts = torch.sum(y, dim=0)

        # Avoid division by zero for empty classes
        class_counts[class_counts == 0] = 1

        # Divide class sums by class counts to get centroids
        centroids = (class_sums.T / class_counts).T

        return centroids

    def ini_centroids(self, src_dl):
        with TrainingModeManager([self.feature_extractor, self.classifier], train=False) as mgr, \
                Accumulator(['ctr']) as eval_accumulator, \
                torch.no_grad():
            for i, (im_s, label_source, id_s) in enumerate(tqdm(src_dl, desc='testing')):
                #print(im_s.shape)
                im_s = im_s.cuda()
                label_source = label_source.cuda()
                feature_ex_s = self.feature_extractor.forward(im_s)
                before_lincls_feat_s, after_lincls_t = self.classifier(feature_ex_s)
                norm_feat_s = F.normalize(before_lincls_feat_s)
                y_src = torch.eye(self.nb_classes, dtype=torch.int8).cuda()
                y_src = torch.eye(self.nb_classes, dtype=torch.int8).cuda()[label_source]
                ctr = self.class_centroids(norm_feat_s, y_src).unsqueeze(0)
                val = dict()
                for name in eval_accumulator.names:
                    val[name] = locals()[name].cpu().data.numpy()
                eval_accumulator.updateData(val)

        for x in eval_accumulator:
            val[x] = eval_accumulator[x]
        ctr = val['ctr']
        del val
        return torch.tensor(ctr.mean(axis=0))

    def centroids_target(self, trg_dl, K):
        X = []
        with TrainingModeManager([self.feature_extractor], train=False) as mgr, \
                torch.no_grad():
            for i, (im_t, label_target, id_t) in enumerate(tqdm(trg_dl, desc='testing')):
                im_t = im_t.cuda()
                feature_ex_t = self.feature_extractor.forward(im_t)
                before_lincls_feat_t, after_lincls_t = self.classifier(feature_ex_t)
                # norm_feat_t = F.normalize(before_lincls_feat_t)
                norm_feat_t = F.normalize(before_lincls_feat_t)
                X.append(norm_feat_t)
        X = torch.cat((X), 0).cpu()  # .numpy()

        kmeans = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(X)
        return kmeans.cluster_centers_

    def update_centroids_target(self, X, cen):
        X = X.cpu()
        cen = cen.cpu()
        dd = torch.cdist(X, cen)
        pred_cen = dd.argmin(axis=1)

        for i in pred_cen:
            ix = i == pred_cen
            cenc = X[ix].mean(axis=0)
            cen[i] = 0.9 * cen[i] + 0.1 * cenc

        return cen

    def update(self, src_loader, trg_loader, avg_meter, logger):
        self.src_loader = src_loader
        self.init_queue(src_loader)
        print("Memory State : ", self.memqueue_feat.is_memory_full())
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        self.src_latent_cluster = self.ini_centroids(src_loader).cuda()
        self.trg_latent_cluster = self.centroids_target(trg_loader, self.hparams['K'])
        self.trg_latent_cluster = torch.from_numpy(self.trg_latent_cluster).to(torch.float)

        nb_pr_epochs = self.hparams["num_epochs_pr"]
        for epoch in range(1, nb_pr_epochs + 1):
            self.pretrain_epoch(src_loader, avg_meter)

            logger.debug(f'[Pr Epoch : {epoch}/{nb_pr_epochs}]')  # TODO : self.hparams["num_pr_epochs"]
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        with torch.no_grad():
            self.network.eval()
            X = self.src_loader.dataset.x_data.cuda()
            Y = self.src_loader.dataset.y_data.numpy()
            _, logits = self.network(X)
            preds = logits.detach().cpu().argmax(axis=1).numpy()
            print("SRC Accuracy : ", (Y == preds).sum() / len(Y))
        self.network.train()
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # source pretraining loop
            # self.pretrain_epoch(src_loader, avg_meter)

            # training loop
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        cnt_i = 0
        self.trg_feats_mem = []
        self.trg_preds_mem = []
        trg_mem_size = self.hparams["trg_mem_size"]
        with torch.no_grad():
            self.network.eval()
            for x, y, id in trg_loader:
                x, y, id = x.to(self.device), y.to(self.device), id.to(self.device)
                feature_ex_t = self.feature_extractor(x)
                before_lincls_feat_t, after_lincls_s = self.classifier(feature_ex_t)
                norm_feat_t = F.normalize(before_lincls_feat_t)
                self.trg_feats_mem.append(norm_feat_t)
                self.trg_preds_mem.append(after_lincls_s)
                # self.memqueue_preds.update_queue(y_src, id.clone())
                cnt_i += after_lincls_s.shape[0]
                if cnt_i > trg_mem_size:
                    break
        self.trg_feats_mem = torch.concatenate(self.trg_feats_mem)[:trg_mem_size]
        self.trg_preds_mem = torch.concatenate(self.trg_preds_mem)[:trg_mem_size]

        if self.hparams['joint_decision']:
            dist_trg_tr = self.compute_cluster_distance(self.trg_feats_mem)
            soft_trg_tr = self.joint_decision(self.trg_preds_mem, dist_trg_tr)
        else:
            soft_trg_tr = F.softmax(self.trg_preds_mem, dim=1)
        conf, preds = soft_trg_tr.max(dim=1)
        #self.final_threshold = self.threshold_method(conf.detach().cpu().numpy())
        new_value = self.threshold_method(conf.detach().cpu().numpy())
        self.final_threshold.data.fill_(new_value)
        #self.final_threshold.data = torch.tensor(new_value, device=self.final_threshold.device)

        #self.register_buffer("final_threshold", torch.tensor(self.final_threshold))
        #self.final_threshold = torch.tensor(self.final_threshold, device=self.final_threshold.device)

        '''X = self.src_loader.dataset.x_data.cuda()
        Y = self.src_loader.dataset.y_data.numpy()
        _, logits = self.network(X)
        preds = logits.detach().cpu().argmax(axis=1).numpy()
        print("SRC Accuracy : ", (Y == preds).sum() / len(Y))'''
        last_model = self.network.state_dict()

        return last_model, best_model

    def pretrain_epoch(self, src_loader, avg_meter):

        for src_x, src_y, _ in src_loader:
            src_x, src_y = src_x.to(self.device), src_y.to(self.device)

            src_feat = self.feature_extractor(src_x)
            _, src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)
            #info_loss = self.infomax_loss(F.softmax(src_pred))

            loss = src_cls_loss #+ info_loss

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            losses = {'Pr_Src_cls_loss': loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

    def compute_cluster_distance(self, x_test):
        #x = self.src_loader.dataset.x_data
        #y = self.src_loader.dataset.y_data
        x = self.memqueue_feat.mem_feat
        '''y = self.memqueue_feat.mem_id
        x, x_test, y = torch.Tensor(x).cuda(), torch.Tensor(x_test), torch.Tensor(y).cuda().long()'''

        #x = F.normalize(before_lincls_feat_t)
        x, x_test = x.squeeze(), x_test.squeeze()
        '''nb_classes = self.nb_classes
        res = torch.empty(x_test.shape[0], nb_classes)
        res = torch.zeros_like(res)
        print(y.unique(return_counts=True))
        for i, ll in enumerate(range(nb_classes)):
            dist = torch.cdist(x[y == ll], x_test).min(axis=0).values
            res[:, int(ll)] = dist'''
        if len(x_test.shape) == 1:
            x_test = x_test.unsqueeze(0)
        res = self.memqueue_feat.compute_distances(x_test)
        #print(res)

        # res = F.tanh(res)
        # res = res/res.max()
        d = -1 * res
        # d = 1-res
        d = F.softmax(d, dim=1)

        return d

    def compute_cluster_distance2(self, x_test):
        x = self.src_loader.dataset.x_data
        y = self.src_loader.dataset.y_data
        x, x_test, y = torch.Tensor(x).cuda(), torch.Tensor(x_test), torch.Tensor(y).cuda().long()
        feature_ex_t = self.feature_extractor(x)
        before_lincls_feat_t, after_lincls_t = self.classifier(feature_ex_t)

        x = F.normalize(before_lincls_feat_t)
        # _, x, _ = net.forward_extractor_logits(x)
        x, x_test = x.squeeze(), x_test.squeeze()
        nb_classes = self.nb_classes
        res = torch.empty(x_test.shape[0], nb_classes)
        res = torch.zeros_like(res)
        for i, ll in enumerate(range(nb_classes)):
            dist = torch.cdist(x[y == ll], x_test).min(axis=0).values
            res[:, int(ll)] = dist

        d = -1 * res
        d = F.softmax(d, dim=1)
        #d = 1-res/res.max()
        return d.max(axis=0).values

    def joint_decision(self, preds, distance):
        preds = torch.tensor(preds).cuda()
        distance = distance.cuda()
        return F.softmax(preds*distance, dim=1)#* distance
        #return F.softmax(F.softmax(preds, dim=1) * distance, dim=1)  # * distance
        #return 0.5*F.softmax(preds, dim=1) + 0.5*distance # * distance
    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders
        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))
        #temp = self.hparams['temp']
        # soft = nn.Softmax(dim=1)
        for step, ((src_x, src_y, id_source), (trg_x, _, id_target)) in joint_loader:
            """if src_x.shape[0] != trg_x.shape[0]:
                continue"""

            if src_x.shape[0] > trg_x.shape[0]:
                src_x = src_x[:trg_x.shape[0]]
                src_y = src_y[:trg_x.shape[0]]
            elif trg_x.shape[0] > src_x.shape[0]:
                trg_x = trg_x[:src_x.shape[0]]

            batch_size = len(src_x)
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(
                self.device)  # extract source features
            feature_ex_s = self.feature_extractor(src_x)
            feature_ex_t = self.feature_extractor(trg_x)

            before_lincls_feat_s, after_lincls_s = self.classifier(feature_ex_s)
            before_lincls_feat_t, after_lincls_t = self.classifier(feature_ex_t)

            norm_feat_s = F.normalize(before_lincls_feat_s)
            norm_feat_t = F.normalize(before_lincls_feat_t)

            # =====Source Supervision=====

            y_src = torch.eye(after_lincls_s.shape[-1], dtype=torch.int8).cuda()[src_y]
            self.memqueue_feat.update_queue(norm_feat_s, src_y.cuda())
            #self.memqueue_preds.update_queue(y_src, id_source.cuda())
            #print(y_src.shape)
            # print("y_src shape : ", y_src.shape)

            centr = self.class_centroids(norm_feat_s, y_src).detach().cpu()
            src_latent_cluster_copy = 0.9 * self.src_latent_cluster + 0.1 * centr.cuda()
            # TRG Centroids
            trg_latent_cluster_copy = self.update_centroids_target(norm_feat_t.detach().cpu(), self.trg_latent_cluster).cuda()
            before_lincls_feat_cen, after_lincls_cen = self.classifier(trg_latent_cluster_copy)
            trg_soft_cen = torch.nn.functional.softmax(after_lincls_cen).double()

            if self.hparams['joint_decision']:
                dist = self.compute_cluster_distance(norm_feat_t)
                soft_t = self.joint_decision(after_lincls_t, dist)
            else:
                soft_t = F.softmax(after_lincls_t)
            conf, preds = soft_t.max(dim=1)
            threshold = self.threshold_method(conf.detach().cpu().numpy())
            if step == 1: print('threshold : ', threshold)
            mask = (conf < threshold).cuda()
            # print("Detected ODD : ", mask.sum().item())

            # print("Number of ODD : ", (trg_label >= cls_output_dim).sum().item())
            # C0 = torch.zeros((len(norm_feat_s) + 1, len(norm_feat_t))).cuda()

            C0 = torch.zeros((len(norm_feat_s) + len(trg_latent_cluster_copy), len(norm_feat_t))).cuda()
            C_latent = src_latent_cluster_copy.mean(axis=0).unsqueeze(0)

            # nonOOD SRC
            C0[:len(norm_feat_s), ~mask] = torch.cdist(norm_feat_s, norm_feat_t[~mask])
            # OOD Dummy
            C0[len(norm_feat_s):, mask] = torch.cdist(trg_latent_cluster_copy, norm_feat_t[mask])

            maxc = torch.max(C0).item()  # *self.hparams['psi']
            # ODD SRC
            C0[:len(norm_feat_s), mask] = maxc
            # nonOOD Dummy
            C0[len(norm_feat_s):, ~mask] = maxc

            C1 = torch.zeros(C0.shape).cuda()
            C_preds = torch.ones((trg_latent_cluster_copy.shape[0], y_src.shape[-1])) / self.nb_classes
            C_preds = C_preds.cuda()

            # nonOOD SRC
            C1[:len(norm_feat_s), ~mask] = torch.cdist(y_src.float(), F.softmax(after_lincls_t)[~mask])
            # OOD Dummy
            C1[len(norm_feat_s):, mask] = torch.cdist(C_preds, F.softmax(after_lincls_t)[mask])
            maxc = torch.max(C1).item()  # *self.hparams['psi']
            # ODD SRC
            C1[:len(norm_feat_s), mask] = maxc
            # nonOOD Dummy
            C1[len(norm_feat_s):, ~mask] = maxc

            C = (self.hparams['alpha'] * C0 + self.hparams['lamb'] * C1)

            with torch.no_grad():

                a, b = ot.unif(C.size(0)), ot.unif(C.size(1))
                ratio = (mask.sum() / len(mask)).detach().cpu().item()
                a[:len(norm_feat_s)] = 0.5 / len(a[:len(norm_feat_s)])  #
                # a[:len(norm_feat_s)] = (1-ratio)/len(a[:len(norm_feat_s)])
                a[len(norm_feat_s):] = 0.5 / len(a[len(norm_feat_s):])  #
                # a[len(norm_feat_s):] = ratio/len(a[len(norm_feat_s):])
                # print(a.sum(), b.sum())

                gamma = ot.unbalanced.mm_unbalanced(a, b, C.detach().cpu().numpy(), reg_m=0.5)
                # gamma = ot.partial.partial_wasserstein(a, b, C.detach().cpu().numpy(), m=temp)
                # gamma = ot.sinkhorn(a, b, C.detach().cpu().numpy(), reg=0.01)
                # gamma = ot.emd(a, b, C.detach().cpu().numpy())
                # mass.append(gamma.sum())
                if step == 1:print('Mass : ', gamma.sum())
                gamma = torch.tensor(gamma).cuda()

            assert not torch.isnan(gamma).any()
            assert not torch.isnan(feature_ex_t).any()
            assert not torch.isnan(feature_ex_s).any()
            assert not torch.isnan(after_lincls_t).any()
            assert not torch.isnan(after_lincls_s).any()

            label_align_loss = (C * gamma)  # .sum()

            label_align_loss_nonOOD = label_align_loss[:len(norm_feat_s), ~mask].sum()
            label_align_loss_OOD = label_align_loss[len(norm_feat_s):, mask].sum()

            #label_align_loss = self.hparams['lamb'] * (label_align_loss_nonOOD + label_align_loss_OOD) / 2
            label_align_loss = (label_align_loss_nonOOD + label_align_loss_OOD) / 2

            criterion = nn.CrossEntropyLoss().cuda()
            loss_cls = self.hparams['src_weight'] * criterion(after_lincls_s, src_y)

            loss_all = loss_cls + label_align_loss #+ info_loss

            self.optimizer_feat.zero_grad()
            self.optimizer_cls.zero_grad()
            loss_all.backward()
            self.optimizer_feat.step()
            self.optimizer_cls.step()

            self.classifier.ProtoCLS.weight_norm()  # very important for proto-classifier

            losses = {'Total_loss': loss_all.item(), 'loss_cls': loss_cls.item(),
                      'loss_align': label_align_loss.item()}#, 'info_loss': info_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

    def evaluate(self, test_loader, trg_private_class, src=False):
        self.feature_extractor.eval()
        self.classifier.eval()

        total_loss, preds_list, labels_list = [], [], []



        '''dist_trg_tr = self.compute_cluster_distance(self.trg_feats_mem)
        soft_trg_tr = self.joint_decision(self.trg_preds_mem, dist_trg_tr)
        conf, preds = soft_trg_tr.max(dim=1)
        threshold = self.threshold_method(conf.detach().cpu().numpy())'''

        with torch.no_grad():
            for data, labels, ids in test_loader:
                data = data.float().to(self.device)
                '''if data.shape[0] == 1:
                    continue'''
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features = self.feature_extractor(data)
                before_lincls_feat_t, predictions = self.classifier(features)
                soft = F.softmax(predictions)

                if self.hparams['joint_decision']:
                    norm_feat_t = F.normalize(before_lincls_feat_t)
                    dist = self.compute_cluster_distance(norm_feat_t)
                    soft = self.joint_decision(predictions, dist)
                else:
                    soft = F.softmax(predictions)

                # preds_t = soft_t.argmax(dim=1)
                conf, preds = soft.max(dim=1)
                #threshold = self.threshold_method(conf.detach().cpu().numpy())
                #print("Finale Threshold : ", threshold)
                #print(self.final_threshold)
                mask = conf < self.final_threshold #threshold
                if not src:
                    predictions[mask.squeeze()] *= 0

                if self.is_uniDA:
                    mask = labels >= predictions.shape[-1]
                    labels[mask] = predictions.shape[-1]

                mask = labels < predictions.shape[-1]
                # z = torch.zeros((len(predictions), 1))
                # predictions = torch.cat((predictions, z.to(predictions.device)), dim=1)
                loss = F.cross_entropy(predictions[mask], labels[mask])
                total_loss.append(loss.detach().cpu().item())
                # predictions = self.algorithm.correct_predictions(predictions)
                pred = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)

        loss = torch.tensor(total_loss).mean()  # average loss
        full_preds = torch.cat((preds_list))
        full_labels = torch.cat((labels_list))
        return loss, full_preds, full_labels

    def get_latent_features(self, dataloader):
        feature_set = []
        label_set = []
        logits_set = []
        self.feature_extractor.eval()
        self.classifier.eval()
        with torch.no_grad():
            for _, (data, label, ids) in enumerate(dataloader):
                data = data.to(self.device)
                '''if data.shape[0] == 1:
                    continue'''
                feature = self.feature_extractor(data)
                _, logit = self.classifier(feature)
                feature_set.append(feature.cpu())
                label_set.append(label.cpu())
                logits_set.append(logit.cpu())
            feature_set = torch.cat(feature_set, dim=0)
            feature_set = F.normalize(feature_set, p=2, dim=-1)
            label_set = torch.cat(label_set, dim=0)
            logits_set = torch.cat(logits_set, dim=0)
        return feature_set, label_set, logits_set

    def decision_function(self, preds):
        mask = preds.sum(axis=1) == 0.0
        confidence, pred = preds.max(dim=1)
        pred[mask] = -1
        return pred

class UniJDOT_NoJoint(Algorithm):
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        print(configs)
        # device
        self.device = device
        self.feature_extractor = backbone(configs).to(self.device)
        #self.classifier = CLS(configs, temp=hparams['temp']).to(self.device)
        self.classifier = CLS(configs).to(self.device)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # hparams
        self.hparams = hparams
        print("Batch Size : ", hparams['batch_size'])
        self.nb_classes = configs.num_classes

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer_feat = torch.optim.Adam(
            self.feature_extractor.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer_cls = torch.optim.Adam(
            self.classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.beta = None
        self.softmax = torch.nn.Softmax(dim=1)
        self.bce = BCELoss()
        self.is_uniDA = True
        self.src_latent_cluster = None


        self.threshold_method = self.get_thresholding_method() #sfil.threshold_yen #sfil.threshold_triangle #sfil.threshold_yen
        #self.threshold_method = self.v
        #self.threshold_method = sfil.threshold_yen  # sfil.threshold_triangle #sfil.threshold_yen
        if configs.isFNO:
            feat_dim = configs.final_out_channels + 2*configs.fourier_modes
        else:
            feat_dim = configs.final_out_channels #+ 2 * configs.fourier_modes
        self.memqueue_feat = ClassMemoryQueue(feat_dim, self.nb_classes, hparams['n_batch']).cuda()
        #self.memqueue_preds = MemoryQueue(configs.num_classes, hparams['batch_size'], hparams['n_batch']).cuda()

    '''def v(self, x):
        return 0.5 * sfil.threshold_yen(x) + 0.5 * sfil.threshold_otsu(x)'''
    def init_queue(self, dataloader):
        cnt_i = 0
        for x,y, id in dataloader:
            x, y, id = x.to(self.device), y.to(self.device), id.to(self.device)
            feature_ex_s = self.feature_extractor(x)
            before_lincls_feat_s, after_lincls_s = self.classifier(feature_ex_s)
            self.memqueue_feat.update_queue(F.normalize(before_lincls_feat_s), y)
            #self.memqueue_preds.update_queue(y_src, id.clone())
            cnt_i += 1
            if self.memqueue_feat.is_memory_full().all():
                break
        print('Memory after init : ', self.memqueue_feat.is_memory_full())

    def infomax_loss(self, cluster_assignments, eps=1e-10):
        """
        InfoMax loss function to maximize mutual information between inputs and cluster assignments.

        Args:
            cluster_assignments (torch.Tensor): Tensor of shape (N, K) representing the probability distribution
                                                over K clusters for each sample.
            eps (float): Small constant to avoid numerical issues with log.

        Returns:
            torch.Tensor: Scalar loss value (InfoMax loss).
        """
        # Batch size and number of clusters
        N, K = cluster_assignments.shape

        # Step 1: Compute the marginal distribution p(z) over clusters (averaging over samples)
        marginal_prob = cluster_assignments.mean(dim=0)  # Shape (K,)

        # Step 2: Compute H(Z) - Entropy of the marginal distribution (cluster assignments)
        H_Z = -torch.sum(marginal_prob * torch.log(marginal_prob + eps))

        # Step 3: Compute H(Z|X) - Conditional entropy of cluster assignment given input
        H_Z_given_X = -torch.sum(cluster_assignments * torch.log(cluster_assignments + eps)) / N

        # Step 4: InfoMax loss is H(Z) - H(Z|X)
        infomax_loss_value = H_Z - H_Z_given_X

        return infomax_loss_value

    def get_thresholding_method(self):
        return getattr(sfil, self.hparams['threshold_method'])

    '''def threshold_method(self, x):
        if sfil.threshold_yen(x) < 1-1/self.nb_classes:
            return sfil.threshold_yen(x[x>1-1/self.nb_classes])
        return max(sfil.threshold_yen(x), 1-1/self.nb_classes)'''

    def class_centroids(self, x, y):
        # Get the number of classes

        # Reduce the sum of each feature across samples within a class
        class_sums = torch.einsum("ji,jk->ki", x, y.float().cuda())
        # return class_sums
        # Count the number of samples in each class (sum along the sample dimension)
        class_counts = torch.sum(y, dim=0)

        # Avoid division by zero for empty classes
        class_counts[class_counts == 0] = 1

        # Divide class sums by class counts to get centroids
        centroids = (class_sums.T / class_counts).T

        return centroids

    def ini_centroids(self, src_dl):
        with TrainingModeManager([self.feature_extractor, self.classifier], train=False) as mgr, \
                Accumulator(['ctr']) as eval_accumulator, \
                torch.no_grad():
            for i, (im_s, label_source, id_s) in enumerate(tqdm(src_dl, desc='testing')):
                #print(im_s.shape)
                im_s = im_s.cuda()
                label_source = label_source.cuda()
                feature_ex_s = self.feature_extractor.forward(im_s)
                before_lincls_feat_s, after_lincls_t = self.classifier(feature_ex_s)
                norm_feat_s = F.normalize(before_lincls_feat_s)
                y_src = torch.eye(self.nb_classes, dtype=torch.int8).cuda()
                y_src = torch.eye(self.nb_classes, dtype=torch.int8).cuda()[label_source]
                ctr = self.class_centroids(norm_feat_s, y_src).unsqueeze(0)
                val = dict()
                for name in eval_accumulator.names:
                    val[name] = locals()[name].cpu().data.numpy()
                eval_accumulator.updateData(val)

        for x in eval_accumulator:
            val[x] = eval_accumulator[x]
        ctr = val['ctr']
        del val
        return torch.tensor(ctr.mean(axis=0))

    def centroids_target(self, trg_dl, K):
        X = []
        with TrainingModeManager([self.feature_extractor], train=False) as mgr, \
                torch.no_grad():
            for i, (im_t, label_target, id_t) in enumerate(tqdm(trg_dl, desc='testing')):
                im_t = im_t.cuda()
                feature_ex_t = self.feature_extractor.forward(im_t)
                before_lincls_feat_t, after_lincls_t = self.classifier(feature_ex_t)
                # norm_feat_t = F.normalize(before_lincls_feat_t)
                norm_feat_t = F.normalize(before_lincls_feat_t)
                X.append(norm_feat_t)
        X = torch.cat((X), 0).cpu()  # .numpy()

        kmeans = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(X)
        return kmeans.cluster_centers_

    def update_centroids_target(self, X, cen):
        X = X.cpu()
        cen = cen.cpu()
        dd = torch.cdist(X, cen)
        pred_cen = dd.argmin(axis=1)

        for i in pred_cen:
            ix = i == pred_cen
            cenc = X[ix].mean(axis=0)
            cen[i] = 0.9 * cen[i] + 0.1 * cenc

        return cen

    def update(self, src_loader, trg_loader, avg_meter, logger):
        self.src_loader = src_loader
        self.init_queue(src_loader)
        print("Memory State : ", self.memqueue_feat.is_memory_full())
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        self.src_latent_cluster = self.ini_centroids(src_loader).cuda()
        self.trg_latent_cluster = self.centroids_target(trg_loader, self.hparams['K'])
        self.trg_latent_cluster = torch.from_numpy(self.trg_latent_cluster).to(torch.float)

        nb_pr_epochs = self.hparams["num_epochs_pr"]
        for epoch in range(1, nb_pr_epochs + 1):
            self.pretrain_epoch(src_loader, avg_meter)

            logger.debug(f'[Pr Epoch : {epoch}/{nb_pr_epochs}]')  # TODO : self.hparams["num_pr_epochs"]
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        with torch.no_grad():
            self.network.eval()
            X = self.src_loader.dataset.x_data.cuda()
            Y = self.src_loader.dataset.y_data.numpy()
            _, logits = self.network(X)
            preds = logits.detach().cpu().argmax(axis=1).numpy()
            print("SRC Accuracy : ", (Y == preds).sum() / len(Y))
        self.network.train()
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # source pretraining loop
            # self.pretrain_epoch(src_loader, avg_meter)

            # training loop
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        cnt_i = 0
        self.trg_feats_mem = []
        self.trg_preds_mem = []
        trg_mem_size = self.hparams["trg_mem_size"]
        with torch.no_grad():
            self.network.eval()
            for x, y, id in trg_loader:
                x, y, id = x.to(self.device), y.to(self.device), id.to(self.device)
                feature_ex_t = self.feature_extractor(x)
                before_lincls_feat_t, after_lincls_s = self.classifier(feature_ex_t)
                norm_feat_t = F.normalize(before_lincls_feat_t)
                self.trg_feats_mem.append(norm_feat_t)
                self.trg_preds_mem.append(after_lincls_s)
                # self.memqueue_preds.update_queue(y_src, id.clone())
                cnt_i += after_lincls_s.shape[0]
                if cnt_i > trg_mem_size:
                    break
        self.trg_feats_mem = torch.concatenate(self.trg_feats_mem)[:trg_mem_size]
        self.trg_preds_mem = torch.concatenate(self.trg_preds_mem)[:trg_mem_size]

        if self.hparams['joint_decision']:
            dist_trg_tr = self.compute_cluster_distance(self.trg_feats_mem)
            soft_trg_tr = self.joint_decision(self.trg_preds_mem, dist_trg_tr)
        else:
            soft_trg_tr = F.softmax(self.trg_preds_mem, dim=1)
        conf, preds = soft_trg_tr.max(dim=1)
        self.final_threshold = self.threshold_method(conf.detach().cpu().numpy())




        '''X = self.src_loader.dataset.x_data.cuda()
        Y = self.src_loader.dataset.y_data.numpy()
        _, logits = self.network(X)
        preds = logits.detach().cpu().argmax(axis=1).numpy()
        print("SRC Accuracy : ", (Y == preds).sum() / len(Y))'''
        last_model = self.network.state_dict()

        return last_model, best_model

    def pretrain_epoch(self, src_loader, avg_meter):

        for src_x, src_y, _ in src_loader:
            src_x, src_y = src_x.to(self.device), src_y.to(self.device)

            src_feat = self.feature_extractor(src_x)
            _, src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)
            #info_loss = self.infomax_loss(F.softmax(src_pred))

            loss = src_cls_loss #+ info_loss

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            losses = {'Pr_Src_cls_loss': loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

    def compute_cluster_distance(self, x_test):
        #x = self.src_loader.dataset.x_data
        #y = self.src_loader.dataset.y_data
        x = self.memqueue_feat.mem_feat
        '''y = self.memqueue_feat.mem_id
        x, x_test, y = torch.Tensor(x).cuda(), torch.Tensor(x_test), torch.Tensor(y).cuda().long()'''

        #x = F.normalize(before_lincls_feat_t)
        x, x_test = x.squeeze(), x_test.squeeze()
        '''nb_classes = self.nb_classes
        res = torch.empty(x_test.shape[0], nb_classes)
        res = torch.zeros_like(res)
        print(y.unique(return_counts=True))
        for i, ll in enumerate(range(nb_classes)):
            dist = torch.cdist(x[y == ll], x_test).min(axis=0).values
            res[:, int(ll)] = dist'''
        if len(x_test.shape) == 1:
            x_test = x_test.unsqueeze(0)
        res = self.memqueue_feat.compute_distances(x_test)
        #print(res)

        # res = F.tanh(res)
        # res = res/res.max()
        d = -1 * res
        # d = 1-res
        d = F.softmax(d, dim=1)

        return d

    def compute_cluster_distance2(self, x_test):
        x = self.src_loader.dataset.x_data
        y = self.src_loader.dataset.y_data
        x, x_test, y = torch.Tensor(x).cuda(), torch.Tensor(x_test), torch.Tensor(y).cuda().long()
        feature_ex_t = self.feature_extractor(x)
        before_lincls_feat_t, after_lincls_t = self.classifier(feature_ex_t)

        x = F.normalize(before_lincls_feat_t)
        # _, x, _ = net.forward_extractor_logits(x)
        x, x_test = x.squeeze(), x_test.squeeze()
        nb_classes = self.nb_classes
        res = torch.empty(x_test.shape[0], nb_classes)
        res = torch.zeros_like(res)
        for i, ll in enumerate(range(nb_classes)):
            dist = torch.cdist(x[y == ll], x_test).min(axis=0).values
            res[:, int(ll)] = dist

        d = -1 * res
        d = F.softmax(d, dim=1)
        #d = 1-res/res.max()
        return d.max(axis=0).values

    def joint_decision(self, preds, distance):
        preds = torch.tensor(preds).cuda()
        distance = distance.cuda()
        return F.softmax(preds*distance, dim=1)#* distance
        #return F.softmax(F.softmax(preds, dim=1) * distance, dim=1)  # * distance
        #return 0.5*F.softmax(preds, dim=1) + 0.5*distance # * distance
    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders
        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))
        #temp = self.hparams['temp']
        # soft = nn.Softmax(dim=1)
        for step, ((src_x, src_y, id_source), (trg_x, _, id_target)) in joint_loader:
            """if src_x.shape[0] != trg_x.shape[0]:
                continue"""

            if src_x.shape[0] > trg_x.shape[0]:
                src_x = src_x[:trg_x.shape[0]]
                src_y = src_y[:trg_x.shape[0]]
            elif trg_x.shape[0] > src_x.shape[0]:
                trg_x = trg_x[:src_x.shape[0]]

            batch_size = len(src_x)
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(
                self.device)  # extract source features
            feature_ex_s = self.feature_extractor(src_x)
            feature_ex_t = self.feature_extractor(trg_x)

            before_lincls_feat_s, after_lincls_s = self.classifier(feature_ex_s)
            before_lincls_feat_t, after_lincls_t = self.classifier(feature_ex_t)

            norm_feat_s = F.normalize(before_lincls_feat_s)
            norm_feat_t = F.normalize(before_lincls_feat_t)

            # =====Source Supervision=====

            y_src = torch.eye(after_lincls_s.shape[-1], dtype=torch.int8).cuda()[src_y]
            self.memqueue_feat.update_queue(norm_feat_s, src_y.cuda())
            #self.memqueue_preds.update_queue(y_src, id_source.cuda())
            #print(y_src.shape)
            # print("y_src shape : ", y_src.shape)

            centr = self.class_centroids(norm_feat_s, y_src).detach().cpu()
            src_latent_cluster_copy = 0.9 * self.src_latent_cluster + 0.1 * centr.cuda()
            # TRG Centroids
            trg_latent_cluster_copy = self.update_centroids_target(norm_feat_t.detach().cpu(), self.trg_latent_cluster).cuda()
            before_lincls_feat_cen, after_lincls_cen = self.classifier(trg_latent_cluster_copy)
            trg_soft_cen = torch.nn.functional.softmax(after_lincls_cen).double()

            if self.hparams['joint_decision']:
                dist = self.compute_cluster_distance(norm_feat_t)
                soft_t = self.joint_decision(after_lincls_t, dist)
            else:
                soft_t = F.softmax(after_lincls_t)
            conf, preds = soft_t.max(dim=1)
            threshold = self.threshold_method(conf.detach().cpu().numpy())
            if step == 1: print('threshold : ', threshold)
            mask = (conf < threshold).cuda()
            # print("Detected ODD : ", mask.sum().item())

            # print("Number of ODD : ", (trg_label >= cls_output_dim).sum().item())
            # C0 = torch.zeros((len(norm_feat_s) + 1, len(norm_feat_t))).cuda()

            C0 = torch.zeros((len(norm_feat_s) + len(trg_latent_cluster_copy), len(norm_feat_t))).cuda()
            C_latent = src_latent_cluster_copy.mean(axis=0).unsqueeze(0)

            # nonOOD SRC
            C0[:len(norm_feat_s), ~mask] = torch.cdist(norm_feat_s, norm_feat_t[~mask])
            # OOD Dummy
            C0[len(norm_feat_s):, mask] = torch.cdist(trg_latent_cluster_copy, norm_feat_t[mask])

            maxc = torch.max(C0).item()  # *self.hparams['psi']
            # ODD SRC
            C0[:len(norm_feat_s), mask] = maxc
            # nonOOD Dummy
            C0[len(norm_feat_s):, ~mask] = maxc

            C1 = torch.zeros(C0.shape).cuda()
            C_preds = torch.ones((trg_latent_cluster_copy.shape[0], y_src.shape[-1])) / self.nb_classes
            C_preds = C_preds.cuda()

            # nonOOD SRC
            C1[:len(norm_feat_s), ~mask] = torch.cdist(y_src.float(), F.softmax(after_lincls_t)[~mask])
            # OOD Dummy
            C1[len(norm_feat_s):, mask] = torch.cdist(C_preds, F.softmax(after_lincls_t)[mask])
            maxc = torch.max(C1).item()  # *self.hparams['psi']
            # ODD SRC
            C1[:len(norm_feat_s), mask] = maxc
            # nonOOD Dummy
            C1[len(norm_feat_s):, ~mask] = maxc

            C = (self.hparams['alpha'] * C0 + self.hparams['lamb'] * C1)

            with torch.no_grad():

                a, b = ot.unif(C.size(0)), ot.unif(C.size(1))
                ratio = (mask.sum() / len(mask)).detach().cpu().item()
                a[:len(norm_feat_s)] = 0.5 / len(a[:len(norm_feat_s)])  #
                # a[:len(norm_feat_s)] = (1-ratio)/len(a[:len(norm_feat_s)])
                a[len(norm_feat_s):] = 0.5 / len(a[len(norm_feat_s):])  #
                # a[len(norm_feat_s):] = ratio/len(a[len(norm_feat_s):])
                # print(a.sum(), b.sum())

                gamma = ot.unbalanced.mm_unbalanced(a, b, C.detach().cpu().numpy(), reg_m=0.5)
                # gamma = ot.partial.partial_wasserstein(a, b, C.detach().cpu().numpy(), m=temp)
                # gamma = ot.sinkhorn(a, b, C.detach().cpu().numpy(), reg=0.01)
                # gamma = ot.emd(a, b, C.detach().cpu().numpy())
                # mass.append(gamma.sum())
                if step == 1:print('Mass : ', gamma.sum())
                gamma = torch.tensor(gamma).cuda()

            assert not torch.isnan(gamma).any()
            assert not torch.isnan(feature_ex_t).any()
            assert not torch.isnan(feature_ex_s).any()
            assert not torch.isnan(after_lincls_t).any()
            assert not torch.isnan(after_lincls_s).any()

            label_align_loss = (C * gamma)  # .sum()

            label_align_loss_nonOOD = label_align_loss[:len(norm_feat_s), ~mask].sum()
            label_align_loss_OOD = label_align_loss[len(norm_feat_s):, mask].sum()

            #label_align_loss = self.hparams['lamb'] * (label_align_loss_nonOOD + label_align_loss_OOD) / 2
            label_align_loss = (label_align_loss_nonOOD + label_align_loss_OOD) / 2

            criterion = nn.CrossEntropyLoss().cuda()
            loss_cls = self.hparams['src_weight'] * criterion(after_lincls_s, src_y)

            loss_all = loss_cls + label_align_loss #+ info_loss

            self.optimizer_feat.zero_grad()
            self.optimizer_cls.zero_grad()
            loss_all.backward()
            self.optimizer_feat.step()
            self.optimizer_cls.step()

            self.classifier.ProtoCLS.weight_norm()  # very important for proto-classifier

            losses = {'Total_loss': loss_all.item(), 'loss_cls': loss_cls.item(),
                      'loss_align': label_align_loss.item()}#, 'info_loss': info_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

    def evaluate(self, test_loader, trg_private_class, src=False):
        self.feature_extractor.eval()
        self.classifier.eval()

        total_loss, preds_list, labels_list = [], [], []



        '''dist_trg_tr = self.compute_cluster_distance(self.trg_feats_mem)
        soft_trg_tr = self.joint_decision(self.trg_preds_mem, dist_trg_tr)
        conf, preds = soft_trg_tr.max(dim=1)
        threshold = self.threshold_method(conf.detach().cpu().numpy())'''

        with torch.no_grad():
            for data, labels, ids in test_loader:
                data = data.float().to(self.device)
                '''if data.shape[0] == 1:
                    continue'''
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features = self.feature_extractor(data)
                before_lincls_feat_t, predictions = self.classifier(features)
                soft = F.softmax(predictions)

                if self.hparams['joint_decision']:
                    norm_feat_t = F.normalize(before_lincls_feat_t)
                    dist = self.compute_cluster_distance(norm_feat_t)
                    soft = self.joint_decision(predictions, dist)
                else:
                    soft = F.softmax(predictions)

                # preds_t = soft_t.argmax(dim=1)
                conf, preds = soft.max(dim=1)
                threshold = self.threshold_method(conf.detach().cpu().numpy())
                print("Finale Threshold : ", threshold)
                mask = conf < self.final_threshold #threshold
                if not src:
                    predictions[mask.squeeze()] *= 0

                if self.is_uniDA:
                    mask = labels >= predictions.shape[-1]
                    labels[mask] = predictions.shape[-1]

                mask = labels < predictions.shape[-1]
                # z = torch.zeros((len(predictions), 1))
                # predictions = torch.cat((predictions, z.to(predictions.device)), dim=1)
                loss = F.cross_entropy(predictions[mask], labels[mask])
                total_loss.append(loss.detach().cpu().item())
                # predictions = self.algorithm.correct_predictions(predictions)
                pred = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)

        loss = torch.tensor(total_loss).mean()  # average loss
        full_preds = torch.cat((preds_list))
        full_labels = torch.cat((labels_list))
        return loss, full_preds, full_labels

    def get_latent_features(self, dataloader):
        feature_set = []
        label_set = []
        logits_set = []
        self.feature_extractor.eval()
        self.classifier.eval()
        with torch.no_grad():
            for _, (data, label, ids) in enumerate(dataloader):
                data = data.to(self.device)
                '''if data.shape[0] == 1:
                    continue'''
                feature = self.feature_extractor(data)
                _, logit = self.classifier(feature)
                feature_set.append(feature.cpu())
                label_set.append(label.cpu())
                logits_set.append(logit.cpu())
            feature_set = torch.cat(feature_set, dim=0)
            feature_set = F.normalize(feature_set, p=2, dim=-1)
            label_set = torch.cat(label_set, dim=0)
            logits_set = torch.cat(logits_set, dim=0)
        return feature_set, label_set, logits_set

    def decision_function(self, preds):
        mask = preds.sum(axis=1) == 0.0
        confidence, pred = preds.max(dim=1)
        pred[mask] = -1
        return pred

class UniJDOT_THR(Algorithm):
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        print(configs)
        # device
        self.device = device
        self.feature_extractor = backbone(configs).to(self.device)
        #self.classifier = CLS(configs, temp=hparams['temp']).to(self.device)
        self.classifier = CLS(configs).to(self.device)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # hparams
        self.hparams = hparams
        print("Batch Size : ", hparams['batch_size'])
        self.nb_classes = configs.num_classes

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer_feat = torch.optim.Adam(
            self.feature_extractor.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer_cls = torch.optim.Adam(
            self.classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.beta = None
        self.softmax = torch.nn.Softmax(dim=1)
        self.bce = BCELoss()
        self.is_uniDA = True
        self.src_latent_cluster = None


        self.manual_thr = hparams['threshold']
        self.threshold_method = self.v #sfil.threshold_yen #sfil.threshold_triangle #sfil.threshold_yen
        #self.threshold_method = self.v
        #self.threshold_method = sfil.threshold_yen  # sfil.threshold_triangle #sfil.threshold_yen
        if configs.isFNO:
            feat_dim = configs.final_out_channels + 2*configs.fourier_modes
        else:
            feat_dim = configs.final_out_channels #+ 2 * configs.fourier_modes
        self.memqueue_feat = ClassMemoryQueue(feat_dim, self.nb_classes, hparams['n_batch']).cuda()
        #self.memqueue_preds = MemoryQueue(configs.num_classes, hparams['batch_size'], hparams['n_batch']).cuda()

    '''def v(self, x):
        return 0.5 * sfil.threshold_yen(x) + 0.5 * sfil.threshold_otsu(x)'''

    def v(self, x):
        return self.manual_thr
    def init_queue(self, dataloader):
        cnt_i = 0
        for x,y, id in dataloader:
            x, y, id = x.to(self.device), y.to(self.device), id.to(self.device)
            feature_ex_s = self.feature_extractor(x)
            before_lincls_feat_s, after_lincls_s = self.classifier(feature_ex_s)
            self.memqueue_feat.update_queue(F.normalize(before_lincls_feat_s), y)
            #self.memqueue_preds.update_queue(y_src, id.clone())
            cnt_i += 1
            if self.memqueue_feat.is_memory_full().all():
                break
        print('Memory after init : ', self.memqueue_feat.is_memory_full())

    def infomax_loss(self, cluster_assignments, eps=1e-10):
        """
        InfoMax loss function to maximize mutual information between inputs and cluster assignments.

        Args:
            cluster_assignments (torch.Tensor): Tensor of shape (N, K) representing the probability distribution
                                                over K clusters for each sample.
            eps (float): Small constant to avoid numerical issues with log.

        Returns:
            torch.Tensor: Scalar loss value (InfoMax loss).
        """
        # Batch size and number of clusters
        N, K = cluster_assignments.shape

        # Step 1: Compute the marginal distribution p(z) over clusters (averaging over samples)
        marginal_prob = cluster_assignments.mean(dim=0)  # Shape (K,)

        # Step 2: Compute H(Z) - Entropy of the marginal distribution (cluster assignments)
        H_Z = -torch.sum(marginal_prob * torch.log(marginal_prob + eps))

        # Step 3: Compute H(Z|X) - Conditional entropy of cluster assignment given input
        H_Z_given_X = -torch.sum(cluster_assignments * torch.log(cluster_assignments + eps)) / N

        # Step 4: InfoMax loss is H(Z) - H(Z|X)
        infomax_loss_value = H_Z - H_Z_given_X

        return infomax_loss_value

    def get_thresholding_method(self):
        return getattr(sfil, self.hparams['threshold_method'])

    '''def threshold_method(self, x):
        if sfil.threshold_yen(x) < 1-1/self.nb_classes:
            return sfil.threshold_yen(x[x>1-1/self.nb_classes])
        return max(sfil.threshold_yen(x), 1-1/self.nb_classes)'''

    def class_centroids(self, x, y):
        # Get the number of classes

        # Reduce the sum of each feature across samples within a class
        class_sums = torch.einsum("ji,jk->ki", x, y.float().cuda())
        # return class_sums
        # Count the number of samples in each class (sum along the sample dimension)
        class_counts = torch.sum(y, dim=0)

        # Avoid division by zero for empty classes
        class_counts[class_counts == 0] = 1

        # Divide class sums by class counts to get centroids
        centroids = (class_sums.T / class_counts).T

        return centroids

    def ini_centroids(self, src_dl):
        with TrainingModeManager([self.feature_extractor, self.classifier], train=False) as mgr, \
                Accumulator(['ctr']) as eval_accumulator, \
                torch.no_grad():
            for i, (im_s, label_source, id_s) in enumerate(tqdm(src_dl, desc='testing')):
                #print(im_s.shape)
                im_s = im_s.cuda()
                label_source = label_source.cuda()
                feature_ex_s = self.feature_extractor.forward(im_s)
                before_lincls_feat_s, after_lincls_t = self.classifier(feature_ex_s)
                norm_feat_s = F.normalize(before_lincls_feat_s)
                y_src = torch.eye(self.nb_classes, dtype=torch.int8).cuda()
                y_src = torch.eye(self.nb_classes, dtype=torch.int8).cuda()[label_source]
                ctr = self.class_centroids(norm_feat_s, y_src).unsqueeze(0)
                val = dict()
                for name in eval_accumulator.names:
                    val[name] = locals()[name].cpu().data.numpy()
                eval_accumulator.updateData(val)

        for x in eval_accumulator:
            val[x] = eval_accumulator[x]
        ctr = val['ctr']
        del val
        return torch.tensor(ctr.mean(axis=0))

    def centroids_target(self, trg_dl, K):
        X = []
        with TrainingModeManager([self.feature_extractor], train=False) as mgr, \
                torch.no_grad():
            for i, (im_t, label_target, id_t) in enumerate(tqdm(trg_dl, desc='testing')):
                im_t = im_t.cuda()
                feature_ex_t = self.feature_extractor.forward(im_t)
                before_lincls_feat_t, after_lincls_t = self.classifier(feature_ex_t)
                # norm_feat_t = F.normalize(before_lincls_feat_t)
                norm_feat_t = F.normalize(before_lincls_feat_t)
                X.append(norm_feat_t)
        X = torch.cat((X), 0).cpu()  # .numpy()

        kmeans = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(X)
        return kmeans.cluster_centers_

    def update_centroids_target(self, X, cen):
        X = X.cpu()
        cen = cen.cpu()
        dd = torch.cdist(X, cen)
        pred_cen = dd.argmin(axis=1)

        for i in pred_cen:
            ix = i == pred_cen
            cenc = X[ix].mean(axis=0)
            cen[i] = 0.9 * cen[i] + 0.1 * cenc

        return cen

    def update(self, src_loader, trg_loader, avg_meter, logger):
        self.src_loader = src_loader
        self.init_queue(src_loader)
        print("Memory State : ", self.memqueue_feat.is_memory_full())
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        self.src_latent_cluster = self.ini_centroids(src_loader).cuda()
        self.trg_latent_cluster = self.centroids_target(trg_loader, self.hparams['K'])
        self.trg_latent_cluster = torch.from_numpy(self.trg_latent_cluster).to(torch.float)

        nb_pr_epochs = self.hparams["num_epochs_pr"]
        for epoch in range(1, nb_pr_epochs + 1):
            self.pretrain_epoch(src_loader, avg_meter)

            logger.debug(f'[Pr Epoch : {epoch}/{nb_pr_epochs}]')  # TODO : self.hparams["num_pr_epochs"]
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        with torch.no_grad():
            self.network.eval()
            X = self.src_loader.dataset.x_data.cuda()
            Y = self.src_loader.dataset.y_data.numpy()
            _, logits = self.network(X)
            preds = logits.detach().cpu().argmax(axis=1).numpy()
            print("SRC Accuracy : ", (Y == preds).sum() / len(Y))
        self.network.train()
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # source pretraining loop
            # self.pretrain_epoch(src_loader, avg_meter)

            # training loop
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        cnt_i = 0
        self.trg_feats_mem = []
        self.trg_preds_mem = []
        trg_mem_size = self.hparams["trg_mem_size"]
        with torch.no_grad():
            self.network.eval()
            for x, y, id in trg_loader:
                x, y, id = x.to(self.device), y.to(self.device), id.to(self.device)
                feature_ex_t = self.feature_extractor(x)
                before_lincls_feat_t, after_lincls_s = self.classifier(feature_ex_t)
                norm_feat_t = F.normalize(before_lincls_feat_t)
                self.trg_feats_mem.append(norm_feat_t)
                self.trg_preds_mem.append(after_lincls_s)
                # self.memqueue_preds.update_queue(y_src, id.clone())
                cnt_i += after_lincls_s.shape[0]
                if cnt_i > trg_mem_size:
                    break
        self.trg_feats_mem = torch.concatenate(self.trg_feats_mem)[:trg_mem_size]
        self.trg_preds_mem = torch.concatenate(self.trg_preds_mem)[:trg_mem_size]

        if self.hparams['joint_decision']:
            dist_trg_tr = self.compute_cluster_distance(self.trg_feats_mem)
            soft_trg_tr = self.joint_decision(self.trg_preds_mem, dist_trg_tr)
        else:
            soft_trg_tr = F.softmax(self.trg_preds_mem, dim=1)
        conf, preds = soft_trg_tr.max(dim=1)
        self.final_threshold = self.threshold_method(conf.detach().cpu().numpy())




        '''X = self.src_loader.dataset.x_data.cuda()
        Y = self.src_loader.dataset.y_data.numpy()
        _, logits = self.network(X)
        preds = logits.detach().cpu().argmax(axis=1).numpy()
        print("SRC Accuracy : ", (Y == preds).sum() / len(Y))'''
        last_model = self.network.state_dict()

        return last_model, best_model

    def pretrain_epoch(self, src_loader, avg_meter):

        for src_x, src_y, _ in src_loader:
            src_x, src_y = src_x.to(self.device), src_y.to(self.device)

            src_feat = self.feature_extractor(src_x)
            _, src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)
            #info_loss = self.infomax_loss(F.softmax(src_pred))

            loss = src_cls_loss #+ info_loss

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            losses = {'Pr_Src_cls_loss': loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

    def compute_cluster_distance(self, x_test):
        #x = self.src_loader.dataset.x_data
        #y = self.src_loader.dataset.y_data
        x = self.memqueue_feat.mem_feat
        '''y = self.memqueue_feat.mem_id
        x, x_test, y = torch.Tensor(x).cuda(), torch.Tensor(x_test), torch.Tensor(y).cuda().long()'''

        #x = F.normalize(before_lincls_feat_t)
        x, x_test = x.squeeze(), x_test.squeeze()
        '''nb_classes = self.nb_classes
        res = torch.empty(x_test.shape[0], nb_classes)
        res = torch.zeros_like(res)
        print(y.unique(return_counts=True))
        for i, ll in enumerate(range(nb_classes)):
            dist = torch.cdist(x[y == ll], x_test).min(axis=0).values
            res[:, int(ll)] = dist'''
        if len(x_test.shape) == 1:
            x_test = x_test.unsqueeze(0)
        res = self.memqueue_feat.compute_distances(x_test)
        #print(res)

        # res = F.tanh(res)
        # res = res/res.max()
        d = -1 * res
        # d = 1-res
        d = F.softmax(d, dim=1)

        return d

    def compute_cluster_distance2(self, x_test):
        x = self.src_loader.dataset.x_data
        y = self.src_loader.dataset.y_data
        x, x_test, y = torch.Tensor(x).cuda(), torch.Tensor(x_test), torch.Tensor(y).cuda().long()
        feature_ex_t = self.feature_extractor(x)
        before_lincls_feat_t, after_lincls_t = self.classifier(feature_ex_t)

        x = F.normalize(before_lincls_feat_t)
        # _, x, _ = net.forward_extractor_logits(x)
        x, x_test = x.squeeze(), x_test.squeeze()
        nb_classes = self.nb_classes
        res = torch.empty(x_test.shape[0], nb_classes)
        res = torch.zeros_like(res)
        for i, ll in enumerate(range(nb_classes)):
            dist = torch.cdist(x[y == ll], x_test).min(axis=0).values
            res[:, int(ll)] = dist

        d = -1 * res
        d = F.softmax(d, dim=1)
        #d = 1-res/res.max()
        return d.max(axis=0).values

    def joint_decision(self, preds, distance):
        preds = torch.tensor(preds).cuda()
        distance = distance.cuda()
        return F.softmax(preds*distance, dim=1)#* distance
        #return F.softmax(F.softmax(preds, dim=1) * distance, dim=1)  # * distance
        #return 0.5*F.softmax(preds, dim=1) + 0.5*distance # * distance
    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders
        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))
        #temp = self.hparams['temp']
        # soft = nn.Softmax(dim=1)
        for step, ((src_x, src_y, id_source), (trg_x, _, id_target)) in joint_loader:
            """if src_x.shape[0] != trg_x.shape[0]:
                continue"""

            if src_x.shape[0] > trg_x.shape[0]:
                src_x = src_x[:trg_x.shape[0]]
                src_y = src_y[:trg_x.shape[0]]
            elif trg_x.shape[0] > src_x.shape[0]:
                trg_x = trg_x[:src_x.shape[0]]

            batch_size = len(src_x)
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(
                self.device)  # extract source features
            feature_ex_s = self.feature_extractor(src_x)
            feature_ex_t = self.feature_extractor(trg_x)

            before_lincls_feat_s, after_lincls_s = self.classifier(feature_ex_s)
            before_lincls_feat_t, after_lincls_t = self.classifier(feature_ex_t)

            norm_feat_s = F.normalize(before_lincls_feat_s)
            norm_feat_t = F.normalize(before_lincls_feat_t)

            # =====Source Supervision=====

            y_src = torch.eye(after_lincls_s.shape[-1], dtype=torch.int8).cuda()[src_y]
            self.memqueue_feat.update_queue(norm_feat_s, src_y.cuda())
            #self.memqueue_preds.update_queue(y_src, id_source.cuda())
            #print(y_src.shape)
            # print("y_src shape : ", y_src.shape)

            centr = self.class_centroids(norm_feat_s, y_src).detach().cpu()
            src_latent_cluster_copy = 0.9 * self.src_latent_cluster + 0.1 * centr.cuda()
            # TRG Centroids
            trg_latent_cluster_copy = self.update_centroids_target(norm_feat_t.detach().cpu(), self.trg_latent_cluster).cuda()
            before_lincls_feat_cen, after_lincls_cen = self.classifier(trg_latent_cluster_copy)
            trg_soft_cen = torch.nn.functional.softmax(after_lincls_cen).double()

            if self.hparams['joint_decision']:
                dist = self.compute_cluster_distance(norm_feat_t)
                soft_t = self.joint_decision(after_lincls_t, dist)
            else:
                soft_t = F.softmax(after_lincls_t)
            conf, preds = soft_t.max(dim=1)
            threshold = self.threshold_method(conf.detach().cpu().numpy())
            if step == 1: print('threshold : ', threshold)
            mask = (conf < threshold).cuda()
            # print("Detected ODD : ", mask.sum().item())

            # print("Number of ODD : ", (trg_label >= cls_output_dim).sum().item())
            # C0 = torch.zeros((len(norm_feat_s) + 1, len(norm_feat_t))).cuda()

            C0 = torch.zeros((len(norm_feat_s) + len(trg_latent_cluster_copy), len(norm_feat_t))).cuda()
            C_latent = src_latent_cluster_copy.mean(axis=0).unsqueeze(0)

            # nonOOD SRC
            C0[:len(norm_feat_s), ~mask] = torch.cdist(norm_feat_s, norm_feat_t[~mask])
            # OOD Dummy
            C0[len(norm_feat_s):, mask] = torch.cdist(trg_latent_cluster_copy, norm_feat_t[mask])

            maxc = torch.max(C0).item()  # *self.hparams['psi']
            # ODD SRC
            C0[:len(norm_feat_s), mask] = maxc
            # nonOOD Dummy
            C0[len(norm_feat_s):, ~mask] = maxc

            C1 = torch.zeros(C0.shape).cuda()
            C_preds = torch.ones((trg_latent_cluster_copy.shape[0], y_src.shape[-1])) / self.nb_classes
            C_preds = C_preds.cuda()

            # nonOOD SRC
            C1[:len(norm_feat_s), ~mask] = torch.cdist(y_src.float(), F.softmax(after_lincls_t)[~mask])
            # OOD Dummy
            C1[len(norm_feat_s):, mask] = torch.cdist(C_preds, F.softmax(after_lincls_t)[mask])
            maxc = torch.max(C1).item()  # *self.hparams['psi']
            # ODD SRC
            C1[:len(norm_feat_s), mask] = maxc
            # nonOOD Dummy
            C1[len(norm_feat_s):, ~mask] = maxc

            C = (self.hparams['alpha'] * C0 + self.hparams['lamb'] * C1)

            with torch.no_grad():

                a, b = ot.unif(C.size(0)), ot.unif(C.size(1))
                ratio = (mask.sum() / len(mask)).detach().cpu().item()
                a[:len(norm_feat_s)] = 0.5 / len(a[:len(norm_feat_s)])  #
                # a[:len(norm_feat_s)] = (1-ratio)/len(a[:len(norm_feat_s)])
                a[len(norm_feat_s):] = 0.5 / len(a[len(norm_feat_s):])  #
                # a[len(norm_feat_s):] = ratio/len(a[len(norm_feat_s):])
                # print(a.sum(), b.sum())

                gamma = ot.unbalanced.mm_unbalanced(a, b, C.detach().cpu().numpy(), reg_m=0.5)
                # gamma = ot.partial.partial_wasserstein(a, b, C.detach().cpu().numpy(), m=temp)
                # gamma = ot.sinkhorn(a, b, C.detach().cpu().numpy(), reg=0.01)
                # gamma = ot.emd(a, b, C.detach().cpu().numpy())
                # mass.append(gamma.sum())
                if step == 1:print('Mass : ', gamma.sum())
                gamma = torch.tensor(gamma).cuda()

            assert not torch.isnan(gamma).any()
            assert not torch.isnan(feature_ex_t).any()
            assert not torch.isnan(feature_ex_s).any()
            assert not torch.isnan(after_lincls_t).any()
            assert not torch.isnan(after_lincls_s).any()

            label_align_loss = (C * gamma)  # .sum()

            label_align_loss_nonOOD = label_align_loss[:len(norm_feat_s), ~mask].sum()
            label_align_loss_OOD = label_align_loss[len(norm_feat_s):, mask].sum()

            #label_align_loss = self.hparams['lamb'] * (label_align_loss_nonOOD + label_align_loss_OOD) / 2
            label_align_loss = (label_align_loss_nonOOD + label_align_loss_OOD) / 2

            criterion = nn.CrossEntropyLoss().cuda()
            loss_cls = self.hparams['src_weight'] * criterion(after_lincls_s, src_y)

            loss_all = loss_cls + label_align_loss #+ info_loss

            self.optimizer_feat.zero_grad()
            self.optimizer_cls.zero_grad()
            loss_all.backward()
            self.optimizer_feat.step()
            self.optimizer_cls.step()

            self.classifier.ProtoCLS.weight_norm()  # very important for proto-classifier

            losses = {'Total_loss': loss_all.item(), 'loss_cls': loss_cls.item(),
                      'loss_align': label_align_loss.item()}#, 'info_loss': info_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

    def evaluate(self, test_loader, trg_private_class, src=False):
        self.feature_extractor.eval()
        self.classifier.eval()

        total_loss, preds_list, labels_list = [], [], []



        '''dist_trg_tr = self.compute_cluster_distance(self.trg_feats_mem)
        soft_trg_tr = self.joint_decision(self.trg_preds_mem, dist_trg_tr)
        conf, preds = soft_trg_tr.max(dim=1)
        threshold = self.threshold_method(conf.detach().cpu().numpy())'''

        with torch.no_grad():
            for data, labels, ids in test_loader:
                data = data.float().to(self.device)
                '''if data.shape[0] == 1:
                    continue'''
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features = self.feature_extractor(data)
                before_lincls_feat_t, predictions = self.classifier(features)
                soft = F.softmax(predictions)

                if self.hparams['joint_decision']:
                    norm_feat_t = F.normalize(before_lincls_feat_t)
                    dist = self.compute_cluster_distance(norm_feat_t)
                    soft = self.joint_decision(predictions, dist)
                else:
                    soft = F.softmax(predictions)

                # preds_t = soft_t.argmax(dim=1)
                conf, preds = soft.max(dim=1)
                threshold = self.threshold_method(conf.detach().cpu().numpy())
                print("Finale Threshold : ", threshold)
                mask = conf < self.final_threshold #threshold
                if not src:
                    predictions[mask.squeeze()] *= 0

                if self.is_uniDA:
                    mask = labels >= predictions.shape[-1]
                    labels[mask] = predictions.shape[-1]

                mask = labels < predictions.shape[-1]
                # z = torch.zeros((len(predictions), 1))
                # predictions = torch.cat((predictions, z.to(predictions.device)), dim=1)
                loss = F.cross_entropy(predictions[mask], labels[mask])
                total_loss.append(loss.detach().cpu().item())
                # predictions = self.algorithm.correct_predictions(predictions)
                pred = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)

        loss = torch.tensor(total_loss).mean()  # average loss
        full_preds = torch.cat((preds_list))
        full_labels = torch.cat((labels_list))
        return loss, full_preds, full_labels

    def get_latent_features(self, dataloader):
        feature_set = []
        label_set = []
        logits_set = []
        self.feature_extractor.eval()
        self.classifier.eval()
        with torch.no_grad():
            for _, (data, label, ids) in enumerate(dataloader):
                data = data.to(self.device)
                '''if data.shape[0] == 1:
                    continue'''
                feature = self.feature_extractor(data)
                _, logit = self.classifier(feature)
                feature_set.append(feature.cpu())
                label_set.append(label.cpu())
                logits_set.append(logit.cpu())
            feature_set = torch.cat(feature_set, dim=0)
            feature_set = F.normalize(feature_set, p=2, dim=-1)
            label_set = torch.cat(label_set, dim=0)
            logits_set = torch.cat(logits_set, dim=0)
        return feature_set, label_set, logits_set

    def decision_function(self, preds):
        mask = preds.sum(axis=1) == 0.0
        confidence, pred = preds.max(dim=1)
        pred[mask] = -1
        return pred

class UniJDOT_THR_NOJOINT(Algorithm):
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        print(configs)
        # device
        self.device = device
        self.feature_extractor = backbone(configs).to(self.device)
        #self.classifier = CLS(configs, temp=hparams['temp']).to(self.device)
        self.classifier = CLS(configs).to(self.device)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # hparams
        self.hparams = hparams
        print("Batch Size : ", hparams['batch_size'])
        self.nb_classes = configs.num_classes

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer_feat = torch.optim.Adam(
            self.feature_extractor.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer_cls = torch.optim.Adam(
            self.classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.beta = None
        self.softmax = torch.nn.Softmax(dim=1)
        self.bce = BCELoss()
        self.is_uniDA = True
        self.src_latent_cluster = None


        self.manual_thr = hparams['threshold']
        self.threshold_method = self.v #sfil.threshold_yen #sfil.threshold_triangle #sfil.threshold_yen
        #self.threshold_method = self.v
        #self.threshold_method = sfil.threshold_yen  # sfil.threshold_triangle #sfil.threshold_yen
        if configs.isFNO:
            feat_dim = configs.final_out_channels + 2*configs.fourier_modes
        else:
            feat_dim = configs.final_out_channels #+ 2 * configs.fourier_modes
        self.memqueue_feat = ClassMemoryQueue(feat_dim, self.nb_classes, hparams['n_batch']).cuda()
        #self.memqueue_preds = MemoryQueue(configs.num_classes, hparams['batch_size'], hparams['n_batch']).cuda()

    '''def v(self, x):
        return 0.5 * sfil.threshold_yen(x) + 0.5 * sfil.threshold_otsu(x)'''
    def v(self, x):
        return self.manual_thr
    def init_queue(self, dataloader):
        cnt_i = 0
        for x,y, id in dataloader:
            x, y, id = x.to(self.device), y.to(self.device), id.to(self.device)
            feature_ex_s = self.feature_extractor(x)
            before_lincls_feat_s, after_lincls_s = self.classifier(feature_ex_s)
            self.memqueue_feat.update_queue(F.normalize(before_lincls_feat_s), y)
            #self.memqueue_preds.update_queue(y_src, id.clone())
            cnt_i += 1
            if self.memqueue_feat.is_memory_full().all():
                break
        print('Memory after init : ', self.memqueue_feat.is_memory_full())

    def infomax_loss(self, cluster_assignments, eps=1e-10):
        """
        InfoMax loss function to maximize mutual information between inputs and cluster assignments.

        Args:
            cluster_assignments (torch.Tensor): Tensor of shape (N, K) representing the probability distribution
                                                over K clusters for each sample.
            eps (float): Small constant to avoid numerical issues with log.

        Returns:
            torch.Tensor: Scalar loss value (InfoMax loss).
        """
        # Batch size and number of clusters
        N, K = cluster_assignments.shape

        # Step 1: Compute the marginal distribution p(z) over clusters (averaging over samples)
        marginal_prob = cluster_assignments.mean(dim=0)  # Shape (K,)

        # Step 2: Compute H(Z) - Entropy of the marginal distribution (cluster assignments)
        H_Z = -torch.sum(marginal_prob * torch.log(marginal_prob + eps))

        # Step 3: Compute H(Z|X) - Conditional entropy of cluster assignment given input
        H_Z_given_X = -torch.sum(cluster_assignments * torch.log(cluster_assignments + eps)) / N

        # Step 4: InfoMax loss is H(Z) - H(Z|X)
        infomax_loss_value = H_Z - H_Z_given_X

        return infomax_loss_value

    def get_thresholding_method(self):
        return getattr(sfil, self.hparams['threshold_method'])

    '''def threshold_method(self, x):
        if sfil.threshold_yen(x) < 1-1/self.nb_classes:
            return sfil.threshold_yen(x[x>1-1/self.nb_classes])
        return max(sfil.threshold_yen(x), 1-1/self.nb_classes)'''

    def class_centroids(self, x, y):
        # Get the number of classes

        # Reduce the sum of each feature across samples within a class
        class_sums = torch.einsum("ji,jk->ki", x, y.float().cuda())
        # return class_sums
        # Count the number of samples in each class (sum along the sample dimension)
        class_counts = torch.sum(y, dim=0)

        # Avoid division by zero for empty classes
        class_counts[class_counts == 0] = 1

        # Divide class sums by class counts to get centroids
        centroids = (class_sums.T / class_counts).T

        return centroids

    def ini_centroids(self, src_dl):
        with TrainingModeManager([self.feature_extractor, self.classifier], train=False) as mgr, \
                Accumulator(['ctr']) as eval_accumulator, \
                torch.no_grad():
            for i, (im_s, label_source, id_s) in enumerate(tqdm(src_dl, desc='testing')):
                #print(im_s.shape)
                im_s = im_s.cuda()
                label_source = label_source.cuda()
                feature_ex_s = self.feature_extractor.forward(im_s)
                before_lincls_feat_s, after_lincls_t = self.classifier(feature_ex_s)
                norm_feat_s = F.normalize(before_lincls_feat_s)
                y_src = torch.eye(self.nb_classes, dtype=torch.int8).cuda()
                y_src = torch.eye(self.nb_classes, dtype=torch.int8).cuda()[label_source]
                ctr = self.class_centroids(norm_feat_s, y_src).unsqueeze(0)
                val = dict()
                for name in eval_accumulator.names:
                    val[name] = locals()[name].cpu().data.numpy()
                eval_accumulator.updateData(val)

        for x in eval_accumulator:
            val[x] = eval_accumulator[x]
        ctr = val['ctr']
        del val
        return torch.tensor(ctr.mean(axis=0))

    def centroids_target(self, trg_dl, K):
        X = []
        with TrainingModeManager([self.feature_extractor], train=False) as mgr, \
                torch.no_grad():
            for i, (im_t, label_target, id_t) in enumerate(tqdm(trg_dl, desc='testing')):
                im_t = im_t.cuda()
                feature_ex_t = self.feature_extractor.forward(im_t)
                before_lincls_feat_t, after_lincls_t = self.classifier(feature_ex_t)
                # norm_feat_t = F.normalize(before_lincls_feat_t)
                norm_feat_t = F.normalize(before_lincls_feat_t)
                X.append(norm_feat_t)
        X = torch.cat((X), 0).cpu()  # .numpy()

        kmeans = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(X)
        return kmeans.cluster_centers_

    def update_centroids_target(self, X, cen):
        X = X.cpu()
        cen = cen.cpu()
        dd = torch.cdist(X, cen)
        pred_cen = dd.argmin(axis=1)

        for i in pred_cen:
            ix = i == pred_cen
            cenc = X[ix].mean(axis=0)
            cen[i] = 0.9 * cen[i] + 0.1 * cenc

        return cen

    def update(self, src_loader, trg_loader, avg_meter, logger):
        self.src_loader = src_loader
        self.init_queue(src_loader)
        print("Memory State : ", self.memqueue_feat.is_memory_full())
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        self.src_latent_cluster = self.ini_centroids(src_loader).cuda()
        self.trg_latent_cluster = self.centroids_target(trg_loader, self.hparams['K'])
        self.trg_latent_cluster = torch.from_numpy(self.trg_latent_cluster).to(torch.float)

        nb_pr_epochs = self.hparams["num_epochs_pr"]
        for epoch in range(1, nb_pr_epochs + 1):
            self.pretrain_epoch(src_loader, avg_meter)

            logger.debug(f'[Pr Epoch : {epoch}/{nb_pr_epochs}]')  # TODO : self.hparams["num_pr_epochs"]
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        with torch.no_grad():
            self.network.eval()
            X = self.src_loader.dataset.x_data.cuda()
            Y = self.src_loader.dataset.y_data.numpy()
            _, logits = self.network(X)
            preds = logits.detach().cpu().argmax(axis=1).numpy()
            print("SRC Accuracy : ", (Y == preds).sum() / len(Y))
        self.network.train()
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # source pretraining loop
            # self.pretrain_epoch(src_loader, avg_meter)

            # training loop
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        cnt_i = 0
        self.trg_feats_mem = []
        self.trg_preds_mem = []
        trg_mem_size = self.hparams["trg_mem_size"]
        with torch.no_grad():
            self.network.eval()
            for x, y, id in trg_loader:
                x, y, id = x.to(self.device), y.to(self.device), id.to(self.device)
                feature_ex_t = self.feature_extractor(x)
                before_lincls_feat_t, after_lincls_s = self.classifier(feature_ex_t)
                norm_feat_t = F.normalize(before_lincls_feat_t)
                self.trg_feats_mem.append(norm_feat_t)
                self.trg_preds_mem.append(after_lincls_s)
                # self.memqueue_preds.update_queue(y_src, id.clone())
                cnt_i += after_lincls_s.shape[0]
                if cnt_i > trg_mem_size:
                    break
        self.trg_feats_mem = torch.concatenate(self.trg_feats_mem)[:trg_mem_size]
        self.trg_preds_mem = torch.concatenate(self.trg_preds_mem)[:trg_mem_size]

        if self.hparams['joint_decision']:
            dist_trg_tr = self.compute_cluster_distance(self.trg_feats_mem)
            soft_trg_tr = self.joint_decision(self.trg_preds_mem, dist_trg_tr)
        else:
            soft_trg_tr = F.softmax(self.trg_preds_mem, dim=1)
        conf, preds = soft_trg_tr.max(dim=1)
        self.final_threshold = self.threshold_method(conf.detach().cpu().numpy())




        '''X = self.src_loader.dataset.x_data.cuda()
        Y = self.src_loader.dataset.y_data.numpy()
        _, logits = self.network(X)
        preds = logits.detach().cpu().argmax(axis=1).numpy()
        print("SRC Accuracy : ", (Y == preds).sum() / len(Y))'''
        last_model = self.network.state_dict()

        return last_model, best_model

    def pretrain_epoch(self, src_loader, avg_meter):

        for src_x, src_y, _ in src_loader:
            src_x, src_y = src_x.to(self.device), src_y.to(self.device)

            src_feat = self.feature_extractor(src_x)
            _, src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)
            #info_loss = self.infomax_loss(F.softmax(src_pred))

            loss = src_cls_loss #+ info_loss

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            losses = {'Pr_Src_cls_loss': loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

    def compute_cluster_distance(self, x_test):
        #x = self.src_loader.dataset.x_data
        #y = self.src_loader.dataset.y_data
        x = self.memqueue_feat.mem_feat
        '''y = self.memqueue_feat.mem_id
        x, x_test, y = torch.Tensor(x).cuda(), torch.Tensor(x_test), torch.Tensor(y).cuda().long()'''

        #x = F.normalize(before_lincls_feat_t)
        x, x_test = x.squeeze(), x_test.squeeze()
        '''nb_classes = self.nb_classes
        res = torch.empty(x_test.shape[0], nb_classes)
        res = torch.zeros_like(res)
        print(y.unique(return_counts=True))
        for i, ll in enumerate(range(nb_classes)):
            dist = torch.cdist(x[y == ll], x_test).min(axis=0).values
            res[:, int(ll)] = dist'''
        if len(x_test.shape) == 1:
            x_test = x_test.unsqueeze(0)
        res = self.memqueue_feat.compute_distances(x_test)
        #print(res)

        # res = F.tanh(res)
        # res = res/res.max()
        d = -1 * res
        # d = 1-res
        d = F.softmax(d, dim=1)

        return d

    def compute_cluster_distance2(self, x_test):
        x = self.src_loader.dataset.x_data
        y = self.src_loader.dataset.y_data
        x, x_test, y = torch.Tensor(x).cuda(), torch.Tensor(x_test), torch.Tensor(y).cuda().long()
        feature_ex_t = self.feature_extractor(x)
        before_lincls_feat_t, after_lincls_t = self.classifier(feature_ex_t)

        x = F.normalize(before_lincls_feat_t)
        # _, x, _ = net.forward_extractor_logits(x)
        x, x_test = x.squeeze(), x_test.squeeze()
        nb_classes = self.nb_classes
        res = torch.empty(x_test.shape[0], nb_classes)
        res = torch.zeros_like(res)
        for i, ll in enumerate(range(nb_classes)):
            dist = torch.cdist(x[y == ll], x_test).min(axis=0).values
            res[:, int(ll)] = dist

        d = -1 * res
        d = F.softmax(d, dim=1)
        #d = 1-res/res.max()
        return d.max(axis=0).values

    def joint_decision(self, preds, distance):
        preds = torch.tensor(preds).cuda()
        distance = distance.cuda()
        return F.softmax(preds*distance, dim=1)#* distance
        #return F.softmax(F.softmax(preds, dim=1) * distance, dim=1)  # * distance
        #return 0.5*F.softmax(preds, dim=1) + 0.5*distance # * distance
    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders
        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))
        #temp = self.hparams['temp']
        # soft = nn.Softmax(dim=1)
        for step, ((src_x, src_y, id_source), (trg_x, _, id_target)) in joint_loader:
            """if src_x.shape[0] != trg_x.shape[0]:
                continue"""

            if src_x.shape[0] > trg_x.shape[0]:
                src_x = src_x[:trg_x.shape[0]]
                src_y = src_y[:trg_x.shape[0]]
            elif trg_x.shape[0] > src_x.shape[0]:
                trg_x = trg_x[:src_x.shape[0]]

            batch_size = len(src_x)
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(
                self.device)  # extract source features
            feature_ex_s = self.feature_extractor(src_x)
            feature_ex_t = self.feature_extractor(trg_x)

            before_lincls_feat_s, after_lincls_s = self.classifier(feature_ex_s)
            before_lincls_feat_t, after_lincls_t = self.classifier(feature_ex_t)

            norm_feat_s = F.normalize(before_lincls_feat_s)
            norm_feat_t = F.normalize(before_lincls_feat_t)

            # =====Source Supervision=====

            y_src = torch.eye(after_lincls_s.shape[-1], dtype=torch.int8).cuda()[src_y]
            self.memqueue_feat.update_queue(norm_feat_s, src_y.cuda())
            #self.memqueue_preds.update_queue(y_src, id_source.cuda())
            #print(y_src.shape)
            # print("y_src shape : ", y_src.shape)

            centr = self.class_centroids(norm_feat_s, y_src).detach().cpu()
            src_latent_cluster_copy = 0.9 * self.src_latent_cluster + 0.1 * centr.cuda()
            # TRG Centroids
            trg_latent_cluster_copy = self.update_centroids_target(norm_feat_t.detach().cpu(), self.trg_latent_cluster).cuda()
            before_lincls_feat_cen, after_lincls_cen = self.classifier(trg_latent_cluster_copy)
            trg_soft_cen = torch.nn.functional.softmax(after_lincls_cen).double()

            if self.hparams['joint_decision']:
                dist = self.compute_cluster_distance(norm_feat_t)
                soft_t = self.joint_decision(after_lincls_t, dist)
            else:
                soft_t = F.softmax(after_lincls_t)
            conf, preds = soft_t.max(dim=1)
            threshold = self.threshold_method(conf.detach().cpu().numpy())
            if step == 1: print('threshold : ', threshold)
            mask = (conf < threshold).cuda()
            # print("Detected ODD : ", mask.sum().item())

            # print("Number of ODD : ", (trg_label >= cls_output_dim).sum().item())
            # C0 = torch.zeros((len(norm_feat_s) + 1, len(norm_feat_t))).cuda()

            C0 = torch.zeros((len(norm_feat_s) + len(trg_latent_cluster_copy), len(norm_feat_t))).cuda()
            C_latent = src_latent_cluster_copy.mean(axis=0).unsqueeze(0)

            # nonOOD SRC
            C0[:len(norm_feat_s), ~mask] = torch.cdist(norm_feat_s, norm_feat_t[~mask])
            # OOD Dummy
            C0[len(norm_feat_s):, mask] = torch.cdist(trg_latent_cluster_copy, norm_feat_t[mask])

            maxc = torch.max(C0).item()  # *self.hparams['psi']
            # ODD SRC
            C0[:len(norm_feat_s), mask] = maxc
            # nonOOD Dummy
            C0[len(norm_feat_s):, ~mask] = maxc

            C1 = torch.zeros(C0.shape).cuda()
            C_preds = torch.ones((trg_latent_cluster_copy.shape[0], y_src.shape[-1])) / self.nb_classes
            C_preds = C_preds.cuda()

            # nonOOD SRC
            C1[:len(norm_feat_s), ~mask] = torch.cdist(y_src.float(), F.softmax(after_lincls_t)[~mask])
            # OOD Dummy
            C1[len(norm_feat_s):, mask] = torch.cdist(C_preds, F.softmax(after_lincls_t)[mask])
            maxc = torch.max(C1).item()  # *self.hparams['psi']
            # ODD SRC
            C1[:len(norm_feat_s), mask] = maxc
            # nonOOD Dummy
            C1[len(norm_feat_s):, ~mask] = maxc

            C = (self.hparams['alpha'] * C0 + self.hparams['lamb'] * C1)

            with torch.no_grad():

                a, b = ot.unif(C.size(0)), ot.unif(C.size(1))
                ratio = (mask.sum() / len(mask)).detach().cpu().item()
                a[:len(norm_feat_s)] = 0.5 / len(a[:len(norm_feat_s)])  #
                # a[:len(norm_feat_s)] = (1-ratio)/len(a[:len(norm_feat_s)])
                a[len(norm_feat_s):] = 0.5 / len(a[len(norm_feat_s):])  #
                # a[len(norm_feat_s):] = ratio/len(a[len(norm_feat_s):])
                # print(a.sum(), b.sum())

                gamma = ot.unbalanced.mm_unbalanced(a, b, C.detach().cpu().numpy(), reg_m=0.5)
                # gamma = ot.partial.partial_wasserstein(a, b, C.detach().cpu().numpy(), m=temp)
                # gamma = ot.sinkhorn(a, b, C.detach().cpu().numpy(), reg=0.01)
                # gamma = ot.emd(a, b, C.detach().cpu().numpy())
                # mass.append(gamma.sum())
                if step == 1:print('Mass : ', gamma.sum())
                gamma = torch.tensor(gamma).cuda()

            assert not torch.isnan(gamma).any()
            assert not torch.isnan(feature_ex_t).any()
            assert not torch.isnan(feature_ex_s).any()
            assert not torch.isnan(after_lincls_t).any()
            assert not torch.isnan(after_lincls_s).any()

            label_align_loss = (C * gamma)  # .sum()

            label_align_loss_nonOOD = label_align_loss[:len(norm_feat_s), ~mask].sum()
            label_align_loss_OOD = label_align_loss[len(norm_feat_s):, mask].sum()

            #label_align_loss = self.hparams['lamb'] * (label_align_loss_nonOOD + label_align_loss_OOD) / 2
            label_align_loss = (label_align_loss_nonOOD + label_align_loss_OOD) / 2

            criterion = nn.CrossEntropyLoss().cuda()
            loss_cls = self.hparams['src_weight'] * criterion(after_lincls_s, src_y)

            loss_all = loss_cls + label_align_loss #+ info_loss

            self.optimizer_feat.zero_grad()
            self.optimizer_cls.zero_grad()
            loss_all.backward()
            self.optimizer_feat.step()
            self.optimizer_cls.step()

            self.classifier.ProtoCLS.weight_norm()  # very important for proto-classifier

            losses = {'Total_loss': loss_all.item(), 'loss_cls': loss_cls.item(),
                      'loss_align': label_align_loss.item()}#, 'info_loss': info_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

    def evaluate(self, test_loader, trg_private_class, src=False):
        self.feature_extractor.eval()
        self.classifier.eval()

        total_loss, preds_list, labels_list = [], [], []



        '''dist_trg_tr = self.compute_cluster_distance(self.trg_feats_mem)
        soft_trg_tr = self.joint_decision(self.trg_preds_mem, dist_trg_tr)
        conf, preds = soft_trg_tr.max(dim=1)
        threshold = self.threshold_method(conf.detach().cpu().numpy())'''

        with torch.no_grad():
            for data, labels, ids in test_loader:
                data = data.float().to(self.device)
                '''if data.shape[0] == 1:
                    continue'''
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features = self.feature_extractor(data)
                before_lincls_feat_t, predictions = self.classifier(features)
                soft = F.softmax(predictions)

                if self.hparams['joint_decision']:
                    norm_feat_t = F.normalize(before_lincls_feat_t)
                    dist = self.compute_cluster_distance(norm_feat_t)
                    soft = self.joint_decision(predictions, dist)
                else:
                    soft = F.softmax(predictions)

                # preds_t = soft_t.argmax(dim=1)
                conf, preds = soft.max(dim=1)
                threshold = self.threshold_method(conf.detach().cpu().numpy())
                print("Finale Threshold : ", threshold)
                mask = conf < self.final_threshold #threshold
                if not src:
                    predictions[mask.squeeze()] *= 0

                if self.is_uniDA:
                    mask = labels >= predictions.shape[-1]
                    labels[mask] = predictions.shape[-1]

                mask = labels < predictions.shape[-1]
                # z = torch.zeros((len(predictions), 1))
                # predictions = torch.cat((predictions, z.to(predictions.device)), dim=1)
                loss = F.cross_entropy(predictions[mask], labels[mask])
                total_loss.append(loss.detach().cpu().item())
                # predictions = self.algorithm.correct_predictions(predictions)
                pred = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)

        loss = torch.tensor(total_loss).mean()  # average loss
        full_preds = torch.cat((preds_list))
        full_labels = torch.cat((labels_list))
        return loss, full_preds, full_labels

    def get_latent_features(self, dataloader):
        feature_set = []
        label_set = []
        logits_set = []
        self.feature_extractor.eval()
        self.classifier.eval()
        with torch.no_grad():
            for _, (data, label, ids) in enumerate(dataloader):
                data = data.to(self.device)
                '''if data.shape[0] == 1:
                    continue'''
                feature = self.feature_extractor(data)
                _, logit = self.classifier(feature)
                feature_set.append(feature.cpu())
                label_set.append(label.cpu())
                logits_set.append(logit.cpu())
            feature_set = torch.cat(feature_set, dim=0)
            feature_set = F.normalize(feature_set, p=2, dim=-1)
            label_set = torch.cat(label_set, dim=0)
            logits_set = torch.cat(logits_set, dim=0)
        return feature_set, label_set, logits_set

    def decision_function(self, preds):
        mask = preds.sum(axis=1) == 0.0
        confidence, pred = preds.max(dim=1)
        pred[mask] = -1
        return pred
