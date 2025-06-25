import faiss
import ot
import torch
import torch.nn.functional as F
from easydl import variable_to_numpy
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch import nn as nn

import random
import os
import sys
import logging
import numpy as np
import pandas as pd
from shutil import copy
from datetime import datetime

from skorch import NeuralNetClassifier  # for DIV Risk
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def fix_randomness(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def create_logger(exp_log_dir):
    log_dir = os.path.join(exp_log_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_file_name = os.path.join(log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    log_file_name = os.path.join(log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    return logger

def starting_task(data_type, da_method, exp_log_dir, src_id, tgt_id, run_id, logger):
    log_dir = os.path.join(exp_log_dir, src_id + "_to_" + tgt_id + "_run_" + str(run_id))
    os.makedirs(log_dir, exist_ok=True)
    log_file_name = os.path.join(log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger.debug("=" * 45)
    logger.debug(f'Dataset: {data_type}')
    logger.debug(f'Method:  {da_method}')
    logger.debug("=" * 45)
    logger.debug(f'Source: {src_id} ---> Target: {tgt_id}')
    logger.debug(f'Run ID: {run_id}')
    logger.debug("=" * 45)
    return log_dir

def starting_logs(data_type, da_method, exp_log_dir, src_id, tgt_id, run_id):
    log_dir = os.path.join(exp_log_dir, src_id + "_to_" + tgt_id + "_run_" + str(run_id))
    os.makedirs(log_dir, exist_ok=True)
    log_file_name = os.path.join(log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Dataset: {data_type}')
    logger.debug(f'Method:  {da_method}')
    logger.debug("=" * 45)
    logger.debug(f'Source: {src_id} ---> Target: {tgt_id}')
    logger.debug(f'Run ID: {run_id}')
    logger.debug("=" * 45)
    return logger, log_dir


def save_checkpoint(home_path, algorithm, log_dir, last_model, best_model):
    save_dict = {
        "last": last_model,
        "best": best_model
    }
    # save classification report
    save_path = os.path.join(home_path, log_dir, f"checkpoint.pt")

    torch.save(save_dict, save_path)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)


def _calc_metrics(pred_labels, true_labels, log_dir, home_path, target_names):
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    r = classification_report(true_labels, pred_labels, target_names=target_names, digits=6, output_dict=True)

    df = pd.DataFrame(r)
    accuracy = accuracy_score(true_labels, pred_labels)
    df["accuracy"] = accuracy
    df = df * 100

    # save classification report
    file_name = "classification_report.xlsx"
    report_Save_path = os.path.join(home_path, log_dir, file_name)
    df.to_excel(report_Save_path)

    return accuracy * 100, r["macro avg"]["f1-score"] * 100


def copy_Files(destination):
    destination_dir = os.path.join(destination, "MODEL_BACKUP_FILES")
    os.makedirs(destination_dir, exist_ok=True)
    copy("main.py", os.path.join(destination_dir, "main.py"))
    copy("utils.py", os.path.join(destination_dir, "utils.py"))
    copy(f"trainer.py", os.path.join(destination_dir, f"trainer.py"))
    copy(f"same_domain_trainer.py", os.path.join(destination_dir, f"same_domain_trainer.py"))
    copy("dataloader/dataloader.py", os.path.join(destination_dir, "dataloader.py"))
    copy(f"models/models.py", os.path.join(destination_dir, f"models.py"))
    copy(f"models/loss.py", os.path.join(destination_dir, f"loss.py"))
    copy("algorithms/algorithms.py", os.path.join(destination_dir, "algorithms.py"))
    copy(f"configs/data_model_configs.py", os.path.join(destination_dir, f"data_model_configs.py"))
    copy(f"configs/hparams.py", os.path.join(destination_dir, f"hparams.py"))
    copy(f"configs/sweep_params.py", os.path.join(destination_dir, f"sweep_params.py"))




def get_iwcv_value(weight, error):
    N, d = weight.shape
    _N, _d = error.shape
    assert N == _N and d == _d, 'dimension mismatch!'
    weighted_error = weight * error
    return np.mean(weighted_error)


def get_dev_value(weight, error):
    """
    :param weight: shape [N, 1], the importance weight for N source samples in the validation set
    :param error: shape [N, 1], the error value for each source sample in the validation set
    (typically 0 for correct classification and 1 for wrong classification)
    """
    N, d = weight.shape
    _N, _d = error.shape
    assert N == _N and d == _d, 'dimension mismatch!'
    weighted_error = weight * error
    cov = np.cov(np.concatenate((weighted_error, weight), axis=1), rowvar=False)[0][1]
    var_w = np.var(weight, ddof=1)
    eta = - cov / var_w
    return np.mean(weighted_error) + eta * np.mean(weight) - eta


class simple_MLP(nn.Module):
    def __init__(self, inp_units, out_units=2):
        super(simple_MLP, self).__init__()

        self.dense0 = nn.Linear(inp_units, inp_units // 2)
        self.nonlin = nn.ReLU()
        self.output = nn.Linear(inp_units // 2, out_units)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, **kwargs):
        x = self.nonlin(self.dense0(x))
        x = self.softmax(self.output(x))
        return x


def get_weight_gpu(source_feature, target_feature, validation_feature, configs, device):
    """
    :param source_feature: shape [N_tr, d], features from training set
    :param target_feature: shape [N_te, d], features from test set
    :param validation_feature: shape [N_v, d], features from validation set
    :return:
    """
    import copy
    N_s, d = source_feature.shape
    N_t, _d = target_feature.shape
    source_feature = copy.deepcopy(source_feature.detach().cpu())  # source_feature.clone()
    target_feature = copy.deepcopy(target_feature.detach().cpu())  # target_feature.clone()
    source_feature = source_feature.to(device)
    target_feature = target_feature.to(device)
    all_feature = torch.cat((source_feature, target_feature), dim=0)
    all_label = torch.from_numpy(np.asarray([1] * N_s + [0] * N_t, dtype=np.int32)).long()

    feature_for_train, feature_for_test, label_for_train, label_for_test = train_test_split(all_feature, all_label,
                                                                                            train_size=0.8)
    learning_rates = [1e-1, 5e-2, 1e-2]
    val_acc = []
    domain_classifiers = []

    for lr in learning_rates:
        domain_classifier = NeuralNetClassifier(
            simple_MLP,
            module__inp_units=configs.final_out_channels * configs.features_len,
            max_epochs=30,
            lr=lr,
            device=device,
            # Shuffle training data on each epoch
            iterator_train__shuffle=True,
            callbacks="disable"
        )
        domain_classifier.fit(feature_for_train.float(), label_for_train.long())
        output = domain_classifier.predict(feature_for_test)
        acc = np.mean((label_for_test.numpy() == output).astype(np.float32))
        val_acc.append(acc)
        domain_classifiers.append(domain_classifier)

    index = val_acc.index(max(val_acc))
    domain_classifier = domain_classifiers[index]

    domain_out = domain_classifier.predict_proba(validation_feature.to(device).float())
    return domain_out[:, :1] / domain_out[:, 1:] * N_s * 1.0 / N_t


def calc_dev_risk(target_model, src_train_dl, tgt_train_dl, src_valid_dl, configs, device):
    src_train_feats = target_model.feature_extractor(src_train_dl.dataset.x_data.float().to(device))
    tgt_train_feats = target_model.feature_extractor(tgt_train_dl.dataset.x_data.float().to(device))
    src_valid_feats = target_model.feature_extractor(src_valid_dl.dataset.x_data.float().to(device))
    src_valid_pred = target_model.classifier(src_valid_feats)

    dev_weights = get_weight_gpu(src_train_feats.to(device), tgt_train_feats.to(device),
                                 src_valid_feats.to(device), configs, device)
    dev_error = F.cross_entropy(src_valid_pred, src_valid_dl.dataset.y_data.long().to(device), reduction='none')
    dev_risk = get_dev_value(dev_weights, dev_error.unsqueeze(1).detach().cpu().numpy())
    # iwcv_risk = get_iwcv_value(dev_weights, dev_error.unsqueeze(1).detach().cpu().numpy())
    return dev_risk


def calculate_risk(target_model, risk_dataloader, device):
    if type(risk_dataloader) == tuple:
        x_data = torch.cat((risk_dataloader[0].dataset.x_data, risk_dataloader[1].dataset.x_data), axis=0)
        y_data = torch.cat((risk_dataloader[0].dataset.y_data, risk_dataloader[1].dataset.y_data), axis=0)
    else:
        x_data = risk_dataloader.dataset.x_data
        y_data = risk_dataloader.dataset.y_data

    feat = target_model.feature_extractor(x_data.float().to(device))
    pred = target_model.classifier(feat)
    cls_loss = F.cross_entropy(pred, y_data.long().to(device))
    return cls_loss.item()

class DictAsObject:
    def __init__(self, d):
        self.__dict__ = d

    def __getattr__(self, name):
        try:
            return self.__dict__[name]
        except KeyError:
            raise AttributeError(f"'DictAsObject' object has no attribute '{name}'")


#For UniOT

def sinkhorn(out, epsilon, sinkhorn_iterations):
    """
    from https://github.com/facebookresearch/swav
    """
    Q = torch.exp(out / epsilon).t()  # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1]  # number of samples to assign
    K = Q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    # Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()


def ubot_CCD(sim, beta, fake_size=0, fill_size=0, mode='minibatch', stopThr=1e-4):
    # fake_size (Adaptive filling) + fill_size (memory queue filling) + mini-batch size
    M = -sim
    alpha = ot.unif(sim.size(0))

    Q_st = ot.unbalanced.sinkhorn_knopp_unbalanced(alpha, beta, M.detach().cpu().numpy(),
                                                   reg=0.01, reg_m=0.5, stopThr=stopThr)
    Q_st = torch.from_numpy(Q_st).float().cuda()

    # make sum equals to 1
    sum_pi = torch.sum(Q_st)
    Q_st_bar = Q_st / sum_pi

    # highly confident target samples selected by statistics mean
    if mode == 'minibatch':
        Q_anchor = Q_st_bar[fake_size + fill_size:, :]
    if mode == 'all':
        Q_anchor = Q_st_bar

    # confidence score w^t_i
    wt_i, pseudo_label = torch.max(Q_anchor, 1)
    # confidence score w^s_j
    ws_j = torch.sum(Q_st_bar, 0)

    # filter by statistics mean
    uniformed_index = Q_st_bar.size(1)
    conf_label = torch.where(wt_i > 1 / Q_st_bar.size(0), pseudo_label, uniformed_index)
    high_conf_label = conf_label.clone()
    source_private_label = torch.nonzero(ws_j < 1 / Q_st_bar.size(1))
    for i in source_private_label:
        high_conf_label = torch.where(high_conf_label == i, uniformed_index, high_conf_label)
    high_conf_label_id = torch.nonzero(high_conf_label != uniformed_index).view(-1)

    # for adaptive update
    new_beta = torch.sum(Q_st_bar, 0).cpu().numpy()

    return high_conf_label_id, high_conf_label, conf_label, new_beta

def ubot_CCD2(sim, beta, fake_size=0, fill_size=0, mode='minibatch', stopThr=1e-4):
    # fake_size (Adaptive filling) + fill_size (memory queue filling) + mini-batch size
    M = -sim
    alpha = ot.unif(sim.size(0))

    Q_st = ot.unbalanced.mm_unbalanced(alpha, beta, M.detach().cpu().numpy(), reg_m=0.5, stopThr=stopThr)
    Q_st = torch.from_numpy(Q_st).float().cuda()

    # make sum equals to 1
    sum_pi = torch.sum(Q_st)
    Q_st_bar = Q_st / sum_pi

    # highly confident target samples selected by statistics mean
    if mode == 'minibatch':
        Q_anchor = Q_st_bar[fake_size + fill_size:, :]
    if mode == 'all':
        Q_anchor = Q_st_bar

    # confidence score w^t_i
    wt_i, pseudo_label = torch.max(Q_anchor, 1)
    # confidence score w^s_j
    ws_j = torch.sum(Q_st_bar, 0)

    # filter by statistics mean
    uniformed_index = Q_st_bar.size(1)
    conf_label = torch.where(wt_i > 1 / Q_st_bar.size(0), pseudo_label, uniformed_index)
    high_conf_label = conf_label.clone()
    source_private_label = torch.nonzero(ws_j < 1 / Q_st_bar.size(1))
    for i in source_private_label:
        high_conf_label = torch.where(high_conf_label == i, uniformed_index, high_conf_label)
    high_conf_label_id = torch.nonzero(high_conf_label != uniformed_index).view(-1)

    # for adaptive update
    new_beta = torch.sum(Q_st_bar, 0).cpu().numpy()

    return high_conf_label_id, high_conf_label, conf_label, new_beta
def adaptive_filling2(ubot_feature_t, source_prototype, gamma, beta, fill_size, stopThr=1e-4):
    sim = torch.matmul(ubot_feature_t, source_prototype.t())
    max_sim, _ = torch.max(sim, 1)
    pos_id = torch.nonzero(max_sim > gamma).reshape(-1)
    pos_rate = pos_id.size(0) / max_sim.size(0)
    pos_num = pos_id.size(0)
    neg_num = max_sim.size(0) - pos_num
    if pos_rate <= 0.5:
        # positive filling
        fake_size = neg_num - pos_num
        if fake_size > 0:
            # do 1st OT find top confident target samples
            high_conf_label_id, _, __, ___ = ubot_CCD2(sim, beta, fake_size=0, fill_size=fill_size,
                                                      mode='all', stopThr=stopThr)
            if high_conf_label_id.size(0) > 0:
                select_id = torch.randint(0, high_conf_label_id.size(0), (fake_size,)).cuda()
                fill_pos = sim[high_conf_label_id[select_id], :]
                newsim = torch.cat([fill_pos, sim], 0)
            else:
                fake_size = 0
                newsim = sim
        else:
            newsim = sim
    else:
        # negative filling
        fake_size = pos_num - neg_num
        if fake_size > 0:
            farthest_sproto_id = torch.argmin(sim, 1)
            fake_private = 0.5 * ubot_feature_t + 0.5 * source_prototype.data[farthest_sproto_id, :]
            fake_private = F.normalize(fake_private)
            select_id = torch.randint(0, fake_private.size(0), (fake_size,)).cuda()
            fill_neg = fake_private[select_id, :]
            fake_sim = torch.matmul(fill_neg, source_prototype.t())
            newsim = torch.cat([fake_sim, sim], 0)
        else:
            newsim = sim

    return newsim, fake_size
def adaptive_filling(ubot_feature_t, source_prototype, gamma, beta, fill_size, stopThr=1e-4):
    sim = torch.matmul(ubot_feature_t, source_prototype.t())
    max_sim, _ = torch.max(sim, 1)
    pos_id = torch.nonzero(max_sim > gamma).reshape(-1)
    pos_rate = pos_id.size(0) / max_sim.size(0)
    pos_num = pos_id.size(0)
    neg_num = max_sim.size(0) - pos_num
    if pos_rate <= 0.5:
        # positive filling
        fake_size = neg_num - pos_num
        if fake_size > 0:
            # do 1st OT find top confident target samples
            high_conf_label_id, _, __, ___ = ubot_CCD(sim, beta, fake_size=0, fill_size=fill_size,
                                                      mode='all', stopThr=stopThr)
            if high_conf_label_id.size(0) > 0:
                select_id = torch.randint(0, high_conf_label_id.size(0), (fake_size,)).cuda()
                fill_pos = sim[high_conf_label_id[select_id], :]
                newsim = torch.cat([fill_pos, sim], 0)
            else:
                fake_size = 0
                newsim = sim
        else:
            newsim = sim
    else:
        # negative filling
        fake_size = pos_num - neg_num
        if fake_size > 0:
            farthest_sproto_id = torch.argmin(sim, 1)
            fake_private = 0.5 * ubot_feature_t + 0.5 * source_prototype.data[farthest_sproto_id, :]
            fake_private = F.normalize(fake_private)
            select_id = torch.randint(0, fake_private.size(0), (fake_size,)).cuda()
            fill_neg = fake_private[select_id, :]
            fake_sim = torch.matmul(fill_neg, source_prototype.t())
            newsim = torch.cat([fake_sim, sim], 0)
        else:
            newsim = sim

    return newsim, fake_size

def run_kmeans(L2_feat, ncentroids, init_centroids=None, seed=None, gpu=False, min_points_per_centroid=1):
    if seed is None:
        seed = int(os.environ['PYTHONHASHSEED'])
    dim = L2_feat.shape[1]
    kmeans = faiss.Kmeans(d=dim, k=ncentroids, seed=seed, gpu=gpu, niter=20, verbose=False, \
                        nredo=5, min_points_per_centroid=min_points_per_centroid, spherical=True)
    if torch.is_tensor(L2_feat):
        L2_feat = variable_to_numpy(L2_feat)
    kmeans.train(L2_feat, init_centroids=init_centroids)
    _, pred_centroid = kmeans.index.search(L2_feat, 1)
    pred_centroid = np.squeeze(pred_centroid)
    return pred_centroid, kmeans.centroids