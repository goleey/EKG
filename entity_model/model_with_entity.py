import csv
import math
import sys
from pprint import pprint

sys.path.append("../")
from config import alp_config
import json
import pickle
import time
from logging import getLogger
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from dataset import collate_fn, index_word, index_entity
from utils_ssl import to_cuda
from tensorboardX import SummaryWriter
from torchsummary import summary
import sys
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, "../document_classification_word")
from hierarchical_att_model import HierAttNet

logger = getLogger()
writer = SummaryWriter()
max_number = torch.finfo(torch.float32).max
conf = alp_config()
class_num = 10


class MatchPyramidClassifier(object):

    def __init__(self, params):
        logger.info("Initializing GRUClassifier")
        self.params = params
        self.train_data = params.train_data
        self.test_data = params.test_data
        self.epoch_cnt = 0
        self.lamda_ = params.lamda
        self.relevance_model = word_relevance_model(params)
        self.optimizer = torch.optim.Adam(
            list(self.relevance_model.parameters()),
            lr=self.params.lr,
            weight_decay=0.01
        )
        self.vocab_size = len(params.word2idx)
        self.relevance_model.cuda()
        # for name, value in self.matchPyramid.named_parameters():
        #     print('name: {0},\t grad: {1}'.format(name, value.requires_grad))
        # exit()

    def run(self):

        for i in range(self.params.n_epochs):
            self.train()
            acc, precision, recall, f1, auc_score = self.evaluate()
            print(acc, precision, recall, f1, auc)
            self.epoch_cnt += 1
       

    def train(self):
        logger.info("Training in epoch %i" % self.epoch_cnt)
        self.relevance_model.train()
        data_loader = DataLoader(self.train_data,
                                 batch_size=self.params.batch_size,
                                 shuffle=True,
                                 collate_fn=collate_fn,
                                 drop_last=True,
                                 )
        pred_list = list()
        label_list = list()
        loss_list = list()
        for data_iter in data_loader:
            pair, f_user_text, t_user_text, f_time, t_time,\
            f_user_entity, t_user_entity, f_entity_time, t_entity_time, \
            labels = data_iter
            f_word_idx, f_time_batch, text_len_f, t_word_idx, t_time_batch, text_len_t, labels = \
                index_word(pair, f_user_text, f_time, t_user_text, t_time, labels, word2idx=self.params.word2idx,
                           max_word_length_f=self.params.max_word_len_f, max_word_length_t=self.params.max_word_len_t)
            # [print(x) for x in [f_word_idx, f_time_batch, t_word_idx, t_time_batch, labels]]
            # exit()
            f_entity_idx, f_entity_time_batch, t_entity_idx, t_entity_time_batch = index_entity(f_user_entity,
                                                                                                t_user_entity,
                                                                                                f_entity_time,
                                                                                                t_entity_time,
                                                                                                entity2idx=self.params.entity2idx,
                                                                                                max_entity_length_f=self.params.max_entity_length_f,
                                                                                                max_entity_length_t=self.params.max_entity_length_t)
            f_word_idx, f_time_batch, t_word_idx, t_time_batch, labels = to_cuda(f_word_idx, f_time_batch, t_word_idx,
                                                                                 t_time_batch, labels)
            f_entity_idx, f_entity_time_batch, t_entity_idx, t_entity_time_batch = to_cuda(f_entity_idx,
                                                                                           f_entity_time_batch,
                                                                                           t_entity_idx,
                                                                                           t_entity_time_batch)
            # [print(x.detach().cpu().tolist()) for x in [f_entity_idx, f_time_batch, t_entity_idx, t_time_batch]]
            # exit()
            foursquare_loc_dis, twitter_loc_dis, doc1_embedding, doc2_embedding, mp_output = self.relevance_model(
                f_word_idx, f_time_batch, text_len_f, t_word_idx, t_time_batch, text_len_t,
                f_entity_idx, f_entity_time_batch, t_entity_idx, t_entity_time_batch)
            # loss = F.cross_entropy(mp_output, labels) + self.lamda_ * F.mse_loss(foursquare_loc_dis,
            #                                                                      doc1_embedding) + self.lamda_ * F.mse_loss(
            #     twitter_loc_dis, doc2_embedding)
            loss = F.cross_entropy(mp_output, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            predictions = mp_output.data.max(1)[1]
            pred_list.extend(predictions.tolist())
            label_list.extend(labels.tolist())
            loss_list.append(loss.detach().cpu().numpy())
            # print(mp_output.data)
            # print(predictions)
            # print(labels.data.tolist())
            # print("-----")
            # exit()

        acc = accuracy_score(label_list, pred_list)
        f1 = f1_score(label_list, pred_list)
        losses = np.mean(loss_list)

        writer.add_scalar("train_acc", acc, self.epoch_cnt)
        writer.add_scalar("train_f1", f1, self.epoch_cnt)
        writer.add_scalar("train_losses", losses, self.epoch_cnt)
        logger.info("Train loss in epoch %i :%.4f" % (self.epoch_cnt, losses))
        logger.info("Train ACC score in epoch %i :%.4f" % (self.epoch_cnt, acc))
        logger.info("Train F1 score in epoch %i :%.4f" % (self.epoch_cnt, f1))

    def evaluate(self):
        logger.info("Evaluating in epoch %i" % self.epoch_cnt)
        self.relevance_model.cuda()
        data_loader = DataLoader(self.test_data,
                                 batch_size=self.params.batch_size,
                                 shuffle=False,
                                 collate_fn=collate_fn,
                                 drop_last=True)
        pred_list = list()
        label_list = list()
        loss_list = list()
        with torch.no_grad():
            for i, data_iter in enumerate(data_loader):
                pair, f_user_text, t_user_text, f_time, t_time, \
                f_user_entity, t_user_entity, f_entity_time, t_entity_time, \
                labels = data_iter
                f_word_idx, f_time_batch, text_len_f, t_word_idx, t_time_batch, text_len_t, labels = \
                    index_word(pair, f_user_text, f_time, t_user_text, t_time, labels, word2idx=self.params.word2idx,
                               max_word_length_f=self.params.max_word_len_f,
                               max_word_length_t=self.params.max_word_len_t)
                # [print(x) for x in [f_word_idx, f_time_batch, t_word_idx, t_time_batch, labels]]
                # exit()
                f_entity_idx, f_entity_time_batch, t_entity_idx, t_entity_time_batch = index_entity(f_user_entity,
                                                                                                    t_user_entity,
                                                                                                    f_entity_time,
                                                                                                    t_entity_time,
                                                                                                    entity2idx=self.params.entity2idx,
                                                                                                    max_entity_length_f=self.params.max_entity_length_f,
                                                                                                    max_entity_length_t=self.params.max_entity_length_t)
                f_word_idx, f_time_batch, t_word_idx, t_time_batch, labels = to_cuda(f_word_idx, f_time_batch,
                                                                                     t_word_idx,
                                                                                     t_time_batch, labels)
                f_entity_idx, f_entity_time_batch, t_entity_idx, t_entity_time_batch = to_cuda(f_entity_idx,
                                                                                               f_entity_time_batch,
                                                                                               t_entity_idx,
                                                                                               t_entity_time_batch)
                # [print(x.detach().cpu().tolist()) for x in [f_entity_idx, f_time_batch, t_entity_idx, t_time_batch]]
                # exit()
                foursquare_loc_dis, twitter_loc_dis, doc1_embedding, doc2_embedding, mp_output = self.relevance_model(
                    f_word_idx, f_time_batch, text_len_f, t_word_idx, t_time_batch, text_len_t,
                    f_entity_idx, f_entity_time_batch, t_entity_idx, t_entity_time_batch)
                # loss = F.cross_entropy(mp_output, labels) + self.lamda_ * F.mse_loss(foursquare_loc_dis, doc1_embedding) + self.lamda_ * F.mse_loss(twitter_loc_dis, doc2_embedding)
                loss = F.cross_entropy(mp_output, labels)
                predictions = mp_output.data.max(1)[1]
                pred_list.extend(predictions.tolist())
                label_list.extend(labels.tolist())
                loss_list.append(loss.detach().cpu().numpy())
        acc = accuracy_score(label_list, pred_list)
        precision = precision_score(label_list, pred_list)
        recall = recall_score(label_list, pred_list)
        f1 = f1_score(label_list, pred_list)
        fpr, tpr, thresholds = roc_curve(label_list, pred_list, pos_label=1)
        auc_score = auc(fpr, tpr)
        losses = np.mean(loss_list)
        writer.add_scalar("test_acc", acc, self.epoch_cnt)
        writer.add_scalar("test_f1", f1, self.epoch_cnt)
        writer.add_scalar("test_losses", losses, self.epoch_cnt)
        logger.info("Test loss in epoch %i :%.4f" % (self.epoch_cnt, losses))
        logger.info("Test ACC score in epoch %i :%.4f" % (self.epoch_cnt, acc))
        logger.info("Test F1 score in epoch %i :%.4f" % (self.epoch_cnt, f1))
        return (acc, precision, recall, f1, auc_score)


class MatchPyramid_identity(torch.nn.Module):

    def __init__(self, params):
        super().__init__()
        self.max_len_f = params.max_word_len_f
        self.max_len_t = params.max_word_len_t
        self.conv1_size = [int(_) for _ in params.conv1_size.split("_")]
        self.pool1_size = [int(_) for _ in params.pool1_size.split("_")]
        self.conv2_size = [int(_) for _ in params.conv2_size.split("_")]
        self.pool2_size = [int(_) for _ in params.pool2_size.split("_")]
        self.conv3_size = [int(_) for _ in params.conv3_size.split("_")]
        self.conv_mlp_out = params.conv_mlp_out
        # self.alpha = torch.nn.Parameter(torch.tensor(-0.0019))
        self.alpha = params.word_alpha
        self.filter_threshold = params.word_filter_threshold
        self.conv1 = torch.nn.Conv2d(in_channels=1,
                                     out_channels=self.conv1_size[-1],
                                     kernel_size=tuple(
                                         self.conv1_size[0:2]),
                                     padding=0,
                                     bias=True
                                     )
        # torch.nn.init.kaiming_normal_(self.conv1.weight)
        # torch.nn.init.ones_(self.conv1.weight)
        self.conv2 = torch.nn.Conv2d(in_channels=self.conv1_size[-1],
                                     out_channels=self.conv2_size[-1],
                                     kernel_size=tuple(
                                         self.conv2_size[0:2]),
                                     padding=0,
                                     bias=True
                                     )
        self.conv3 = torch.nn.Conv2d(in_channels=self.conv2_size[-1],
                                     out_channels=self.conv3_size[-1],
                                     kernel_size=tuple(
                                         self.conv3_size[0:2]),
                                     padding=0,
                                     bias=True
                                     )

        torch.nn.init.ones_(self.conv1.weight)
        torch.nn.init.ones_(self.conv2.weight)
        torch.nn.init.ones_(self.conv3.weight)
        # torch.nn.init.kaiming_normal_(self.conv1.weight)
        # torch.nn.init.kaiming_normal_(self.conv2.weight)
        # torch.nn.init.kaiming_normal_(self.conv3.weight)
        self.pool1 = torch.nn.AdaptiveMaxPool2d(tuple(self.pool1_size))
        self.pool2 = torch.nn.AdaptiveMaxPool2d(tuple(self.pool2_size))
        # self.pool1 = torch.nn.AdaptiveAvgPool2d(tuple(self.pool1_size))
        # self.pool2 = torch.nn.AdaptiveAvgPool2d(tuple(self.pool2_size))
        self.linear1 = torch.nn.Linear(self.pool2_size[0] * self.pool2_size[1] * self.conv3_size[-1],
                                       self.conv_mlp_out, bias=True)
        torch.nn.init.kaiming_normal_(self.linear1.weight)
        if logger:
            self.logger = logger
            self.logger.info("Hyper Parameters of MatchPyramid: %s" % json.dumps(
                {"Kernel": [self.conv1_size, self.conv2_size],
                 "Pooling": [self.pool1_size, self.pool2_size],
                 "MLP": self.conv_mlp_out}))

    def forward(self, x1, t1, x2, t2):
        # x1,x2:[batch, seq_len, dim_xlm]
        # t1,t2:[batch, seq_len]
        bs, seq_len_f, dim_xlm = x1.size()
        _, seq_len_t, _ = x2.size()
        pad1 = self.max_len_f - seq_len_f
        pad2 = self.max_len_t - seq_len_t
        # simi_img:[batch, 1, seq_len, seq_len]
        # cosine similarity
        x1_norm = x1.norm(dim=-1, keepdim=True)
        x1_norm = x1_norm + 1e-8
        x2_norm = x2.norm(dim=-1, keepdim=True)
        x2_norm = x2_norm + 1e-8
        x1 = x1 / x1_norm
        x2 = x2 / x2_norm
        simi_img = torch.matmul(x1, x2.transpose(1, 2))
        # print("img", simi_img.detach().cpu().numpy().tolist())
        simi_time = torch.abs(t1.unsqueeze(-1).expand(-1, -1, self.max_len_t) - t2.unsqueeze(-2)) / 3600.0
        # print(t1.detach().cpu().numpy().tolist())
        # print(t2.detach().cpu().numpy().tolist())
        # print("time", simi_time.detach().cpu().numpy().tolist())
        simi_time = torch.exp(self.alpha * simi_time.float())
        # simi_time = torch.where(simi_time <=120, torch.ones_like(simi_time,dtype=torch.float32).cuda(), torch.zeros_like(simi_time, dtype=torch.float32).cuda())

        # print("time_exp", simi_time.detach().cpu().numpy().tolist())
        simi_img = torch.mul(simi_img, simi_time)
        if pad1 != 0 or pad2 != 0:
            simi_img = F.pad(simi_img, (0, pad2, 0, pad1))
        assert simi_img.size() == (bs, self.max_len_f, self.max_len_t)
        # print(simi_img.detach().cpu().numpy().tolist())
        simi_img = torch.where(simi_img > self.filter_threshold, simi_img, torch.zeros_like(simi_img).cuda())
        # cnt = torch.where(simi_img > 0.99, torch.ones_like(simi_img).cuda(), torch.zeros_like(simi_img).cuda())
        # cnt = torch.sum(cnt, -1)
        # cnt = torch.sum(cnt, -1)
        # # print("cnt", cnt)
        # index_matrix = (simi_img >= 0.99).nonzero()

        # print(simi_img.detach().cpu().numpy().tolist())
        # exist or not?
        simi_img = simi_img.unsqueeze(1)
        # self.logger.info(simi_img.size())
        # [batch, 1, conv1_w, conv1_h]
        # print(simi_img.size())
        simi_img = F.relu(self.conv1(simi_img))
        # [batch, 1, pool1_w, pool1_h]
        # print(simi_img.size())
        simi_img = self.pool1(simi_img)
        # [batch, 1, conv2_w, conv2_h]
        # print("conv1", simi_img.detach().cpu().numpy().tolist())
        # print(simi_img.size())
        simi_img = F.relu(self.conv2(simi_img))
        # # [batch, 1, pool2_w, pool2_h]
        # print(simi_img.size())
        simi_img = self.pool2(simi_img)
        print(simi_img.size())
        # print("conv2", simi_img.detach().cpu().numpy().tolist())
        # assert simi_img.size()[1] == 1
        # [batch, pool1_w * pool1_h * conv2_out]
        # print(simi_img.size())
        # simi_img = torch.sum(simi_img, -2)
        # print("conv2-sum", simi_img.detach().cpu().numpy().tolist())
        simi_img = F.relu(self.conv3(simi_img))
        print("conv3", simi_img.size())
        print("conv3", simi_img.detach().cpu().numpy().tolist())
        simi_img = simi_img.squeeze(1).view(bs, -1)
        # print(simi_img.size())
        output = self.linear1(simi_img)
        print(output.detach().cpu().numpy().tolist())
        # print(output.size())
        # exit()
        return output


class MatchPyramid_entity(torch.nn.Module):

    def __init__(self, params):
        super().__init__()
        self.max_len_f = params.max_entity_length_f
        self.max_len_t = params.max_entity_length_t
        self.conv1_size = [int(_) for _ in params.entity_conv1_size.split("_")]
        self.pool1_size = [int(_) for _ in params.entity_pool1_size.split("_")]
        self.conv2_size = [int(_) for _ in params.entity_conv2_size.split("_")]
        self.pool2_size = [int(_) for _ in params.entity_pool2_size.split("_")]
        self.conv3_size = [int(_) for _ in params.entity_conv3_size.split("_")]
        self.conv_mlp_out = params.conv_entity_mlp_out
        # self.alpha = torch.nn.Parameter(torch.tensor(-0.0019))
        self.alpha = params.entity_alpha
        self.filter_threshold = params.entity_filter_threshold
        self.conv1 = torch.nn.Conv2d(in_channels=1,
                                     out_channels=self.conv1_size[-1],
                                     kernel_size=tuple(
                                         self.conv1_size[0:2]),
                                     padding=0,
                                     bias=True
                                     )
        # torch.nn.init.kaiming_normal_(self.conv1.weight)
        # torch.nn.init.ones_(self.conv1.weight)
        self.conv2 = torch.nn.Conv2d(in_channels=self.conv1_size[-1],
                                     out_channels=self.conv2_size[-1],
                                     kernel_size=tuple(
                                         self.conv2_size[0:2]),
                                     padding=0,
                                     bias=True
                                     )
        self.conv3 = torch.nn.Conv2d(in_channels=self.conv2_size[-1],
                                     out_channels=self.conv3_size[-1],
                                     kernel_size=tuple(
                                         self.conv3_size[0:2]),
                                     padding=0,
                                     bias=True
                                     )
        # torch.nn.init.ones_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        self.pool1 = torch.nn.AdaptiveMaxPool2d(tuple(self.pool1_size))
        self.pool2 = torch.nn.AdaptiveMaxPool2d(tuple(self.pool2_size))
        self.linear1 = torch.nn.Linear(self.pool2_size[0] * self.pool2_size[1] * self.conv3_size[-1],
                                       self.conv_mlp_out, bias=True)
        torch.nn.init.kaiming_normal_(self.linear1.weight)
        if logger:
            self.logger = logger
            self.logger.info("Hyper Parameters of MatchPyramid: %s" % json.dumps(
                {"Kernel": [self.conv1_size, self.conv2_size],
                 "Pooling": [self.pool1_size, self.pool2_size],
                 "MLP": self.conv_mlp_out}))

    def forward(self, x1, t1, x2, t2):
        # x1,x2:[batch, seq_len, dim_xlm]
        # t1,t2:[batch, seq_len]
        bs, seq_len_f, dim_xlm = x1.size()
        _, seq_len_t, _ = x2.size()
        pad1 = self.max_len_f - seq_len_f
        pad2 = self.max_len_t - seq_len_t
        # simi_img:[batch, 1, seq_len, seq_len]
        # cosine similarity
        x1_norm = x1.norm(dim=-1, keepdim=True)
        x1_norm = x1_norm + 1e-8
        x2_norm = x2.norm(dim=-1, keepdim=True)
        x2_norm = x2_norm + 1e-8
        x1 = x1 / x1_norm
        x2 = x2 / x2_norm
        simi_img = torch.matmul(x1, x2.transpose(1, 2))
        print(x1.detach().cpu().numpy().tolist())
        print(x2.detach().cpu().numpy().tolist())
        print("img", simi_img.detach().cpu().numpy().tolist())
        simi_time = torch.abs(t1.unsqueeze(-1).expand(-1, -1, self.max_len_t) - t2.unsqueeze(-2)) / 3600.0

        simi_time = torch.exp(self.alpha * simi_time.float())
        # simi_time = torch.where(simi_time <=720, torch.ones_like(simi_time,dtype=torch.float32).cuda(), torch.zeros_like(simi_time, dtype=torch.float32).cuda())
        print("simi_time", simi_time.detach().cpu().numpy().tolist())
        simi_img = torch.mul(simi_img, simi_time)
        if pad1 != 0 or pad2 != 0:
            simi_img = F.pad(simi_img, (0, pad2, 0, pad1))
        assert simi_img.size() == (bs, self.max_len_f, self.max_len_t)
        print(simi_img.detach().cpu().numpy().tolist())
        simi_img = torch.where(simi_img >= self.filter_threshold, simi_img, torch.zeros_like(simi_img).cuda())
        print(simi_img.detach().cpu().numpy().tolist())
        # cnt = torch.where(simi_img >= 0.9, torch.ones_like(simi_img).cuda(), torch.zeros_like(simi_img).cuda())
        # cnt = torch.sum(cnt, -1)
        # cnt = torch.sum(cnt, -1)
        # print("cnt", cnt.detach().cpu().numpy().tolist())
        # index_matrix = (simi_img >= 0.99).nonzero()

        # print(simi_img.detach().cpu().numpy().tolist())
        # exist or not?
        simi_img = simi_img.unsqueeze(1)
        # self.logger.info(simi_img.size())
        # [batch, 1, conv1_w, conv1_h]
        # print(simi_img.size())
        simi_img = F.relu(self.conv1(simi_img))
        # [batch, 1, pool1_w, pool1_h]
        # print(simi_img.size())
        simi_img = self.pool1(simi_img)
        # [batch, 1, conv2_w, conv2_h]
        print("conv1", simi_img.detach().cpu().numpy().tolist())
        # print(simi_img.size())
        simi_img = F.relu(self.conv2(simi_img))
        # # [batch, 1, pool2_w, pool2_h]
        # print(simi_img.size())
        simi_img = self.pool2(simi_img)
        # print(simi_img.size())
        print("conv2", simi_img.detach().cpu().numpy().tolist())
        # assert simi_img.size()[1] == 1
        # [batch, pool1_w * pool1_h * conv2_out]
        # print(simi_img.size())
        # simi_img = torch.sum(simi_img, -2)
        # print("conv2-sum", simi_img.detach().cpu().numpy().tolist())
        print("conv3", simi_img.size())
        print("conv3", simi_img.detach().cpu().numpy().tolist())
        simi_img = simi_img.squeeze(1).view(bs, -1)
        # print(simi_img.size())
        output = self.linear1(simi_img)
        print(output.detach().cpu().numpy().tolist())
        # print(output.size())
        # exit()
        return output


class word_relevance_model(torch.nn.Module):
    def __init__(self, params):
        super(word_relevance_model, self).__init__()
        word_dict = pd.read_csv(filepath_or_buffer=params.embedding_path, header=None, sep=" ",
                                quoting=csv.QUOTE_NONE).values[:, 1:]
        word_dict_len, word_embed_size = word_dict.shape
        word_dict = torch.from_numpy(word_dict.astype(np.float32))

        entity_emb = np.memmap(params.entity_emb_path, dtype='float32', mode='r')
        entity_emb = torch.cat(
            [torch.from_numpy(np.zeros([1, 50], dtype=np.float32)), torch.from_numpy(entity_emb).view(-1, 50), ], dim=0)
        entity_dict_len, entity_embed_size = entity_emb.shape

        self.word_lookup = torch.nn.Embedding(num_embeddings=word_dict_len,
                                              embedding_dim=word_embed_size).from_pretrained(word_dict,
                                                                                             freeze=True)
        self.entity_lookup = torch.nn.Embedding(num_embeddings=entity_dict_len,
                                                embedding_dim=entity_embed_size).from_pretrained(entity_emb,
                                                                                                 freeze=True)
        # self.matchparamid = MatchPyramid(params)
        self.doc_fc = torch.nn.Linear(word_embed_size, class_num)

        self.kd_model = params.kd_model
        # for name, parameters in self.kd_model.named_parameters():
        #     print(name, parameters.requires_grad)
        # exit()
        self.matchparamid_text = MatchPyramid_identity(params)
        self.matchparamid_entity = MatchPyramid_entity(params)
        self.mlp = torch.nn.Linear(params.conv_mlp_out + params.conv_entity_mlp_out, params.dim_out)
        # self.dropout_mlp = torch.nn.Dropout(p=0.01)
        self.dropout_mlp = torch.nn.Dropout(p=0.4)

        self.params = params
        # self.word_rev = torch.nn.ModuleList([self.gru, self.matchparamid])
        self._init()

    def _init(self):
        torch.nn.init.normal_(self.doc_fc.weight)
        torch.nn.init.constant_(self.doc_fc.bias.data, 0)
        torch.nn.init.kaiming_normal_(self.mlp.weight)

    def forward(self, f_word_idx, f_time_batch, word_len_day_f, t_word_idx, t_time_batch, word_len_day_t, f_entity_idx,
                f_entity_time_batch, t_entity_idx, t_entity_time_batch):
        sen1_embedding, sen2_embedding = self.word_lookup(f_word_idx), self.word_lookup(t_word_idx)
        entity1_embedding, entity2_embedding = self.entity_lookup(f_entity_idx), self.entity_lookup(t_entity_idx),
        _, word_len_t, _ = sen2_embedding.size()
        foursquare_loc_dis, twitter_loc_dis, doc1_embedding_student, doc2_embedding_student = "","","",""
        # doc1_embedding_student = []
        # doc2_embedding_student = []
        #
        # f_teacher_input = []
        # t_teacher_input = []
        #
        # for i in range(batch_size):
        #     sum_f = 0
        #     sum_t = 0
        #     for idx, length in enumerate(word_len_day_f[i]):
        #         word_idx = f_word_idx[i, sum_f:sum_f + length]
        #         word_emb = sen1_embedding[i, sum_f:sum_f + length, :]
        #         # print(i, sum_f, length)
        #         # print(word_emb.detach().cpu().numpy().tolist())
        #         doc_emb = torch.mean(word_emb, dim=0)
        #         doc1_embedding_student.append(doc_emb)
        #         f_teacher_input.append(word_idx)
        #         sum_f += length
        #     for idx, length in enumerate(word_len_day_t[i]):
        #         word_idx = t_word_idx[i, sum_t:sum_t + length]
        #         word_emb = sen2_embedding[i, sum_t:sum_t + length, :]
        #         doc_emb = torch.mean(word_emb, dim=0)
        #         doc2_embedding_student.append(doc_emb)
        #         t_teacher_input.append(word_idx)
        #         sum_t += length
        # doc1_embedding_student = torch.stack(doc1_embedding_student, dim=0)
        # doc2_embedding_student = torch.stack(doc2_embedding_student, dim=0)
        # #
        # # print("###")
        # # print(doc1_embedding_student.size(), doc1_embedding_student.detach().cpu().numpy().tolist())
        # # print(doc2_embedding_student.size(), doc2_embedding_student.detach().cpu().numpy().tolist())
        # doc1_embedding_student = self.doc_fc(doc1_embedding_student)
        # doc2_embedding_student = self.doc_fc(doc2_embedding_student)
        #
        # # print(doc1_embedding_student.size(), doc1_embedding_student)
        # # print(doc2_embedding_student.size(), doc2_embedding_student)
        #
        # # [print(x.size(),x) for x in f_teacher_input]
        # # print("--")
        # # [print(x.size(),x) for x in t_teacher_input]
        # f_teacher_input = pad_sequence(f_teacher_input, batch_first=True)
        # t_teacher_input = pad_sequence(t_teacher_input, batch_first=True)
        #
        # # print(f_teacher_input.size(), f_teacher_input)
        # # print(t_teacher_input.size(), t_teacher_input)
        # # exit()
        # foursquare_loc_dis, _ = self.kd_model("", f_teacher_input)
        # twitter_loc_dis, _ = self.kd_model("", t_teacher_input)
        # # print(foursquare_loc_dis.size(), foursquare_loc_dis)
        # # print(twitter_loc_dis.size(), twitter_loc_dis)
        # # foursquare_loc_dis = foursquare_loc_dis.view(batch_size, seq_len, class_num)
        # # twitter_loc_dis = twitter_loc_dis.view(batch_size, seq_len, class_num)
        # # loc_dis = jsd(foursquare_loc_dis, twitter_loc_dis)
        mp_output_word = self.matchparamid_text(sen1_embedding, f_time_batch, sen2_embedding, t_time_batch)
        mp_output_entity = self.matchparamid_entity(entity1_embedding, f_entity_time_batch, entity2_embedding, t_entity_time_batch)
        # mp_out = self.mlp(mp_output_word+mp_output_entity)
        # attention mechanism
        # mp_out = torch.stack([mp_output_word, mp_output_entity], dim=1)
        # f_output = matrix_mul(mp_out, self.context_weight).squeeze(-1).permute(1, 0)
        # f_output = F.softmax(f_output, dim=-1)
        # mp_out = element_wise_mul(f_output, mp_out.permute(1, 0))
        mp_out = self.mlp(torch.cat([mp_output_word, mp_output_entity], dim=-1))

        mp_out = self.dropout_mlp(mp_out)
        return foursquare_loc_dis, twitter_loc_dis, doc1_embedding_student, doc2_embedding_student, mp_out
