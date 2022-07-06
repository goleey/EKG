# 从word_relevance_model修改而来，过滤推特的文本，只保留tf-idf有用的部分
import os
import re
import sys
import torch
import numpy as np
import argparse

sys.path.append("../")
from document_classification_word.hierarchical_att_model import HierAttNet
from utils_ssl import init_logger, load_w2v

# from dataset_flag import dataset
from dataset import dataset
from entity_model import model_kd_embedding,model_with_teacher,model_only_entity, \
    model_with_entity,model_naive_entity,model_only_word, model_with_entity_attention,model_with_entity_bn,\
    model_with_entity_staircase,model_with_entity_linear
from config import alp_config
import os
import pickle as pkl

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(
    description='Train and Evaluate MatchPyramid on MSRP dataset')

# main parameters
parser.add_argument("--training_ratio", type=float, default=1.0,
                    help="")
parser.add_argument("--delta_t", type=int, default=5,
                    help="")
parser.add_argument("--filter_threshold", type=float, default=0.9,
                    help="")
parser.add_argument("--lamda", type=float, default=0.05,
                    help="")
parser.add_argument("--data_path", type=str, default="/home/shenhuawei/gaohao/DCMH-ALP/data/foursquare_twitter/",
                    help="")
parser.add_argument("--dump_path", type=str, default="./dump/",
                    help="")
parser.add_argument("--embedding_path", type=str,
                    default="/home/shenhuawei/gaohao/DCMH-ALP/data/foursquare_twitter/glove.6B.50d.txt",
                    help="")
# parser.add_argument("--filter_word_method", type=str, choices=["tfidf", "lda", "dc"], default="dc",
#                     help="")
parser.add_argument("--max_seq_len", type=int, default=10,
                    help="")
parser.add_argument("--max_word_len_f", type=int, default=500,
                    help="")
parser.add_argument("--max_word_len_t", type=int, default=3000,
                    help="")
parser.add_argument("--max_entity_length_f", type=int, default=50,
                    help="")
parser.add_argument("--max_entity_length_t", type=int, default=200,
                    help="")
# parser.add_argument("--max_word_len", type=int, default=10,
#                     help="")
parser.add_argument("--batch_size", type=int, default=2,
                    help="")
parser.add_argument("--lr", type=float, default=0.01,
                    help="")
parser.add_argument("--n_epochs", type=int, default=150,
                    help="")
# word convolution model parameters
parser.add_argument("--conv1_size", type=str, default="3_3_1",
                    help="")
parser.add_argument("--pool1_size", type=str, default="400_400",
                    help="")
parser.add_argument("--conv2_size", type=str, default="3_3_8",
                    help="")
parser.add_argument("--pool2_size", type=str, default="20_20",
                    help="")
parser.add_argument("--conv3_size", type=str, default="1_1_8",
                    help="")
parser.add_argument("--word_alpha", type=float, default=-0.0006,
                    help="")
parser.add_argument("--word_filter_threshold", type=float, default=0.8,
                    help="")
parser.add_argument("--conv_mlp_out", type=int, default=5,
                    help="")
# entity convolution model parameters
parser.add_argument("--entity_conv1_size", type=str, default="3_3_1",
                    help="")
parser.add_argument("--entity_pool1_size", type=str, default="20_20",
                    help="")
parser.add_argument("--entity_conv2_size", type=str, default="3_3_8",
                    help="")
parser.add_argument("--entity_pool2_size", type=str, default="10_10",
                    help="")
parser.add_argument("--entity_conv3_size", type=str, default="1_1_8",
                    help="")
parser.add_argument("--entity_alpha", type=float, default=-0.0001,
                    help="")
parser.add_argument("--entity_filter_threshold", type=float, default=0.9,
                    help="")
parser.add_argument("--conv_entity_mlp_out", type=int, default=5,
                    help="")
# gru model parameters
parser.add_argument("--dim_embedding", type=int, default=50,
                    help="")
parser.add_argument("--gru_hidden", type=int, default=10,
                    help="")
parser.add_argument("--dim_mapping_out", type=int, default=1,
                    help="")
# final mp
parser.add_argument("--dim_out", type=int, default=2,
                    help="")
parser.add_argument("--model", type=str,
                    choices=["model_kd_embedding", "model_with_teacher", "model_with_entity",
                             "model_only_entity", "model_naive_entity", "model_only_word", "model_with_entity_attention",
                             "model_with_entity_bn", "model_with_entity_staircase", "model_with_entity_linear"],
                    default="model_kd_embedding",
                    help="")

parser.add_argument("--kd_model", type=str, choices=["foursquare_dc_model"], default="foursquare_dc_model",
                    help="")
# parse arguments
params = parser.parse_args()

# check parameters

logger = init_logger(params)

params.word2idx, params.idx2word, params.glove_weight = load_w2v(params.embedding_path, params.dim_embedding)

conf = alp_config(params.data_path)
train_data = dataset(conf.x_train, conf.y_train, conf.foursquare_checkins, conf.twitter_uid_tweeets,
                     conf.f_entityid_time_pair, conf.t_entityid_time_pair, params.delta_t)
test_data = dataset(conf.x_test, conf.y_test, conf.foursquare_checkins, conf.twitter_uid_tweeets,
                    conf.f_entityid_time_pair, conf.t_entityid_time_pair, params.delta_t)

params.train_data = train_data
params.test_data = test_data
if params.kd_model == "foursquare_dc_model":
    params.kd_model = HierAttNet(50, 50, 128, 10,
                                 "/home/shenhuawei/gaohao/DCMH-ALP/data/foursquare_twitter/glove.6B.50d.txt", 5, 300)
    params.kd_model.load_state_dict(
        torch.load("/home/shenhuawei/gaohao/mulmodal-alp-mlp/document_classification_word/foursquare/han_model")[
            "state_dict"])
    for name, parameters in params.kd_model.named_parameters():
        # print(name, parameters.requires_grad)
        parameters.requires_grad = False

params.entity_emb_path = "/home/shenhuawei/gaohao/Wikidata/embeddings/dimension_50/transe/entity2vec.bin"
entity2idx = {}
with open("/home/shenhuawei/gaohao/Wikidata/knowledge_graphs/entity2id.txt") as f:
    lines = f.readlines()
    for line in lines[1:]:
        # print(line.strip())
        # print(re.split("\s+", line.strip()))
        x = re.split("\s+", line.strip())
        entity2idx[x[0]] = int(x[1])
params.entity2idx = entity2idx
if params.model == "model_kd_embedding":
    mp_model = model_kd_embedding.MatchPyramidClassifier(params)
    mp_model.run()
elif params.model == "model_with_teacher":
    mp_model = model_with_teacher.MatchPyramidClassifier(params)
    mp_model.run()
elif params.model == "model_only_entity":
    mp_model = model_only_entity.MatchPyramidClassifier(params)
    mp_model.run()
elif params.model == "model_with_entity":
    mp_model = model_with_entity.MatchPyramidClassifier(params)
    mp_model.run()
elif params.model == "model_naive_entity":
    mp_model = model_naive_entity.MatchPyramidClassifier(params)
    mp_model.run()
elif params.model == "model_only_word":
    mp_model = model_only_word.MatchPyramidClassifier(params)
    mp_model.run()
elif params.model == "model_with_entity_attention":
    mp_model = model_with_entity_attention.MatchPyramidClassifier(params)
    mp_model.run()
elif params.model == "model_with_entity_bn":
    mp_model = model_with_entity_bn.MatchPyramidClassifier(params)
    mp_model.run()
elif params.model == "model_with_entity_staircase":
    mp_model = model_with_entity_staircase.MatchPyramidClassifier(params)
    mp_model.run()
elif params.model == "model_with_entity_linear":
    mp_model = model_with_entity_linear.MatchPyramidClassifier(params)
    mp_model.run()
# elif params.model == "model_kd_embedding_delta_t":
#     print(params.delta_t)
#     mp_model = model_kd_embedding_delta_t.MatchPyramidClassifier(params)
#     mp_model.run()
# elif params.model == "model_kd_embedding_thre_r":
#     mp_model = model_kd_embedding_thre_r.MatchPyramidClassifier(params)
#     mp_model.run()
# elif params.model == "model_kd_embedding_coe_lamda":
#     mp_model = model_kd_embedding_coe_lamda.MatchPyramidClassifier(params)
#     mp_model.run()
# elif params.model == "model_kd_embedding_wo_tm":
#     mp_model = model_kd_embedding_wo_tm.MatchPyramidClassifier(params)
#     mp_model.run()
# elif params.model == "model_kd_embedding_training_ratio":
#     mp_model = model_kd_embedding_training_ratio.MatchPyramidClassifier(params)
#     mp_model.run()
# elif params.model == "model_kd_embedding_casestudy":
#     mp_model = model_kd_embedding_casestudy.MatchPyramidClassifier(params)
#     mp_model.run()
