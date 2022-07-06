import codecs
import math
import time
from collections import Counter
from pprint import pprint

import nltk
import torch
import numpy as np
from torch.utils.data import Dataset
import pickle as pkl
import sys
from utils_ssl import datetime2stamp
from nltk.corpus import stopwords, wordnet
import re, string
from nltk import word_tokenize, sent_tokenize, WordNetLemmatizer
from bs4 import BeautifulSoup
import scipy.stats as st
from nltk import pos_tag
from nltk.stem import PorterStemmer
import spacy
from spacy import displacy
import en_core_web_sm

nlp = en_core_web_sm.load()
ent_categories = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", 'WORK_OF_ART', 'LAW']

punctuation = '~`!#$%^&*()+-=|\';":/.,?><~·！#￥%……&*（）——+=“：’；、。，？》《{}'
my_stopwords = stopwords.words("english")
for w in ['!', ',', '.', '?', '-s', '-ly', '</s>', 's', "RT", "rt", ]:
    my_stopwords.append(w)
my_stopwords = set(my_stopwords)

# remove_list_f = ["bardinass", "mfcmag", "jakubptacin", "miriamtorrente", "alex_baburin", "kkoba33"]
# remove_list_t = ["chellsdragonfly", "adamgranak", "frosty5798", "zinan76", "alex_baburin", "'alexissunshine'"]
remove_pairs = pkl.load(open("./null_pair", "rb"))
remove_list_f = [x[0] for x in remove_pairs]
remove_list_t = [x[1] for x in remove_pairs]
# tf-idf
# word_filtered_dict = set(np.load('/home/shenhuawei/gaohao/DCMH-ALP/data/foursquare_twitter/word_dict.npy', allow_pickle=True).item().keys())
# lda
# word_filtered_dict = pkl.load(open("/home/shenhuawei/gaohao/DCMH-ALP/data/foursquare_twitter/common_lda_words.pkl", "rb"))
# document classification-foursquare
word_filtered_dict = pkl.load(
    open("/home/shenhuawei/gaohao/DCMH-ALP/data/foursquare/document_classification_words.pkl", "rb"))


def get_entity(text):
    try:
        l = [(x.text, x.label_) for x in nlp(text).ents]
        return l
    except:
        pass


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


# cluster the text in 24 housrs
def _gen_oneday(f_pairs):
    threshold = 3600 * 24.0
    i = 0
    f_oneday = {}
    while i < len(f_pairs):
        if i == len(f_pairs) - 1:
            f_oneday.setdefault(time.strftime('%Y-%m-%d 00:00:00', time.localtime(f_pairs[i][1])), []).append(
                f_pairs[i][0])
            break
        else:
            jump = 1
            f_oneday.setdefault(time.strftime('%Y-%m-%d 00:00:00', time.localtime(f_pairs[i][1])), []).append(
                f_pairs[i][0])
            for index_now in range(i + 1, len(f_pairs)):
                if math.fabs(f_pairs[i][1] - f_pairs[index_now][1]) <= threshold:
                    # print(time.strftime('%Y-%m-%d', time.localtime(f_pairs[i][1])))
                    f_oneday.setdefault(time.strftime('%Y-%m-%d 00:00:00', time.localtime(f_pairs[i][1])), []).append(
                        f_pairs[index_now][0])
                    jump += 1
                else:
                    break
            i = i + jump
    return f_oneday


def gen_text_time_pair(f_text, f_time, t_text, t_time):
    f_pairs = list(zip(f_text, f_time))
    t_pairs = list(zip(t_text, t_time))

    f_pairs_set = set()
    t_pairs_set = set()
    for f in f_pairs:
        for t in t_pairs:
            if math.fabs(f[1] - t[1]) <= 7 * 24 * 3600.0 and f[0] != "" and t[0] != "":
                f_pairs_set.add(f)
                t_pairs_set.add(t)
    f_pairs = list(f_pairs_set)
    t_pairs = list(t_pairs_set)
    f_pairs = sorted(f_pairs, key=lambda x: x[1], reverse=True)
    t_pairs = sorted(t_pairs, key=lambda x: x[1], reverse=True)

    # [print(x) for x in [[x[0] for x in f_pairs],[x[0] for x in t_pairs],[x[1] for x in f_pairs],[x[1] for x in t_pairs]]]
    return [x[0] for x in f_pairs],[x[0] for x in t_pairs],[x[1] for x in f_pairs],[x[1] for x in t_pairs]


def gen_entity_time_pair(f_entity, f_time, t_entity, t_time):
    f_pairs = list(zip(f_entity, f_time))
    t_pairs = list(zip(t_entity, t_time))
    # print(f_pairs)
    # print(t_pairs)
    if len(f_pairs) == 0:
        f_pairs.append(("NONE", 0.0))
    if len(t_pairs) == 0:
        t_pairs.append(("NONE", 0.0))

    f_pairs_set = set()
    t_pairs_set = set()
    for f in f_pairs:
        for t in t_pairs:
            if math.fabs(f[1] - t[1]) <= 30 * 24 * 3600.0 and f[0] != "" and t[0] != "":
                f_pairs_set.add(f)
                t_pairs_set.add(t)
    f_pairs = list(f_pairs_set)
    t_pairs = list(t_pairs_set)
    f_pairs = sorted(f_pairs, key=lambda x: x[1], reverse=True)
    t_pairs = sorted(t_pairs, key=lambda x: x[1], reverse=True)
    f_entity_new, t_entity_new, f_time_new, t_time_new = [x[0] for x in f_pairs], [x[0] for x in t_pairs], [x[1] for x in f_pairs], [x[1] for x in t_pairs]
    if len(f_time_new) == 0:
        f_time_new.append(0.0)
        f_entity_new.append("NONE")
    if len(t_time_new) == 0:
        t_time_new.append(0.0)
        t_entity_new.append("NONE")
    [print(x) for x in [f_entity_new, t_entity_new, f_time_new, t_time_new]]

    return f_entity_new, t_entity_new, f_time_new, t_time_new


class dataset(Dataset):
    def __init__(self, idpair_file, label_file, f_trace, t_trace, f_entity, t_entity, threshold=7.0):
        self.datalist = []
        self.idpairs = pkl.load(open(idpair_file, "rb"))
        self.labels = pkl.load(open(label_file, "rb"))
        self.f_entity = pkl.load(open(f_entity, "rb"))
        self.t_entity = pkl.load(open(t_entity, "rb"))
        # for idx, id_all in enumerate(self.idpairs):
        #     for f_id in remove_list_f:
        #         if f_id == id_all[0] and id_all in self.idpairs:
        #             self.idpairs.remove(id_all)
        #             del self.labels[idx]
        #     for t_id in remove_list_t:
        #         if t_id == id_all[1] and id_all in self.idpairs:
        #             self.idpairs.remove(id_all)
        #             del self.labels[idx]
        pairs_filter = []
        label_filter = []
        for idx, id_all in enumerate(self.idpairs):
            # if id_all[0] in remove_list_f or id_all[1] in remove_list_t:
            #     pass
            if id_all in remove_pairs:
                pass
            else:
                pairs_filter.append(id_all)
                label_filter.append(self.labels[idx])
        self.idpairs = pairs_filter
        self.labels = label_filter
        self.f_trace = pkl.load(open(f_trace, "rb"), encoding="bytes")
        self.t_trace = pkl.load(open(t_trace, "rb"), encoding="bytes")
        self.idpairs_filtered = []
        self.threshold = threshold * 24 * 3600.0
        # len_list = []
        for idx, pair in enumerate(self.idpairs):
            f_user_text = [x[2] for x in self.f_trace[pair[0]]]
            f_text_time = [datetime2stamp(x[1]) for x in self.f_trace[pair[0]]]
            t_user_text = [x[0] for x in self.t_trace[pair[1]]]
            t_text_time = [datetime2stamp(x[1]) for x in self.t_trace[pair[1]]]

            f_entity_info = self.f_entity[pair[0]] if pair[0] in self.f_entity else []
            t_entity_info = self.t_entity[pair[1]] if pair[1] in self.t_entity else []

            f_entity_nl = [x[0] for x in f_entity_info]
            f_entity_time = [datetime2stamp(x[1]) for x in f_entity_info]
            t_entity_nl = [x[0] for x in t_entity_info]
            t_entity_time = [datetime2stamp(x[1]) for x in t_entity_info]

            label = self.labels[idx]
            f_user_text, t_user_text, f_time, t_time = gen_text_time_pair(f_user_text, f_text_time, t_user_text,
                                                                          t_text_time, )
            f_user_entity, t_user_entity, f_entity_time, t_entity_time = gen_entity_time_pair(f_entity_nl,
                                                                                              f_entity_time,
                                                                                              t_entity_nl,
                                                                                              t_entity_time)
            text_len = len(f_user_text)
            # print(pair, label)
            # len_list.append(text_len)
            if text_len != 0:
                self.datalist.append((pair, f_user_text, t_user_text, f_time, t_time, f_user_entity, t_user_entity,
                                      f_entity_time, t_entity_time, label))
                self.idpairs_filtered.append(pair)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        return self.idpairs_filtered[idx], self.datalist[idx]


def collate_fn(batch):
    pair, f_user_text, t_user_text, f_time, t_time, \
    f_user_entity, t_user_entity, f_entity_time, t_entity_time, \
    labels = [], [], [], [], [], [], [], [], [], []
    # text:[bs,user, doc]
    for idpair, x in batch:
        pair.append(x[0])
        f_user_text.append(x[1])
        t_user_text.append(x[2])
        f_time.append(x[3])
        t_time.append(x[4])
        f_user_entity.append(x[5])
        t_user_entity.append(x[6])
        f_entity_time.append(x[7])
        t_entity_time.append(x[8])
        labels.append(x[9])
    return pair, f_user_text, t_user_text, f_time, t_time, \
           f_user_entity, t_user_entity, f_entity_time, t_entity_time, \
           labels


# ps = PorterStemmer()
def _index(doc, word2idx, t):
    raw = doc
    doc = doc.replace("_", " ")
    # stem the word
    stoplist = stopwords.words('english') + list(string.punctuation)
    # stemmer = SnowballStemmer('english')
    doc = re.sub(r"_", " ", doc)
    # print(doc)
    doc = re.sub(r"@[\w]*", "", doc)
    # print(doc)
    doc = re.sub(r"&amp;|&nbsp;|&quot;", "", doc)
    # print(doc)
    doc = re.sub(r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", "", doc)
    doc = doc.strip().lower()
    doc = re.sub(r"[%s]+" % punctuation, " ", doc)
    doc_clean = clean_str(doc)
    filtered_doc = [word for word in word_tokenize(text=doc_clean) if
                    word not in stoplist and word in word_filtered_dict]
    document_encode = [word2idx[word] for word in filtered_doc if word in word2idx]
    # document_encode = np.array(document_encode)
    t_list = [t] * len(document_encode)
    # try:
    #     print(raw)
    #     print(doc_clean)
    #     [print(x) for x in [filtered_doc, document_encode, t_list, len(document_encode)]]
    #     print("***********************")
    # except:
    #     print("*****encoding error*****")
    return document_encode, filtered_doc, t_list, len(document_encode)


def index_word(pair, f_user_text, f_time, t_user_text, t_time, labels, word2idx, max_word_length_f, max_word_length_t):
    batch_size = len(f_user_text)
    labels = torch.LongTensor(labels)
    f_word_idx = torch.LongTensor(batch_size, max_word_length_f).fill_(0)
    t_word_idx = torch.LongTensor(batch_size, max_word_length_t).fill_(0)
    f_time_batch = torch.LongTensor(batch_size, max_word_length_f).fill_(0)
    t_time_batch = torch.LongTensor(batch_size, max_word_length_t).fill_(0)

    # print("==============Next_batch==============")
    text_len_f = []
    text_len_t = []
    word_len_day_f = []
    word_len_day_t = []
    print(pair)
    print(labels)
    for i in range(batch_size):
        doc_list_f = []
        doc_list_t = []
        time_f = []
        time_t = []
        word_len_day_f_user = []
        word_len_day_t_user = []
        word_length_f_all = 0
        # print("========")
        # print(pair[i],labels[i])
        for idx, doc_f in enumerate(f_user_text[i]):
            f_t = f_time[i][idx]
            doc_idx_f, filtered_doc_f, f_time_list, word_length_f = _index(doc_f, word2idx, f_t)
            word_length_f_all += word_length_f
            # print(word_length_f)
            if word_length_f_all < max_word_length_f:
                if word_length_f != 0:
                    # print(filtered_doc_f)
                    # print(len(doc_idx_f), len(filtered_doc_f), len(f_time_list), word_length_f)
                    doc_list_f.extend(doc_idx_f)
                    time_f.extend(f_time_list)
                    word_len_day_f_user.append(word_length_f)
            else:
                doc_list_f.extend(doc_idx_f[0:word_length_f + max_word_length_f - word_length_f_all])
                time_f.extend(f_time_list[0:word_length_f + max_word_length_f - word_length_f_all])
                word_len_day_f_user.append(word_length_f + max_word_length_f - word_length_f_all)
                break
        word_len_day_f.append(word_len_day_f_user)
        text_len_f.append(word_length_f_all if word_length_f_all < max_word_length_f else word_length_f_all)
        #
        # print(word_length_f_all, len(doc_list_f))
        # print(doc_list_f)
        # print(time_f)
        # print(word_len_day_f)
        # print(text_len_f)
        # print("------")
        word_length_t_all = 0
        for idx, doc_t in enumerate(t_user_text[i]):
            t_t = t_time[i][idx]
            doc_idx_t, filtered_doc_t, t_time_list, word_length_t = _index(doc_t, word2idx, t_t)
            word_length_t_all += word_length_t
            if word_length_t_all < max_word_length_t:
                if word_length_t != 0:
                    # print(filtered_doc_t)
                    # print(len(doc_idx_t), len(filtered_doc_t), len(t_time_list), word_length_t)
                    doc_list_t.extend(doc_idx_t)
                    time_t.extend(t_time_list)
                    word_len_day_t_user.append(word_length_t)
            else:
                doc_list_t.extend(doc_idx_t[0:word_length_t + max_word_length_t - word_length_t_all])
                time_t.extend(t_time_list[0:word_length_t + max_word_length_t - word_length_t_all])
                word_len_day_t_user.append(word_length_t + max_word_length_t - word_length_t_all)
                break
        word_len_day_t.append(word_len_day_t_user)
        text_len_t.append(word_length_t_all if word_length_t_all < max_word_length_t else max_word_length_t)
        # print(word_length_t_all, len(doc_list_t))
        # print(doc_list_t)
        # print(time_t)
        # print(word_len_day_t)
        # print(text_len_t)
        # print("length", len(word_len_day_f_user), len(word_len_day_t_user))
        f_word_idx[i, :text_len_f[i]] = torch.LongTensor(doc_list_f)[:text_len_f[i]]
        t_word_idx[i, :text_len_t[i]] = torch.LongTensor(doc_list_t)[:text_len_t[i]]
        f_time_batch[i, :text_len_f[i]] = torch.LongTensor(time_f)[:text_len_f[i]]
        t_time_batch[i, :text_len_t[i]] = torch.LongTensor(time_t)[:text_len_t[i]]

    return f_word_idx, f_time_batch, word_len_day_f, t_word_idx, t_time_batch, word_len_day_t, labels


def index_entity(f_entity, t_entity, f_time, t_time, entity2idx, max_entity_length_f, max_entity_length_t):
    # [print(x) for x in [f_entity, t_entity, f_time, t_time]]
    batch_size = len(f_entity)
    f_entity_idx = torch.LongTensor(batch_size, max_entity_length_f).fill_(0)
    t_entity_idx = torch.LongTensor(batch_size, max_entity_length_t).fill_(0)
    f_time_batch = torch.LongTensor(batch_size, max_entity_length_f).fill_(0)
    t_time_batch = torch.LongTensor(batch_size, max_entity_length_t).fill_(0)

    for i in range(batch_size):
        entity_list_f = []
        entity_list_t = []
        time_f = []
        time_t = []
        entity_len_f = 0
        entity_len_t = 0
        for idx, entity in enumerate(f_entity[i]):
            time_now = f_time[i][idx]
            entity_id = entity2idx[entity]+1 if entity in entity2idx else 0
            entity_list_f.append(entity_id)
            time_f.append(time_now)
            entity_len_f += 1
        entity_len_f = entity_len_f if entity_len_f < max_entity_length_f else max_entity_length_f
        for idx, entity in enumerate(t_entity[i]):
            time_now = t_time[i][idx]
            entity_id = entity2idx[entity]+1 if entity in entity2idx else 0
            entity_list_t.append(entity_id)
            time_t.append(time_now)
            entity_len_t += 1
        entity_len_t = entity_len_t if entity_len_t < max_entity_length_t else max_entity_length_t
        f_entity_idx[i, :entity_len_f] = torch.LongTensor(entity_list_f)[:entity_len_f]
        t_entity_idx[i, :entity_len_t] = torch.LongTensor(entity_list_t)[:entity_len_t]
        f_time_batch[i, :entity_len_f] = torch.LongTensor(time_f)[:entity_len_f]
        t_time_batch[i, :entity_len_t] = torch.LongTensor(time_t)[:entity_len_t]
    # [print(x.numpy().tolist()) for x in [f_entity_idx, f_time_batch, t_entity_idx, t_time_batch]]
    return f_entity_idx, f_time_batch, t_entity_idx, t_time_batch


def clean_str(string):
    """
    adapted from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # remove punctuation
    try:
        string = BeautifulSoup(string, "lxml").text
    except:
        return ""
    string = re.sub(r"[^A-Za-z0-9(),!?\"\`]", " ", string)
    string = re.sub(r"\"s", " \"s", string)
    string = re.sub(r"\"ve", " \"ve", string)
    string = re.sub(r"n\"t", " n\"t", string)
    string = re.sub(r"\"re", " \"re", string)
    string = re.sub(r"\"d", " \"d", string)
    string = re.sub(r"\"ll", " \"ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    s = string.strip().lower()
    return s


if __name__ == '__main__':
    # x_train = "/home/shenhuawei/gaohao/DCMH-ALP/data/foursquare_twitter/x_train"
    # a = pkl.load(open(x_train, "rb"))
    # for x in a:
    #     if x[0] != x[1]:
    #         print(x)

    x_stem = "Donald Trump is eating now. We will go to Florida. KFC is amazing. checking the list before we go out and bring the umbrella. Be a happy girl"
    # x_stem = " ".join(ps.stem(x) for x in word_tokenize(x_stem))
    # print(x_stem)
    # tag = pos_tag(word_tokenize(text=clean_str(x_stem)))
    # print(x_stem)
    #
    # print([x for x in tag if x[1] == "NN"])
    # if pos[0][1] == "NN":
    #     filtered_doc_new.append(x_stem)
    print(get_entity(x_stem))
