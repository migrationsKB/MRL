from __future__ import print_function

import argparse
import torch
import pickle
import numpy as np
import os
import json

import math
import random
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io

from itertools import chain

import data

from utils import nearest_neighbors, get_topic_coherence
from sklearn.manifold import TSNE

import pandas as pd

from sklearn import cluster
from sklearn import metrics
from scipy.spatial import distance
from sklearn.decomposition import PCA
from torch import nn, optim
from torch.nn import functional as F

from numpy import dot
from numpy.linalg import norm

parser = argparse.ArgumentParser(description='Get confidence from centroid with ETM....')
parser.add_argument('--lang_code', type=str, default="de", help="the language code for the ETM model")
parser.add_argument('--num_topics', type=int, default=50, help="the number of topics")
parser.add_argument('--model_path', type=str,
                    default="output/models/ETM/de/etm_tweets_K_50_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.005_Bsz_1000_RhoSize_300_trainEmbeddings_0_val_loss_2.496855906969676e+29_epoch_195",
                    help="the path of the pretrained ETM model")
parser.add_argument('--batch_size', type=int, default=1000, help="batch size")
parser.add_argument('--num_words', type=int, default=20, help="number of top words per topic")
args = parser.parse_args()

# define environment
data_dir = os.path.join("output/preprocessed/forTP", args.lang_code)
num_topics = args.num_topics
batch_size = args.batch_size

# define device, either cuda or cpu.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cosine_similarity(list_1, list_2):
    cos_sim = dot(list_1, list_2) / (norm(list_1) * norm(list_2))
    return cos_sim


def load_data_and_vocab(input_dir=data_dir):
    """
    Load the data from preprocessed data directory for inference of topics using trained ETM
    :param input_dir:
    :return:
    """
    # word ids.
    token_file = os.path.join(input_dir, 'data_tokens.mat')
    count_file = os.path.join(input_dir, 'data_counts.mat')
    tokens_ = scipy.io.loadmat(token_file)['tokens'].squeeze()
    counts_ = scipy.io.loadmat(count_file)['counts'].squeeze()
    print(f"tokens shape: {tokens_.shape}, counts shape: {counts_.shape}")

    with open(os.path.join(input_dir, "vocab.pkl"), "rb") as f:
        vocab_ = pickle.load(f)

    word2id_ = {word: idx for idx, word in enumerate(vocab_)}
    id2word_ = {idx: word for idx, word in enumerate(vocab_)}
    return tokens_, counts_, vocab_, word2id_, id2word_


def get_theta_weights_avg(model, tokens, counts, vocab, batch_size, device, ):
    model.eval()
    num_docs = len(tokens)
    vocab_size = len(vocab)
    theta_weights_list = []
    topic2words = dict()

    with torch.no_grad():
        beta = model.get_beta()
        for k in range(0, num_topics):  # topic_indices:
            gamma = beta[k]
            top_words = list(gamma.cpu().numpy().argsort()[-args.num_words:][::-1])
            topic_words = [vocab[a] for a in top_words]
            topic2words[k] = topic_words

        # get most used topics
        indices_ = torch.tensor(range(num_docs))
        indices = torch.split(indices_, batch_size)
        theta_avg = torch.zeros(1, num_topics).to(device)
        thetaWeightedAvg = torch.zeros(1, num_topics).to(device)

        print('theta weighted avg shape: ', thetaWeightedAvg.shape)
        cnt = 0
        for idx, ind in enumerate(indices):
            data_batch = data.get_batch(tokens, counts, ind, vocab_size, device)
            print('data_batch:', len(data_batch))
            sums = data_batch.sum(1).unsqueeze(1)
            cnt += sums.sum(0).squeeze().cpu().numpy()
            normalized_data_batch = data_batch / sums
            theta, _ = model.get_theta(normalized_data_batch)
            theta_avg += theta.sum(0).unsqueeze(0) / num_docs

            weighed_theta = sums * theta
            print(weighed_theta.shape)  # [1000,50]

            theta_weights_list.append(weighed_theta.cpu().detach().numpy())
            weighed_theta_sum = weighed_theta.sum(0).unsqueeze(0)

            print('*' * 40)
            thetaWeightedAvg += weighed_theta_sum
            if idx % 100 == 0 and idx > 0:
                print('batch: {}/{}'.format(idx, len(indices)))
        # get topic words
        theta_weights_avg = thetaWeightedAvg.squeeze().cpu().numpy() / cnt
        # the most used topics in descending order.
        theta_weights_avg_ordered = theta_weights_avg.argsort()[::-1]

    # topic embeddings for the whole data
    theta_weights = np.concatenate(theta_weights_list, axis=0)
    print('sorting theta weights ...')
    theta_weights_sorted = []
    for thetaWeight in theta_weights:
        theta_weights_sorted.append(thetaWeight.argsort()[::-1].tolist())

    return topic2words, theta_weights_avg_ordered, theta_weights, theta_weights_sorted


def save_to_results(df, tokens, topic2words, theta_weights_avg_ordered, theta_weights_sorted, id2word, output_dir):
    tokens_list = []
    for sent in tokens.tolist():
        # from numpy array to list.
        tokens_list.append(sent.tolist()[0])

    topics_ = []
    sentences_ = []
    for topic_nrs, inds in zip(theta_weights_sorted, tokens_list):
        top_nr = topic_nrs[0]
        topics_.append(top_nr)
        sent = [id2word[idx] for idx in inds]
        sentences_.append(sent)

    print(f"len {len(sentences_)} df len {len(df)}")
    df['sentence_etm'] = sentences_
    df['topic'] = topics_
    df.to_csv(os.path.join(output_dir, f"{args.lang_code}_etm.csv"), index=False)

    with open(os.path.join(output_dir, "topic_words.json"), 'w') as f:
        json.dump(topic2words, f)

    print(theta_weights_avg_ordered)
    with open(os.path.join(output_dir, "most_freq_topics.pkl"), "w") as f:
        pickle.dump(theta_weights_avg_ordered, f)


if __name__ == '__main__':
    results_dir = "output/results/ETM"
    output_dir_ = os.path.join(results_dir, args.lang_code)
    if not os.path.exists(output_dir_):
        os.mkdir(output_dir_)

    tokens, counts, vocab, word2id, id2word = load_data_and_vocab(data_dir)
    df_file = os.path.join(data_dir, "de_non_empty.csv")
    df = pd.read_csv(df_file)
    model = torch.load(args.model_path)
    model.to(device)
    topic2words, theta_weights_avg_ordered, theta_weights, theta_weights_sorted = get_theta_weights_avg(model,
                                                                                                        tokens,
                                                                                                        counts,
                                                                                                        vocab,
                                                                                                        batch_size,
                                                                                                        device)
    save_to_results(df, tokens, topic2words, theta_weights_avg_ordered, theta_weights_sorted, id2word, output_dir_)