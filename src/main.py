import numpy as np
import pandas as pd
from pyTsetlinMachine.tools import Binarizer
from tqdm import tqdm
from sklearn import datasets
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
from sklearn.linear_model import SGDClassifier
from sklearn.exceptions import NotFittedError
from sympy.logic.boolalg import to_dnf, to_cnf
import random
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.tree import DecisionTreeClassifier
from contextualbandits.online import LinUCB, LogisticUCB
from sklearn.preprocessing import KBinsDiscretizer
from copy import deepcopy
import re
import os
import argparse

from data_loader import DataLoader
from contextual_bandit_algorithms import LinearUCB, TreeBootstrap, ThompsonSamplingTM, EpsilonGreedyTM, \
    EpsilonGreedyLogisticRegression, EpsilonGreedyNeuralNetwork

'''
----------------------------------------------------------------------------IRIS EXPERIMENT------------------------------------------------------------------------------------------------------------------------------------

'''

'''
Function to run contextual bandits in iris dataset
'''


def default_iris_experiment():
    dl = DataLoader()

    X_processed, X_binarized, y, y_encoded, reward_mat = dl.load_iris_data()

    # Number of independent runs
    num_exp_runs = 10

    # number of rounds for each run
    rounds = 1000

    # Parameters for TM
    clauses = 1200
    T = 1000
    s = 8.0
    number_of_state_bits = 10
    drop_clause_p = 0.0

    # Eploration probability of epsilon greedy
    exploration_probability = 0.1

    # Whether to select randomized context
    randomized_context = True

    # whether to perform interpretability
    interpretability = True

    # Whether to use CUDA to TM training
    CUDA = False

    linucb = LinearUCB(X_processed, y_encoded)
    treebootstrap = TreeBootstrap(X_processed, y_encoded)
    thompsamplingtm = ThompsonSamplingTM(X_processed, X_binarized, y, y_encoded, clauses, T, s, number_of_state_bits,
                                         cuda=CUDA, drop_clause_p=drop_clause_p)
    epsilongreedytm = EpsilonGreedyTM(X_processed, X_binarized, y, y_encoded, clauses, T, s, number_of_state_bits,
                                      cuda=CUDA)
    epsilongreedylogisticregression = EpsilonGreedyLogisticRegression(X_processed, y_encoded)
    epsilongreedyneuralnetwork = EpsilonGreedyNeuralNetwork(X_processed, y_encoded)

    total_rewards_linucb = []
    total_rewards_dt_ts = []
    total_rewards_tm_ts = []
    total_rewards_tm = []
    total_rewards_lr = []
    total_rewards_nn = []
    for run in range(num_exp_runs):

        print("Running Experiment " + str(run + 1))

        # rewards_logucb = train_logistic_ucb(rounds, X_processed, y_encoded, randomized_context=False, reward_mat = None)
        rewards_linucb = linucb.train_linear_ucb(rounds, randomized_context=randomized_context, reward_mat=reward_mat)

        rewards_dt_ts = treebootstrap.train_treebootstrap(rounds, randomized_context=randomized_context,
                                                          reward_mat=reward_mat)

        rewards_tm_ts, L1_tm_ts, Dnf_tm_ts = thompsamplingtm.train_tm_thompson_sampling(rounds,
                                                                                        randomized_context=randomized_context,
                                                                                        interpretability=interpretability,
                                                                                        reward_mat=reward_mat)

        rewards_tm, L1_tm, Dnf_tm = epsilongreedytm.train_tm_epsilon_greedy(rounds,
                                                                            exploration_probability=exploration_probability,
                                                                            randomized_context=randomized_context,
                                                                            reward_mat=reward_mat,
                                                                            interpretabiliy=interpretability)

        rewards_lr = epsilongreedylogisticregression.train_lr_epsilon_greedy(rounds,
                                                                             exploration_probability=exploration_probability,
                                                                             randomized_context=randomized_context,
                                                                             reward_mat=reward_mat)

        rewards_nn = epsilongreedyneuralnetwork.train_nn_epsilon_greedy(rounds,
                                                                        exploration_probability=exploration_probability,
                                                                        randomized_context=randomized_context,
                                                                        reward_mat=reward_mat)

        total_rewards_linucb.append(rewards_linucb)
        total_rewards_dt_ts.append(rewards_dt_ts)
        total_rewards_tm_ts.append(rewards_tm_ts)
        total_rewards_tm.append(rewards_tm)
        total_rewards_lr.append(rewards_lr)
        total_rewards_nn.append(rewards_nn)

        rewards_linucb = np.mean(total_rewards_linucb, axis=0)
        rewards_dt_ts = np.mean(total_rewards_dt_ts, axis=0)
        rewards_tm_ts = np.mean(total_rewards_tm_ts, axis=0)
        rewards_tm = np.mean(total_rewards_tm, axis=0)
        rewards_lr = np.mean(total_rewards_lr, axis=0)
        rewards_nn = np.mean(total_rewards_nn, axis=0)

        # mean_rewards_logucb = np.zeros_like(rewards_logucb, dtype=float)
        mean_rewards_linucb = np.zeros_like(rewards_linucb, dtype=float)
        mean_rewards_dt_ts = np.zeros_like(rewards_dt_ts, dtype=float)
        mean_rewards_tm_ts = np.zeros_like(rewards_tm_ts, dtype=float)
        mean_rewards_tm = np.zeros_like(rewards_tm, dtype=float)
        mean_rewards_lr = np.zeros_like(rewards_lr, dtype=float)
        mean_rewards_nn = np.zeros_like(rewards_nn, dtype=float)
        for index in range(rewards_tm.shape[0]):
            mean_rewards_linucb[index] = np.mean(rewards_linucb[:index + 1])
            mean_rewards_dt_ts[index] = np.mean(rewards_dt_ts[:index + 1])
            mean_rewards_tm_ts[index] = np.mean(rewards_tm_ts[:index + 1])
            mean_rewards_tm[index] = np.mean(rewards_tm[:index + 1])
            mean_rewards_lr[index] = np.mean(rewards_lr[:index + 1])
            mean_rewards_nn[index] = np.mean(rewards_nn[:index + 1])

        path = "Results/"
        if not os.path.exists(path):
            os.makedirs(path)

        exp_name = "IRIS"
        results_path = "Results/" + exp_name

        if not os.path.exists(results_path):
            os.makedirs(results_path)

        if L1_tm_ts != None and Dnf_tm_ts != None:

            f = open(results_path + '/' + 'interpretability_tm_ts.txt', 'w')
            num_classifiers = y_encoded.shape[1]
            for i in range(num_classifiers):
                f.write("-------ARM:" + str(i + 1) + "-------\n")
                f.write("Boolean Exp:  ")
                f.write(str(L1_tm_ts[i]))
                f.write("\n\n\n")
                f.write("DNF Exp:   ")
                f.write(str(Dnf_tm_ts[i]))
                f.write("\n\n\n\n\n")
            f.close()

        if L1_tm != None and Dnf_tm != None:

            f = open(results_path + '/' + 'interpretability_tm.txt', 'w')
            num_classifiers = y_encoded.shape[1]
            for i in range(num_classifiers):
                f.write("-------ARM:" + str(i + 1) + "-------\n")
                f.write("Boolean Exp:  ")
                f.write(str(L1_tm_ts[i]))
                f.write("\n\n\n")
                f.write("DNF Exp:   ")
                f.write(str(Dnf_tm_ts[i]))
                f.write("\n\n\n\n\n")
            f.close()

        np.save(results_path + '/' + 'mean_rewards_linucb.npy', mean_rewards_linucb)
        np.save(results_path + '/' + 'mean_rewards_dt_ts.npy', mean_rewards_dt_ts)
        np.save(results_path + '/' + 'mean_rewards_tm_ts.npy', mean_rewards_tm_ts)
        np.save(results_path + '/' + 'mean_rewards_tm.npy', mean_rewards_tm)
        np.save(results_path + '/' + 'mean_rewards_lr.npy', mean_rewards_lr)
        np.save(results_path + '/' + 'mean_rewards_nn.npy', mean_rewards_nn)

        plt.plot(mean_rewards_linucb, label='Linear UCB', color='cyan')
        plt.plot(mean_rewards_dt_ts, label='Tree Bootstrap', color='purple')
        plt.plot(mean_rewards_tm_ts, label='Tsetlin Machine (TS)', color='orange')
        plt.plot(mean_rewards_tm, label='Tsetlin Machine', color='red')
        plt.plot(mean_rewards_lr, label='Logistic Regression', color='green')
        plt.plot(mean_rewards_nn, label='Neural Network', color='blue')
        plt.xlabel("Steps")
        plt.title("Cumulative Mean Rewards on Iris dataset")
        plt.grid()
        plt.legend()
        plt.savefig(results_path + "/Performance_Comparison.pdf")
        plt.close()


'''
----------------------------------------------------------------------------BREAST CANCER EXPERIMENT---------------------------------------------------------------
'''

'''
Function to run contextual bandit experiment 
with breast cancer dataset.

'''


def default_breast_cancer_experiment():
    dl = DataLoader()

    X_processed, X_binarized, y, y_encoded, reward_mat = dl.load_breast_cancer_data()

    # Number of independent runs
    num_exp_runs = 10

    # Number of rounds for each run
    rounds = 1000

    # Parameters for TM
    clauses = 650
    T = 300
    s = 5.0
    number_of_state_bits = 10

    # Exploration probability of epsilon greedy algorithm
    exploration_probability = 0.1

    # Whether to use randomized context
    randomized_context = True

    # Whether to perform interpretability with TM
    interpretability = True
    # Whether to use CUDA for TM training
    CUDA = False

    linucb = LinearUCB(X_processed, y_encoded)
    treebootstrap = TreeBootstrap(X_processed, y_encoded)
    thompsamplingtm = ThompsonSamplingTM(X_processed, X_binarized, y, y_encoded, clauses, T, s, number_of_state_bits,
                                         cuda=CUDA)
    epsilongreedytm = EpsilonGreedyTM(X_processed, X_binarized, y, y_encoded, clauses, T, s, number_of_state_bits,
                                      cuda=CUDA)
    epsilongreedylogisticregression = EpsilonGreedyLogisticRegression(X_processed, y_encoded)
    epsilongreedyneuralnetwork = EpsilonGreedyNeuralNetwork(X_processed, y_encoded)

    total_rewards_linucb = []
    total_rewards_dt_ts = []
    total_rewards_tm_ts = []
    total_rewards_tm = []
    total_rewards_lr = []
    total_rewards_nn = []

    for run in range(num_exp_runs):

        # Running experiment with Tsetlin machines
        print("Running Experiment " + str(run + 1))

        # rewards_logucb=train_logistic_ucb(rounds, X_processed, y_encoded, randomized_context=False, reward_mat = None)

        rewards_linucb = linucb.train_linear_ucb(rounds, randomized_context=randomized_context, reward_mat=reward_mat)

        rewards_dt_ts = treebootstrap.train_treebootstrap(rounds, randomized_context=randomized_context,
                                                          reward_mat=reward_mat)

        rewards_tm_ts, L1_tm_ts, Dnf_tm_ts = thompsamplingtm.train_tm_thompson_sampling(rounds,
                                                                                        randomized_context=randomized_context,
                                                                                        interpretability=interpretability,
                                                                                        reward_mat=reward_mat)

        rewards_tm, L1_tm, Dnf_tm = epsilongreedytm.train_tm_epsilon_greedy(rounds,
                                                                            exploration_probability=exploration_probability,
                                                                            randomized_context=randomized_context,
                                                                            reward_mat=reward_mat,
                                                                            interpretabiliy=interpretability)

        rewards_lr = epsilongreedylogisticregression.train_lr_epsilon_greedy(rounds,
                                                                             exploration_probability=exploration_probability,
                                                                             randomized_context=randomized_context,
                                                                             reward_mat=reward_mat)

        rewards_nn = epsilongreedyneuralnetwork.train_nn_epsilon_greedy(rounds,
                                                                        exploration_probability=exploration_probability,
                                                                        randomized_context=randomized_context,
                                                                        reward_mat=reward_mat)

        # total_rewards_logucb[run]=rewards_linucb
        total_rewards_linucb.append(rewards_linucb)
        total_rewards_dt_ts.append(rewards_dt_ts)
        total_rewards_tm_ts.append(rewards_tm_ts)
        total_rewards_tm.append(rewards_tm)
        total_rewards_lr.append(rewards_lr)
        total_rewards_nn.append(rewards_nn)

        rewards_linucb = np.mean(total_rewards_linucb, axis=0)
        rewards_dt_ts = np.mean(total_rewards_dt_ts, axis=0)
        rewards_tm_ts = np.mean(total_rewards_tm_ts, axis=0)
        rewards_tm = np.mean(total_rewards_tm, axis=0)
        rewards_lr = np.mean(total_rewards_lr, axis=0)
        rewards_nn = np.mean(total_rewards_nn, axis=0)

        mean_rewards_linucb = np.zeros_like(rewards_linucb, dtype=float)
        mean_rewards_dt_ts = np.zeros_like(rewards_dt_ts, dtype=float)
        mean_rewards_tm_ts = np.zeros_like(rewards_tm_ts, dtype=float)
        mean_rewards_tm = np.zeros_like(rewards_tm, dtype=float)
        mean_rewards_lr = np.zeros_like(rewards_lr, dtype=float)
        mean_rewards_nn = np.zeros_like(rewards_nn, dtype=float)
        for index in range(rewards_tm.shape[0]):
            mean_rewards_linucb[index] = np.mean(rewards_linucb[:index + 1])
            mean_rewards_dt_ts[index] = np.mean(rewards_dt_ts[:index + 1])
            mean_rewards_tm_ts[index] = np.mean(rewards_tm_ts[:index + 1])
            mean_rewards_tm[index] = np.mean(rewards_tm[:index + 1])
            mean_rewards_lr[index] = np.mean(rewards_lr[:index + 1])
            mean_rewards_nn[index] = np.mean(rewards_nn[:index + 1])

        path = "Results/"
        if not os.path.exists(path):
            os.makedirs(path)

        exp_name = "BREAST CANCER"
        results_path = "Results/" + exp_name

        if not os.path.exists(results_path):
            os.makedirs(results_path)

        if L1_tm_ts != None and Dnf_tm_ts != None:

            f = open(results_path + '/' + 'interpretability_tm_ts.txt', 'w')
            num_classifiers = y_encoded.shape[1]
            for i in range(num_classifiers):
                f.write("-------ARM:" + str(i + 1) + "-------\n")
                f.write("Boolean Exp:  ")
                f.write(str(L1_tm_ts[i]))
                f.write("\n\n\n")
                f.write("DNF Exp:   ")
                f.write(str(Dnf_tm_ts[i]))
                f.write("\n\n\n\n\n")
            f.close()

        if L1_tm != None and Dnf_tm != None:

            f = open(results_path + '/' + 'interpretability_tm.txt', 'w')
            num_classifiers = y_encoded.shape[1]
            for i in range(num_classifiers):
                f.write("-------ARM:" + str(i + 1) + "-------\n")
                f.write("Boolean Exp:  ")
                f.write(str(L1_tm_ts[i]))
                f.write("\n\n\n")
                f.write("DNF Exp:   ")
                f.write(str(Dnf_tm_ts[i]))
                f.write("\n\n\n\n\n")
            f.close()

        np.save(results_path + '/' + 'mean_rewards_linucb.npy', mean_rewards_linucb)
        np.save(results_path + '/' + 'mean_rewards_dt_ts.npy', mean_rewards_dt_ts)
        np.save(results_path + '/' + 'mean_rewards_tm_ts.npy', mean_rewards_tm_ts)
        np.save(results_path + '/' + 'mean_rewards_tm.npy', mean_rewards_tm)
        np.save(results_path + '/' + 'mean_rewards_lr.npy', mean_rewards_lr)
        np.save(results_path + '/' + 'mean_rewards_nn.npy', mean_rewards_nn)

        # plt.plot(mean_rewards_dt_ts,label='Logistic UCB',color='grey')
        plt.plot(mean_rewards_linucb, label='Linear UCB', color='cyan')
        plt.plot(mean_rewards_dt_ts, label='Tree Bootstrap', color='purple')
        plt.plot(mean_rewards_tm_ts, label='Tsetlin Machine(TS)', color='orange')
        plt.plot(mean_rewards_tm, label='Tsetlin Machine', color='red')
        plt.plot(mean_rewards_lr, label='Logistic Regression', color='green')
        plt.plot(mean_rewards_nn, label='Neural Network', color='blue')
        plt.xlabel("Steps")
        plt.title("Cumulative Mean Rewards on " + exp_name + " dataset")
        plt.grid()
        plt.legend()
        plt.savefig(results_path + "/Performance_Comparison.pdf")
        plt.close()


'''
---------------------------------------------------------------------------NOISY XOR EXPERIMENT-------------------------------------------------------------------
'''

'''
Function to run contextual bandit experiment 
with noisy xor data 
'''


def default_noisy_xor_experiment():
    dl = DataLoader()

    X_processed, X_binarized, y, y_encoded, reward_mat = dl.load_noisy_xor_data()

    # number of independent experiment runs
    num_exp_runs = 10

    # number of rounds for each run
    rounds = 1000

    # Parameters for TM
    clauses = 1000
    T = 700
    s = 5.0
    number_of_state_bits = 8

    # Exploration probability of epsilon greedy algorithm
    exploration_probability = 0.1

    # whether to use interpretability
    interpretability = True

    # whether to use randomized context
    randomized_context = True

    # whether to use CUDA for training
    CUDA = False

    linucb = LinearUCB(X_processed, y_encoded)
    treebootstrap = TreeBootstrap(X_processed, y_encoded)
    thompsamplingtm = ThompsonSamplingTM(X_processed, X_binarized, y, y_encoded, clauses, T, s, number_of_state_bits,
                                         cuda=CUDA)
    epsilongreedytm = EpsilonGreedyTM(X_processed, X_binarized, y, y_encoded, clauses, T, s, number_of_state_bits,
                                      cuda=CUDA)
    epsilongreedylogisticregression = EpsilonGreedyLogisticRegression(X_processed, y_encoded)
    epsilongreedyneuralnetwork = EpsilonGreedyNeuralNetwork(X_processed, y_encoded)

    total_rewards_linucb = []
    total_rewards_dt_ts = []
    total_rewards_tm_ts = []
    total_rewards_tm = []
    total_rewards_lr = []
    total_rewards_nn = []
    for run in range(num_exp_runs):

        print("Running Experiment " + str(run + 1))

        rewards_linucb = linucb.train_linear_ucb(rounds, randomized_context=randomized_context, reward_mat=reward_mat)

        rewards_dt_ts = treebootstrap.train_treebootstrap(rounds, randomized_context=randomized_context,
                                                          reward_mat=reward_mat)

        rewards_tm_ts, L1_tm_ts, Dnf_tm_ts = thompsamplingtm.train_tm_thompson_sampling(rounds,
                                                                                        randomized_context=randomized_context,
                                                                                        interpretability=interpretability,
                                                                                        reward_mat=reward_mat)

        rewards_tm, L1_tm, Dnf_tm = epsilongreedytm.train_tm_epsilon_greedy(rounds,
                                                                            exploration_probability=exploration_probability,
                                                                            randomized_context=randomized_context,
                                                                            reward_mat=reward_mat,
                                                                            interpretabiliy=interpretability)

        rewards_lr = epsilongreedylogisticregression.train_lr_epsilon_greedy(rounds,
                                                                             exploration_probability=exploration_probability,
                                                                             randomized_context=randomized_context,
                                                                             reward_mat=reward_mat)

        rewards_nn = epsilongreedyneuralnetwork.train_nn_epsilon_greedy(rounds,
                                                                        exploration_probability=exploration_probability,
                                                                        randomized_context=randomized_context,
                                                                        reward_mat=reward_mat)

        total_rewards_linucb.append(rewards_linucb)
        total_rewards_dt_ts.append(rewards_dt_ts)
        total_rewards_tm_ts.append(rewards_tm_ts)
        total_rewards_tm.append(rewards_tm)
        total_rewards_lr.append(rewards_lr)
        total_rewards_nn.append(rewards_nn)

        rewards_linucb = np.mean(total_rewards_linucb, axis=0)
        rewards_dt_ts = np.mean(total_rewards_dt_ts, axis=0)
        rewards_tm_ts = np.mean(total_rewards_tm_ts, axis=0)
        rewards_tm = np.mean(total_rewards_tm, axis=0)
        rewards_lr = np.mean(total_rewards_lr, axis=0)
        rewards_nn = np.mean(total_rewards_nn, axis=0)

        mean_rewards_linucb = np.zeros_like(rewards_linucb, dtype=float)
        mean_rewards_dt_ts = np.zeros_like(rewards_dt_ts, dtype=float)
        mean_rewards_tm_ts = np.zeros_like(rewards_tm_ts, dtype=float)
        mean_rewards_tm = np.zeros_like(rewards_tm, dtype=float)
        mean_rewards_lr = np.zeros_like(rewards_lr, dtype=float)
        mean_rewards_nn = np.zeros_like(rewards_nn, dtype=float)
        for index in range(rewards_tm.shape[0]):
            mean_rewards_linucb[index] = np.mean(rewards_linucb[:index + 1])
            mean_rewards_dt_ts[index] = np.mean(rewards_dt_ts[:index + 1])
            mean_rewards_tm_ts[index] = np.mean(rewards_tm_ts[:index + 1])
            mean_rewards_tm[index] = np.mean(rewards_tm[:index + 1])
            mean_rewards_lr[index] = np.mean(rewards_lr[:index + 1])
            mean_rewards_nn[index] = np.mean(rewards_nn[:index + 1])

        path = "Results/"
        if not os.path.exists(path):
            os.makedirs(path)

        exp_name = "NOISY XOR"
        results_path = "Results/" + exp_name

        if not os.path.exists(results_path):
            os.makedirs(results_path)

        if L1_tm_ts != None and Dnf_tm_ts != None:

            f = open(results_path + '/' + 'interpretability_tm_ts.txt', 'w')
            num_classifiers = y_encoded.shape[1]
            for i in range(num_classifiers):
                f.write("-------ARM:" + str(i + 1) + "-------\n")
                f.write("Boolean Exp:  ")
                f.write(str(L1_tm_ts[i]))
                f.write("\n\n\n")
                f.write("DNF Exp:   ")
                f.write(str(Dnf_tm_ts[i]))
                f.write("\n\n\n\n\n")
            f.close()

        if L1_tm != None and Dnf_tm != None:

            f = open(results_path + '/' + 'interpretability_tm.txt', 'w')
            num_classifiers = y_encoded.shape[1]
            for i in range(num_classifiers):
                f.write("-------ARM:" + str(i + 1) + "-------\n")
                f.write("Boolean Exp:  ")
                f.write(str(L1_tm_ts[i]))
                f.write("\n\n\n")
                f.write("DNF Exp:   ")
                f.write(str(Dnf_tm_ts[i]))
                f.write("\n\n\n\n\n")
            f.close()

        np.save(results_path + '/' + 'mean_rewards_linucb', mean_rewards_linucb)
        np.save(results_path + '/' + 'mean_rewards_dt_ts.npy', mean_rewards_dt_ts)
        np.save(results_path + '/' + 'mean_rewards_tm_ts.npy', mean_rewards_tm_ts)
        np.save(results_path + '/' + 'mean_rewards_tm.npy', mean_rewards_tm)
        np.save(results_path + '/' + 'mean_rewards_lr.npy', mean_rewards_lr)
        np.save(results_path + '/' + 'mean_rewards_nn.npy', mean_rewards_nn)

        plt.plot(mean_rewards_linucb, label='Linear UCB', color='cyan')
        plt.plot(mean_rewards_dt_ts, label='Tree Bootstrap', color='purple')
        plt.plot(mean_rewards_tm_ts, label='Tsetlin Machine(TS)', color='orange')
        plt.plot(mean_rewards_tm, label='Tsetlin Machine', color='red')
        plt.plot(mean_rewards_lr, label='Logistic Regression', color='green')
        plt.plot(mean_rewards_nn, label='Neural Network', color='blue')
        plt.xlabel("Steps")
        plt.title("Cumulative Mean Rewards on " + exp_name + " dataset")
        plt.grid()
        plt.legend()
        plt.savefig(results_path + "/Performance_Comparison.pdf")
        plt.close()


'''
-----------------------------------------------------------------------MOVIELENS EXPERIMENT -----------------------------------------------------------------------
'''

'''
Function to run contextual bandit algorithm 
using movielens dataset
'''


def default_movielens_experiment():
    dl = DataLoader()

    X_processed, X_binarized, y, y_encoded, reward_mat = dl.load_movielens_data()

    # Number of independent runs
    num_exp_runs = 10
    # Number of rounds per run
    rounds = 1000

    # parameters for TM
    clauses = 4000
    T = 3000
    s = 8.0
    number_of_state_bits = 8
    drop_clause_p = 0.0

    # Exploration probability for epsilon greedy algorithms
    exploration_probability = 0.1

    # whether to randomize context
    randomized_context = True
    # whether to perform interpretability
    interpretability = True

    # whether to use CUDA for TM training
    CUDA = False

    linucb = LinearUCB(X_processed, y_encoded)
    treebootstrap = TreeBootstrap(X_processed, y_encoded)
    thompsamplingtm = ThompsonSamplingTM(X_processed, X_binarized, y, y_encoded, clauses, T, s, number_of_state_bits,
                                         cuda=CUDA, drop_clause_p=drop_clause_p)
    epsilongreedytm = EpsilonGreedyTM(X_processed, X_binarized, y, y_encoded, clauses, T, s, number_of_state_bits,
                                      cuda=CUDA, drop_clause_p=drop_clause_p)
    epsilongreedylogisticregression = EpsilonGreedyLogisticRegression(X_processed, y_encoded)
    epsilongreedyneuralnetwork = EpsilonGreedyNeuralNetwork(X_processed, y_encoded)

    total_rewards_linucb = []
    total_rewards_dt_ts = []
    total_rewards_tm_ts = []
    total_rewards_tm = []
    total_rewards_lr = []
    total_rewards_nn = []
    for run in range(num_exp_runs):

        print("Running Experiment " + str(run + 1))

        rewards_linucb = linucb.train_linear_ucb(rounds, randomized_context=randomized_context, reward_mat=reward_mat)

        rewards_dt_ts = treebootstrap.train_treebootstrap(rounds, randomized_context=randomized_context,
                                                          reward_mat=reward_mat)

        rewards_tm_ts, L1_tm_ts, Dnf_tm_ts = thompsamplingtm.train_tm_thompson_sampling(rounds,
                                                                                        randomized_context=randomized_context,
                                                                                        interpretability=interpretability,
                                                                                        reward_mat=reward_mat)

        rewards_tm, L1_tm, Dnf_tm = epsilongreedytm.train_tm_epsilon_greedy(rounds,
                                                                            exploration_probability=exploration_probability,
                                                                            randomized_context=randomized_context,
                                                                            reward_mat=reward_mat,
                                                                            interpretabiliy=interpretability)

        rewards_lr = epsilongreedylogisticregression.train_lr_epsilon_greedy(rounds,
                                                                             exploration_probability=exploration_probability,
                                                                             randomized_context=randomized_context,
                                                                             reward_mat=reward_mat)

        rewards_nn = epsilongreedyneuralnetwork.train_nn_epsilon_greedy(rounds,
                                                                        exploration_probability=exploration_probability,
                                                                        randomized_context=randomized_context,
                                                                        reward_mat=reward_mat)

        total_rewards_linucb.append(rewards_linucb)
        total_rewards_dt_ts.append(rewards_dt_ts)
        total_rewards_tm_ts.append(rewards_tm_ts)
        total_rewards_tm.append(rewards_tm)
        total_rewards_lr.append(rewards_lr)
        total_rewards_nn.append(rewards_nn)

        rewards_linucb = np.mean(total_rewards_linucb, axis=0)
        rewards_dt_ts = np.mean(total_rewards_dt_ts, axis=0)
        rewards_tm_ts = np.mean(total_rewards_tm_ts, axis=0)
        rewards_tm = np.mean(total_rewards_tm, axis=0)
        rewards_lr = np.mean(total_rewards_lr, axis=0)
        rewards_nn = np.mean(total_rewards_nn, axis=0)

        mean_rewards_linucb = np.zeros_like(rewards_linucb, dtype=float)
        mean_rewards_dt_ts = np.zeros_like(rewards_dt_ts, dtype=float)
        mean_rewards_tm_ts = np.zeros_like(rewards_tm_ts, dtype=float)
        mean_rewards_tm = np.zeros_like(rewards_tm, dtype=float)
        mean_rewards_lr = np.zeros_like(rewards_lr, dtype=float)
        mean_rewards_nn = np.zeros_like(rewards_nn, dtype=float)
        for index in range(rewards_tm.shape[0]):
            # mean_rewards_logucb[index] = np.mean(rewards_logucb[:index+1])
            mean_rewards_linucb[index] = np.mean(rewards_linucb[:index + 1])
            mean_rewards_dt_ts[index] = np.mean(rewards_dt_ts[:index + 1])
            mean_rewards_tm_ts[index] = np.mean(rewards_tm_ts[:index + 1])
            mean_rewards_tm[index] = np.mean(rewards_tm[:index + 1])
            mean_rewards_lr[index] = np.mean(rewards_lr[:index + 1])
            mean_rewards_nn[index] = np.mean(rewards_nn[:index + 1])

        path = "Results/"
        if not os.path.exists(path):
            os.makedirs(path)

        exp_name = "MOVIELENS"
        results_path = "Results/" + exp_name

        if not os.path.exists(results_path):
            os.makedirs(results_path)

        if L1_tm_ts != None and Dnf_tm_ts != None:

            f = open(results_path + '/' + 'interpretability_tm_ts.txt', 'w')
            num_classifiers = y_encoded.shape[1]
            for i in range(num_classifiers):
                f.write("-------ARM:" + str(i + 1) + "-------\n")
                f.write("Boolean Exp:  ")
                f.write(str(L1_tm_ts[i]))
                f.write("\n\n\n")
                f.write("DNF Exp:   ")
                f.write(str(Dnf_tm_ts[i]))
                f.write("\n\n\n\n\n")
            f.close()

        if L1_tm != None and Dnf_tm != None:

            f = open(results_path + '/' + 'interpretability_tm.txt', 'w')
            num_classifiers = y_encoded.shape[1]
            for i in range(num_classifiers):
                f.write("-------ARM:" + str(i + 1) + "-------\n")
                f.write("Boolean Exp:  ")
                f.write(str(L1_tm_ts[i]))
                f.write("\n\n\n")
                f.write("DNF Exp:   ")
                f.write(str(Dnf_tm_ts[i]))
                f.write("\n\n\n\n\n")
            f.close()

        np.save(results_path + '/' + 'mean_rewards_linucb.npy', mean_rewards_linucb)
        np.save(results_path + '/' + 'mean_rewards_dt_ts.npy', mean_rewards_dt_ts)
        np.save(results_path + '/' + 'mean_rewards_tm_ts.npy', mean_rewards_tm_ts)
        np.save(results_path + '/' + 'mean_rewards_tm.npy', mean_rewards_tm)
        np.save(results_path + '/' + 'mean_rewards_lr.npy', mean_rewards_lr)
        np.save(results_path + '/' + 'mean_rewards_nn.npy', mean_rewards_nn)

        plt.plot(mean_rewards_linucb, label='Linear UCB', color='cyan')
        plt.plot(mean_rewards_dt_ts, label='Tree Bootstrap', color='purple')
        plt.plot(mean_rewards_tm_ts, label='Tsetlin Machine(TS)', color='orange')
        plt.plot(mean_rewards_tm, label='Tsetlin Machine', color='red')
        plt.plot(mean_rewards_lr, label='Logistic Regression', color='green')
        plt.plot(mean_rewards_nn, label='Neural Network', color='blue')
        plt.xlabel("Steps")
        plt.title("Cumulative Mean Rewards on " + exp_name + " dataset")
        plt.grid()
        plt.legend()
        plt.savefig(results_path + "/Performance_Comparison.pdf")
        plt.close()


'''

-------------------------------------------------------------------SIMULATED ARTICLE EXPERIMENT--------------------------------------------------------------------
'''

'''
Function to run contextual bandit algorithm 
using simulated article dataset
'''


def default_simulated_article_experiment():
    dl = DataLoader()

    X_processed, X_binarized, y, y_encoded, reward_mat = dl.load_simulated_article_data()

    # Number of indepenent runs of the experiments
    num_exp_runs = 10

    # Number of rounds per run
    rounds = 2000

    # Parameters of TM
    clauses = 2000
    T = 1500
    s = 5.0
    number_of_state_bits = 10
    drop_clause_p = 0.3

    # Exploration probability of epsiolon greedy algorithm
    exploration_probability = 0.1

    # Whether to randomize context
    randomized_context = True

    # Whether to use interpretability
    interpretability = True

    # Whether to use CUDA for TM training
    CUDA = False

    linucb = LinearUCB(X_processed, y_encoded)
    treebootstrap = TreeBootstrap(X_processed, y_encoded)
    thompsamplingtm = ThompsonSamplingTM(X_processed, X_binarized, y, y_encoded, clauses, T, s, number_of_state_bits,
                                         cuda=CUDA, drop_clause_p=drop_clause_p)
    epsilongreedytm = EpsilonGreedyTM(X_processed, X_binarized, y, y_encoded, clauses, T, s, number_of_state_bits,
                                      cuda=CUDA, drop_clause_p=drop_clause_p)
    epsilongreedylogisticregression = EpsilonGreedyLogisticRegression(X_processed, y_encoded)
    epsilongreedyneuralnetwork = EpsilonGreedyNeuralNetwork(X_processed, y_encoded)

    # total_rewards_logucb = np.zeros((num_exp_runs,rounds))
    total_rewards_linucb = []
    total_rewards_dt_ts = []
    total_rewards_tm_ts = []
    total_rewards_tm = []
    total_rewards_lr = []
    total_rewards_nn = []
    for run in range(num_exp_runs):

        # Running experiment with Tsetlin machines
        print("Running Experiment " + str(run + 1))

        # rewards_logucb = train_logistic_ucb(rounds,X_processed,y_encoded,randomized_context=False,reward_mat = reward_mat)
        rewards_linucb = linucb.train_linear_ucb(rounds, randomized_context=randomized_context, reward_mat=reward_mat)

        rewards_dt_ts = treebootstrap.train_treebootstrap(rounds, randomized_context=randomized_context,
                                                          reward_mat=reward_mat)

        rewards_tm_ts, L1_tm_ts, Dnf_tm_ts = thompsamplingtm.train_tm_thompson_sampling(rounds,
                                                                                        randomized_context=randomized_context,
                                                                                        interpretability=interpretability,
                                                                                        reward_mat=reward_mat)

        rewards_tm, L1_tm, Dnf_tm = epsilongreedytm.train_tm_epsilon_greedy(rounds,
                                                                            exploration_probability=exploration_probability,
                                                                            randomized_context=randomized_context,
                                                                            reward_mat=reward_mat,
                                                                            interpretabiliy=interpretability)

        rewards_lr = epsilongreedylogisticregression.train_lr_epsilon_greedy(rounds,
                                                                             exploration_probability=exploration_probability,
                                                                             randomized_context=randomized_context,
                                                                             reward_mat=reward_mat)

        rewards_nn = epsilongreedyneuralnetwork.train_nn_epsilon_greedy(rounds,
                                                                        exploration_probability=exploration_probability,
                                                                        randomized_context=randomized_context,
                                                                        reward_mat=reward_mat)

        total_rewards_linucb.append(rewards_linucb)
        total_rewards_dt_ts.append(rewards_dt_ts)
        total_rewards_tm_ts.append(rewards_tm_ts)
        total_rewards_tm.append(rewards_tm)
        total_rewards_lr.append(rewards_lr)
        total_rewards_nn.append(rewards_nn)

        rewards_linucb = np.mean(total_rewards_linucb, axis=0)
        rewards_dt_ts = np.mean(total_rewards_dt_ts, axis=0)
        rewards_tm_ts = np.mean(total_rewards_tm_ts, axis=0)
        rewards_tm = np.mean(total_rewards_tm, axis=0)
        rewards_lr = np.mean(total_rewards_lr, axis=0)
        rewards_nn = np.mean(total_rewards_nn, axis=0)

        mean_rewards_linucb = np.zeros_like(rewards_linucb, dtype=float)
        mean_rewards_dt_ts = np.zeros_like(rewards_dt_ts, dtype=float)
        mean_rewards_tm_ts = np.zeros_like(rewards_tm_ts, dtype=float)
        mean_rewards_tm = np.zeros_like(rewards_tm, dtype=float)
        mean_rewards_lr = np.zeros_like(rewards_lr, dtype=float)
        mean_rewards_nn = np.zeros_like(rewards_nn, dtype=float)
        for index in range(rewards_tm.shape[0]):
            mean_rewards_linucb[index] = np.mean(rewards_linucb[:index + 1])
            mean_rewards_dt_ts[index] = np.mean(rewards_dt_ts[:index + 1])
            mean_rewards_tm_ts[index] = np.mean(rewards_tm_ts[:index + 1])
            mean_rewards_tm[index] = np.mean(rewards_tm[:index + 1])
            mean_rewards_lr[index] = np.mean(rewards_lr[:index + 1])
            mean_rewards_nn[index] = np.mean(rewards_nn[:index + 1])

        path = "Results/"
        if not os.path.exists(path):
            os.makedirs(path)

        exp_name = "SIMULATED ARTICLE"
        results_path = "Results/" + exp_name

        if not os.path.exists(results_path):
            os.makedirs(results_path)

        if L1_tm_ts != None and Dnf_tm_ts != None:

            f = open(results_path + '/' + 'interpretability_tm_ts.txt', 'w')
            num_classifiers = y_encoded.shape[1]
            for i in range(num_classifiers):
                f.write("-------ARM:" + str(i + 1) + "-------\n")
                f.write("Boolean Exp:  ")
                f.write(str(L1_tm_ts[i]))
                f.write("\n\n\n")
                f.write("DNF Exp:   ")
                f.write(str(Dnf_tm_ts[i]))
                f.write("\n\n\n\n\n")
            f.close()

        if L1_tm != None and Dnf_tm != None:

            f = open(results_path + '/' + 'interpretability_tm.txt', 'w')
            num_classifiers = y_encoded.shape[1]
            for i in range(num_classifiers):
                f.write("-------ARM:" + str(i + 1) + "-------\n")
                f.write("Boolean Exp:  ")
                f.write(str(L1_tm_ts[i]))
                f.write("\n\n\n")
                f.write("DNF Exp:   ")
                f.write(str(Dnf_tm_ts[i]))
                f.write("\n\n\n\n\n")
            f.close()

        np.save(results_path + '/' + 'mean_rewards_linucb.npy', mean_rewards_linucb)
        np.save(results_path + '/' + 'mean_rewards_dt_ts.npy', mean_rewards_dt_ts)
        np.save(results_path + '/' + 'mean_rewards_tm_ts.npy', mean_rewards_tm_ts)
        np.save(results_path + '/' + 'mean_rewards_tm.npy', mean_rewards_tm)
        np.save(results_path + '/' + 'mean_rewards_lr.npy', mean_rewards_lr)
        np.save(results_path + '/' + 'mean_rewards_nn.npy', mean_rewards_nn)

        plt.plot(mean_rewards_linucb, label='Linear UCB', color='cyan')
        plt.plot(mean_rewards_dt_ts, label='Tree Bootstrap', color='purple')
        plt.plot(mean_rewards_tm_ts, label='Tsetlin Machine(TS)', color='orange')
        plt.plot(mean_rewards_tm, label='Tsetlin Machine', color='red')
        plt.plot(mean_rewards_lr, label='Logistic Regression', color='green')
        plt.plot(mean_rewards_nn, label='Neural Network', color='blue')
        plt.xlabel("Steps")
        plt.title("Cumulative Mean Rewards on " + exp_name + " dataset")
        plt.grid()
        plt.legend()
        plt.savefig(results_path + "/Performance_Comparison.pdf")
        plt.close()


'''
-----------------------------------------------------------------------------ADULT DATA EXPERIMENT----------------------------------------------------------------
'''

'''
Function to run contextual bandit algorithm
in adut dataset
'''


def default_adult_experiment():
    dl = DataLoader()

    X_processed, X_binarized, y, y_encoded, reward_mat = dl.load_adult_dataset()

    # Number of independent runs
    num_exp_runs = 10

    # Number of rounds for each run
    rounds = 2000

    # TM parameters
    clauses = 1200
    T = 800
    s = 5.0
    drop_clause_p = 0.1
    number_of_state_bits = 8

    # Exploration probability for epsilon greedy algorithm
    exploration_probability = 0.1

    # whether to use randomized context
    randomized_context = True

    # Whether to use interpretability
    interpretability = True

    # Whether to use CUDA for TM training
    CUDA = False

    linucb = LinearUCB(X_processed, y_encoded)
    treebootstrap = TreeBootstrap(X_processed, y_encoded)
    thompsamplingtm = ThompsonSamplingTM(X_processed, X_binarized, y, y_encoded, clauses, T, s, number_of_state_bits,
                                         cuda=CUDA, drop_clause_p=drop_clause_p)
    epsilongreedytm = EpsilonGreedyTM(X_processed, X_binarized, y, y_encoded, clauses, T, s, number_of_state_bits,
                                      cuda=CUDA, drop_clause_p=drop_clause_p)
    epsilongreedylogisticregression = EpsilonGreedyLogisticRegression(X_processed, y_encoded)
    epsilongreedyneuralnetwork = EpsilonGreedyNeuralNetwork(X_processed, y_encoded)

    total_rewards_linucb = []
    total_rewards_dt_ts = []
    total_rewards_tm_ts = []
    total_rewards_tm = []
    total_rewards_lr = []
    total_rewards_nn = []
    for run in range(num_exp_runs):

        print("Running Experiment " + str(run + 1))

        rewards_linucb = linucb.train_linear_ucb(rounds, randomized_context=randomized_context, reward_mat=reward_mat)

        rewards_dt_ts = treebootstrap.train_treebootstrap(rounds, randomized_context=randomized_context,
                                                          reward_mat=reward_mat)

        rewards_tm_ts, L1_tm_ts, Dnf_tm_ts = thompsamplingtm.train_tm_thompson_sampling(rounds,
                                                                                        randomized_context=randomized_context,
                                                                                        interpretability=interpretability,
                                                                                        reward_mat=reward_mat)

        rewards_tm, L1_tm, Dnf_tm = epsilongreedytm.train_tm_epsilon_greedy(rounds,
                                                                            exploration_probability=exploration_probability,
                                                                            randomized_context=randomized_context,
                                                                            reward_mat=reward_mat,
                                                                            interpretabiliy=interpretability)

        rewards_lr = epsilongreedylogisticregression.train_lr_epsilon_greedy(rounds,
                                                                             exploration_probability=exploration_probability,
                                                                             randomized_context=randomized_context,
                                                                             reward_mat=reward_mat)

        rewards_nn = epsilongreedyneuralnetwork.train_nn_epsilon_greedy(rounds,
                                                                        exploration_probability=exploration_probability,
                                                                        randomized_context=randomized_context,
                                                                        reward_mat=reward_mat)

        total_rewards_linucb.append(rewards_linucb)
        total_rewards_dt_ts.append(rewards_dt_ts)
        total_rewards_tm_ts.append(rewards_tm_ts)
        total_rewards_tm.append(rewards_tm)
        total_rewards_lr.append(rewards_lr)
        total_rewards_nn.append(rewards_nn)

        rewards_linucb = np.mean(total_rewards_linucb, axis=0)
        rewards_dt_ts = np.mean(total_rewards_dt_ts, axis=0)
        rewards_tm_ts = np.mean(total_rewards_tm_ts, axis=0)
        rewards_tm = np.mean(total_rewards_tm, axis=0)
        rewards_lr = np.mean(total_rewards_lr, axis=0)
        rewards_nn = np.mean(total_rewards_nn, axis=0)

        mean_rewards_linucb = np.zeros_like(rewards_linucb, dtype=float)
        mean_rewards_dt_ts = np.zeros_like(rewards_dt_ts, dtype=float)
        mean_rewards_tm_ts = np.zeros_like(rewards_tm_ts, dtype=float)
        mean_rewards_tm = np.zeros_like(rewards_tm, dtype=float)
        mean_rewards_lr = np.zeros_like(rewards_lr, dtype=float)
        mean_rewards_nn = np.zeros_like(rewards_nn, dtype=float)
        for index in range(rewards_tm.shape[0]):
            mean_rewards_linucb[index] = np.mean(rewards_linucb[:index + 1])
            mean_rewards_dt_ts[index] = np.mean(rewards_dt_ts[:index + 1])
            mean_rewards_tm_ts[index] = np.mean(rewards_tm_ts[:index + 1])
            mean_rewards_tm[index] = np.mean(rewards_tm[:index + 1])
            mean_rewards_lr[index] = np.mean(rewards_lr[:index + 1])
            mean_rewards_nn[index] = np.mean(rewards_nn[:index + 1])

        path = "Results/"
        if not os.path.exists(path):
            os.makedirs(path)

        exp_name = "ADULT"
        results_path = "Results/" + exp_name

        if not os.path.exists(results_path):
            os.makedirs(results_path)

        if L1_tm_ts != None and Dnf_tm_ts != None:

            f = open(results_path + '/' + 'interpretability_tm_ts.txt', 'w')
            num_classifiers = y_encoded.shape[1]
            for i in range(num_classifiers):
                f.write("-------ARM:" + str(i + 1) + "-------\n")
                f.write("Boolean Exp:  ")
                f.write(str(L1_tm_ts[i]))
                f.write("\n\n\n")
                f.write("DNF Exp:   ")
                f.write(str(Dnf_tm_ts[i]))
                f.write("\n\n\n\n\n")
            f.close()

        if L1_tm != None and Dnf_tm != None:

            f = open(results_path + '/' + 'interpretability_tm.txt', 'w')
            num_classifiers = y_encoded.shape[1]
            for i in range(num_classifiers):
                f.write("-------ARM:" + str(i + 1) + "-------\n")
                f.write("Boolean Exp:  ")
                f.write(str(L1_tm_ts[i]))
                f.write("\n\n\n")
                f.write("DNF Exp:   ")
                f.write(str(Dnf_tm_ts[i]))
                f.write("\n\n\n\n\n")
            f.close()

        np.save(results_path + '/' + 'mean_rewards_linucb.npy', mean_rewards_linucb)
        np.save(results_path + '/' + 'mean_rewards_dt_ts.npy', mean_rewards_dt_ts)
        np.save(results_path + '/' + 'mean_rewards_tm_ts.npy', mean_rewards_tm_ts)
        np.save(results_path + '/' + 'mean_rewards_tm.npy', mean_rewards_tm)
        np.save(results_path + '/' + 'mean_rewards_lr.npy', mean_rewards_lr)
        np.save(results_path + '/' + 'mean_rewards_nn.npy', mean_rewards_nn)

        plt.plot(mean_rewards_linucb, label='Linear UCB', color='cyan')
        plt.plot(mean_rewards_dt_ts, label='Tree Bootstrap', color='purple')
        plt.plot(mean_rewards_tm_ts, label='Tsetlin Machine(TS)', color='orange')
        plt.plot(mean_rewards_tm, label='Tsetlin Machine', color='red')
        plt.plot(mean_rewards_lr, label='Logistic Regression', color='green')
        plt.plot(mean_rewards_nn, label='Neural Network', color='blue')
        plt.xlabel("Steps")
        plt.title("Cumulative Mean Rewards on " + exp_name + " dataset")
        plt.grid()
        plt.legend()
        plt.savefig(results_path + "/Performance_Comparison.pdf")
        plt.close()


'''

---------------------------------------------------------STATLOG(SHUTTLE) EXPERIMENT----------------------------------------------------------------------

'''

'''
Function to run contextual bandit algorithm
using statlog shuttle dataset 
'''


def default_statlog_shuttle_experiment():
    dl = DataLoader()

    X_processed, X_binarized, y, y_encoded, reward_mat = dl.load_statlog_shuttle_dataset()

    # Number of independent runs
    num_exp_runs = 10

    # Number of rounds for each run
    rounds = 2000

    # TM parameters
    clauses = 1200
    T = 800
    s = 5.0
    number_of_state_bits = 8

    # Exploration probability of epsilon greedy algorithm
    exploration_probability = 0.1

    # Whether to use randomized context
    randomized_context = True

    # Whether to use interpretability analysis for TM
    interpretability = True

    # whether to use CUDA for TM Training
    CUDA = False

    linucb = LinearUCB(X_processed, y_encoded)
    treebootstrap = TreeBootstrap(X_processed, y_encoded)
    thompsamplingtm = ThompsonSamplingTM(X_processed, X_binarized, y, y_encoded, clauses, T, s, number_of_state_bits,
                                         cuda=CUDA)
    epsilongreedytm = EpsilonGreedyTM(X_processed, X_binarized, y, y_encoded, clauses, T, s, number_of_state_bits,
                                      cuda=CUDA)
    epsilongreedylogisticregression = EpsilonGreedyLogisticRegression(X_processed, y_encoded)
    epsilongreedyneuralnetwork = EpsilonGreedyNeuralNetwork(X_processed, y_encoded)

    total_rewards_linucb = []
    total_rewards_dt_ts = []
    total_rewards_tm_ts = []
    total_rewards_tm = []
    total_rewards_lr = []
    total_rewards_nn = []
    for run in range(num_exp_runs):

        print("Running Experiment " + str(run + 1))

        rewards_linucb = linucb.train_linear_ucb(rounds, randomized_context=randomized_context, reward_mat=reward_mat)

        rewards_dt_ts = treebootstrap.train_treebootstrap(rounds, randomized_context=randomized_context,
                                                          reward_mat=reward_mat)

        rewards_tm_ts, L1_tm_ts, Dnf_tm_ts = thompsamplingtm.train_tm_thompson_sampling(rounds,
                                                                                        randomized_context=randomized_context,
                                                                                        interpretability=interpretability,
                                                                                        reward_mat=reward_mat)

        rewards_tm, L1_tm, Dnf_tm = epsilongreedytm.train_tm_epsilon_greedy(rounds,
                                                                            exploration_probability=exploration_probability,
                                                                            randomized_context=randomized_context,
                                                                            reward_mat=reward_mat)

        rewards_lr = epsilongreedylogisticregression.train_lr_epsilon_greedy(rounds,
                                                                             exploration_probability=exploration_probability,
                                                                             randomized_context=randomized_context,
                                                                             reward_mat=reward_mat)

        rewards_nn = epsilongreedyneuralnetwork.train_nn_epsilon_greedy(rounds,
                                                                        exploration_probability=exploration_probability,
                                                                        randomized_context=randomized_context,
                                                                        reward_mat=reward_mat)

        total_rewards_linucb.append(rewards_linucb)
        total_rewards_dt_ts.append(rewards_dt_ts)
        total_rewards_tm_ts.append(rewards_tm_ts)
        total_rewards_tm.append(rewards_tm)
        total_rewards_lr.append(rewards_lr)
        total_rewards_nn.append(rewards_nn)

        rewards_linucb = np.mean(total_rewards_linucb, axis=0)
        rewards_dt_ts = np.mean(total_rewards_dt_ts, axis=0)
        rewards_tm_ts = np.mean(total_rewards_tm_ts, axis=0)
        rewards_tm = np.mean(total_rewards_tm, axis=0)
        rewards_lr = np.mean(total_rewards_lr, axis=0)
        rewards_nn = np.mean(total_rewards_nn, axis=0)

        mean_rewards_linucb = np.zeros_like(rewards_linucb, dtype=float)
        mean_rewards_dt_ts = np.zeros_like(rewards_dt_ts, dtype=float)
        mean_rewards_tm_ts = np.zeros_like(rewards_tm_ts, dtype=float)
        mean_rewards_tm = np.zeros_like(rewards_tm, dtype=float)
        mean_rewards_lr = np.zeros_like(rewards_lr, dtype=float)
        mean_rewards_nn = np.zeros_like(rewards_nn, dtype=float)
        for index in range(rewards_tm.shape[0]):
            mean_rewards_linucb[index] = np.mean(rewards_linucb[:index + 1])
            mean_rewards_dt_ts[index] = np.mean(rewards_dt_ts[:index + 1])
            mean_rewards_tm_ts[index] = np.mean(rewards_tm_ts[:index + 1])
            mean_rewards_tm[index] = np.mean(rewards_tm[:index + 1])
            mean_rewards_lr[index] = np.mean(rewards_lr[:index + 1])
            mean_rewards_nn[index] = np.mean(rewards_nn[:index + 1])

        path = "Results/"
        if not os.path.exists(path):
            os.makedirs(path)

        exp_name = "STATLOG(SHUTTLE)"
        results_path = "Results/" + exp_name

        if not os.path.exists(results_path):
            os.makedirs(results_path)

        if L1_tm_ts != None and Dnf_tm_ts != None:

            f = open(results_path + '/' + 'interpretability_tm_ts.txt', 'w')
            num_classifiers = y_encoded.shape[1]
            for i in range(num_classifiers):
                f.write("-------ARM:" + str(i + 1) + "-------\n")
                f.write("Boolean Exp:  ")
                f.write(str(L1_tm_ts[i]))
                f.write("\n\n\n")
                f.write("DNF Exp:   ")
                f.write(str(Dnf_tm_ts[i]))
                f.write("\n\n\n\n\n")
            f.close()

        if L1_tm != None and Dnf_tm != None:

            f = open(results_path + '/' + 'interpretability_tm.txt', 'w')
            num_classifiers = y_encoded.shape[1]
            for i in range(num_classifiers):
                f.write("-------ARM:" + str(i + 1) + "-------\n")
                f.write("Boolean Exp:  ")
                f.write(str(L1_tm_ts[i]))
                f.write("\n\n\n")
                f.write("DNF Exp:   ")
                f.write(str(Dnf_tm_ts[i]))
                f.write("\n\n\n\n\n")
            f.close()

        np.save(results_path + '/' + 'mean_rewards_linucb.npy', mean_rewards_linucb)
        np.save(results_path + '/' + 'mean_rewards_dt_ts.npy', mean_rewards_dt_ts)
        np.save(results_path + '/' + 'mean_rewards_tm_ts.npy', mean_rewards_tm_ts)
        np.save(results_path + '/' + 'mean_rewards_tm.npy', mean_rewards_tm)
        np.save(results_path + '/' + 'mean_rewards_lr.npy', mean_rewards_lr)
        np.save(results_path + '/' + 'mean_rewards_nn.npy', mean_rewards_nn)

        plt.plot(mean_rewards_linucb, label='Linear UCB', color='cyan')
        plt.plot(mean_rewards_dt_ts, label='Tree Bootstrap', color='purple')
        plt.plot(mean_rewards_tm_ts, label='Tsetlin Machine(TS)', color='orange')
        plt.plot(mean_rewards_tm, label='Tsetlin Machine', color='red')
        plt.plot(mean_rewards_lr, label='Logistic Regression', color='green')
        plt.plot(mean_rewards_nn, label='Neural Network', color='blue')
        plt.xlabel("Steps")
        plt.title("Cumulative Mean Rewards on " + exp_name + " dataset")
        plt.grid()
        plt.legend()
        plt.savefig(results_path + "/Performance_Comparison.pdf")
        plt.close()


'''

----------------------------------------------------------------MNIST FLAT Experiment--------------------------------------------------------------------------
'''

'''
Function to run contextual bandit algorithm 
using Mnist dataset 
'''


def default_mnist_experiment():
    dl = DataLoader()

    X_processed, X_binarized, y, y_encoded, reward_mat, X_processed_cnn = dl.load_mnist_flat_dataset()

    # Number of independent runs
    num_exp_runs = 10

    # Number of rounds for each run
    rounds = 1000

    # TM parameters
    clauses = 5000
    T = 4000
    s = 5.0
    number_of_state_bits = 8

    # exploration probability for the epsilon greedy algorithm
    exploration_probability = 0.1

    # whether to use randomized context
    randomized_context = True

    # whether to use interpretability
    interpretability = False

    # whether to use CUDA for TM
    CUDA = False

    # whether to use TM cnn learner
    cnn = False

    linucb = LinearUCB(X_processed, y_encoded)
    treebootstrap = TreeBootstrap(X_processed, y_encoded)

    thompsamplingtm = ThompsonSamplingTM(X_processed, X_binarized, y, y_encoded, clauses, T, s, number_of_state_bits,
                                         cuda=CUDA, cnn=cnn)
    epsilongreedytm = EpsilonGreedyTM(X_processed, X_binarized, y, y_encoded, clauses, T, s, number_of_state_bits,
                                      cuda=CUDA, cnn=cnn)

    epsilongreedylogisticregression = EpsilonGreedyLogisticRegression(X_processed, y_encoded)
    epsilongreedyneuralnetwork = EpsilonGreedyNeuralNetwork(X_processed, y_encoded)

    total_rewards_linucb = []
    total_rewards_dt_ts = []
    total_rewards_tm_ts = []
    total_rewards_tm = []
    total_rewards_lr = []
    total_rewards_nn = []
    for run in range(num_exp_runs):

        print("Running Experiment " + str(run + 1))

        rewards_linucb = linucb.train_linear_ucb(rounds, randomized_context=randomized_context, reward_mat=reward_mat)

        rewards_dt_ts = treebootstrap.train_treebootstrap(rounds, randomized_context=randomized_context,
                                                          reward_mat=reward_mat)

        rewards_tm_ts, L1_tm_ts, Dnf_tm_ts = thompsamplingtm.train_tm_thompson_sampling(rounds,
                                                                                        randomized_context=randomized_context,
                                                                                        interpretability=interpretability,
                                                                                        reward_mat=reward_mat)

        rewards_tm, L1_tm, Dnf_tm = epsilongreedytm.train_tm_epsilon_greedy(rounds,
                                                                            exploration_probability=exploration_probability,
                                                                            randomized_context=randomized_context,
                                                                            reward_mat=reward_mat,
                                                                            interpretabiliy=interpretability)

        rewards_lr = epsilongreedylogisticregression.train_lr_epsilon_greedy(rounds,
                                                                             exploration_probability=exploration_probability,
                                                                             randomized_context=randomized_context,
                                                                             reward_mat=reward_mat)

        rewards_nn = epsilongreedyneuralnetwork.train_nn_epsilon_greedy(rounds,
                                                                        exploration_probability=exploration_probability,
                                                                        randomized_context=randomized_context,
                                                                        reward_mat=reward_mat)

        total_rewards_linucb.append(rewards_linucb)
        total_rewards_dt_ts.append(rewards_dt_ts)
        total_rewards_tm_ts.append(rewards_tm_ts)
        total_rewards_tm.append(rewards_tm)
        total_rewards_lr.append(rewards_lr)
        total_rewards_nn.append(rewards_nn)

        rewards_linucb = np.mean(total_rewards_linucb, axis=0)
        rewards_dt_ts = np.mean(total_rewards_dt_ts, axis=0)
        rewards_tm_ts = np.mean(total_rewards_tm_ts, axis=0)
        rewards_tm = np.mean(total_rewards_tm, axis=0)
        rewards_lr = np.mean(total_rewards_lr, axis=0)
        rewards_nn = np.mean(total_rewards_nn, axis=0)

        mean_rewards_linucb = np.zeros_like(rewards_linucb, dtype=float)
        mean_rewards_dt_ts = np.zeros_like(rewards_dt_ts, dtype=float)
        mean_rewards_tm_ts = np.zeros_like(rewards_tm_ts, dtype=float)
        mean_rewards_tm = np.zeros_like(rewards_tm, dtype=float)
        mean_rewards_lr = np.zeros_like(rewards_lr, dtype=float)
        mean_rewards_nn = np.zeros_like(rewards_nn, dtype=float)
        for index in range(rewards_tm.shape[0]):
            mean_rewards_linucb[index] = np.mean(rewards_linucb[:index + 1])
            mean_rewards_dt_ts[index] = np.mean(rewards_dt_ts[:index + 1])
            mean_rewards_tm_ts[index] = np.mean(rewards_tm_ts[:index + 1])
            mean_rewards_tm[index] = np.mean(rewards_tm[:index + 1])
            mean_rewards_lr[index] = np.mean(rewards_lr[:index + 1])
            mean_rewards_nn[index] = np.mean(rewards_nn[:index + 1])

        path = "Results/"
        if not os.path.exists(path):
            os.makedirs(path)

        exp_name = "MNIST"
        results_path = "Results/" + exp_name

        if not os.path.exists(results_path):
            os.makedirs(results_path)

        if L1_tm_ts != None and Dnf_tm_ts != None:

            f = open(results_path + '/' + 'interpretability_tm_ts.txt', 'w')
            num_classifiers = y_encoded.shape[1]
            for i in range(num_classifiers):
                f.write("-------ARM:" + str(i + 1) + "-------\n")
                f.write("Boolean Exp:  ")
                f.write(str(L1_tm_ts[i]))
                f.write("\n\n\n")
                f.write("DNF Exp:   ")
                f.write(str(Dnf_tm_ts[i]))
                f.write("\n\n\n\n\n")
            f.close()

        if L1_tm != None and Dnf_tm != None:

            f = open(results_path + '/' + 'interpretability_tm.txt', 'w')
            num_classifiers = y_encoded.shape[1]
            for i in range(num_classifiers):
                f.write("-------ARM:" + str(i + 1) + "-------\n")
                f.write("Boolean Exp:  ")
                f.write(str(L1_tm_ts[i]))
                f.write("\n\n\n")
                f.write("DNF Exp:   ")
                f.write(str(Dnf_tm_ts[i]))
                f.write("\n\n\n\n\n")
            f.close()

        np.save(results_path + '/' + 'mean_rewards_linucb.npy', mean_rewards_linucb)
        np.save(results_path + '/' + 'mean_rewards_dt_ts.npy', mean_rewards_dt_ts)
        np.save(results_path + '/' + 'mean_rewards_tm_ts.npy', mean_rewards_tm_ts)
        np.save(results_path + '/' + 'mean_rewards_tm.npy', mean_rewards_tm)
        np.save(results_path + '/' + 'mean_rewards_lr.npy', mean_rewards_lr)
        np.save(results_path + '/' + 'mean_rewards_nn.npy', mean_rewards_nn)

        plt.plot(mean_rewards_linucb, label='Linear UCB', color='cyan')
        plt.plot(mean_rewards_dt_ts, label='Tree Bootstrap', color='purple')
        plt.plot(mean_rewards_tm_ts, label='Tsetlin Machine(TS)', color='orange')
        plt.plot(mean_rewards_tm, label='Tsetlin Machine', color='red')
        plt.plot(mean_rewards_lr, label='Logistic Regression', color='green')
        plt.plot(mean_rewards_nn, label='Neural Network', color='blue')
        plt.xlabel("Steps")
        plt.title("Cumulative Mean Rewards on " + exp_name + " dataset")
        plt.grid()
        plt.legend()
        plt.savefig(results_path + "/Performance_Comparison.pdf")
        plt.close()


'''

-------------------------------------------------------COVERTYPE DATASET---------------------------------------------------------------------------
'''

'''
Function to run contextual bandit algorithm using 
covertype dataset
'''


def default_covertype_experiment():
    dl = DataLoader()

    X_processed, X_binarized, y, y_encoded, reward_mat = dl.load_covertype_dataset()

    # Number of independent experiment runs
    num_exp_runs = 10

    # Number of rounds for each run
    rounds = 2000

    # TM parameters
    clauses = 1200
    T = 800
    s = 5.0
    drop_clause_p = 0.1
    number_of_state_bits = 8

    # Exploration probability for epsilon greedy algorithm
    exploration_probability = 0.1

    # whethet to use randomized context
    randomized_context = True
    # whether to use interpretability for TM
    interpretability = True

    # whethet to use CUDA for TM training
    CUDA = False

    linucb = LinearUCB(X_processed, y_encoded)
    treebootstrap = TreeBootstrap(X_processed, y_encoded)
    thompsamplingtm = ThompsonSamplingTM(X_processed, X_binarized, y, y_encoded, clauses, T, s, number_of_state_bits,
                                         cuda=CUDA, drop_clause_p=drop_clause_p)
    epsilongreedytm = EpsilonGreedyTM(X_processed, X_binarized, y, y_encoded, clauses, T, s, number_of_state_bits,
                                      cuda=CUDA, drop_clause_p=drop_clause_p)
    epsilongreedylogisticregression = EpsilonGreedyLogisticRegression(X_processed, y_encoded)
    epsilongreedyneuralnetwork = EpsilonGreedyNeuralNetwork(X_processed, y_encoded)

    total_rewards_linucb = []
    total_rewards_dt_ts = []
    total_rewards_tm_ts = []
    total_rewards_tm = []
    total_rewards_lr = []
    total_rewards_nn = []
    for run in range(num_exp_runs):

        print("Running Experiment " + str(run + 1))

        rewards_linucb = linucb.train_linear_ucb(rounds, randomized_context=randomized_context, reward_mat=reward_mat)

        rewards_dt_ts = treebootstrap.train_treebootstrap(rounds, randomized_context=randomized_context,
                                                          reward_mat=reward_mat)

        rewards_tm_ts, L1_tm_ts, Dnf_tm_ts = thompsamplingtm.train_tm_thompson_sampling(rounds,
                                                                                        randomized_context=randomized_context,
                                                                                        interpretability=interpretability,
                                                                                        reward_mat=reward_mat)

        rewards_tm, L1_tm, Dnf_tm = epsilongreedytm.train_tm_epsilon_greedy(rounds,
                                                                            exploration_probability=exploration_probability,
                                                                            randomized_context=randomized_context,
                                                                            reward_mat=reward_mat,
                                                                            interpretabiliy=interpretability)

        rewards_lr = epsilongreedylogisticregression.train_lr_epsilon_greedy(rounds,
                                                                             exploration_probability=exploration_probability,
                                                                             randomized_context=randomized_context,
                                                                             reward_mat=reward_mat)

        rewards_nn = epsilongreedyneuralnetwork.train_nn_epsilon_greedy(rounds,
                                                                        exploration_probability=exploration_probability,
                                                                        randomized_context=randomized_context,
                                                                        reward_mat=reward_mat)

        total_rewards_linucb.append(rewards_linucb)
        total_rewards_dt_ts.append(rewards_dt_ts)
        total_rewards_tm_ts.append(rewards_tm_ts)
        total_rewards_tm.append(rewards_tm)
        total_rewards_lr.append(rewards_lr)
        total_rewards_nn.append(rewards_nn)

        rewards_linucb = np.mean(total_rewards_linucb, axis=0)
        rewards_dt_ts = np.mean(total_rewards_dt_ts, axis=0)
        rewards_tm_ts = np.mean(total_rewards_tm_ts, axis=0)
        rewards_tm = np.mean(total_rewards_tm, axis=0)
        rewards_lr = np.mean(total_rewards_lr, axis=0)
        rewards_nn = np.mean(total_rewards_nn, axis=0)

        mean_rewards_linucb = np.zeros_like(rewards_linucb, dtype=float)
        mean_rewards_dt_ts = np.zeros_like(rewards_dt_ts, dtype=float)
        mean_rewards_tm_ts = np.zeros_like(rewards_tm_ts, dtype=float)
        mean_rewards_tm = np.zeros_like(rewards_tm, dtype=float)
        mean_rewards_lr = np.zeros_like(rewards_lr, dtype=float)
        mean_rewards_nn = np.zeros_like(rewards_nn, dtype=float)
        for index in range(rewards_tm.shape[0]):
            mean_rewards_linucb[index] = np.mean(rewards_linucb[:index + 1])
            mean_rewards_dt_ts[index] = np.mean(rewards_dt_ts[:index + 1])
            mean_rewards_tm_ts[index] = np.mean(rewards_tm_ts[:index + 1])
            mean_rewards_tm[index] = np.mean(rewards_tm[:index + 1])
            mean_rewards_lr[index] = np.mean(rewards_lr[:index + 1])
            mean_rewards_nn[index] = np.mean(rewards_nn[:index + 1])

        path = "Results/"
        if not os.path.exists(path):
            os.makedirs(path)

        exp_name = "COVERTYPE"
        results_path = "Results/" + exp_name

        if not os.path.exists(results_path):
            os.makedirs(results_path)

        if L1_tm_ts != None and Dnf_tm_ts != None:

            f = open(results_path + '/' + 'interpretability_tm_ts.txt', 'w')
            num_classifiers = y_encoded.shape[1]
            for i in range(num_classifiers):
                f.write("-------ARM:" + str(i + 1) + "-------\n")
                f.write("Boolean Exp:  ")
                f.write(str(L1_tm_ts[i]))
                f.write("\n\n\n")
                f.write("DNF Exp:   ")
                f.write(str(Dnf_tm_ts[i]))
                f.write("\n\n\n\n\n")
            f.close()

        if L1_tm != None and Dnf_tm != None:

            f = open(results_path + '/' + 'interpretability_tm.txt', 'w')
            num_classifiers = y_encoded.shape[1]
            for i in range(num_classifiers):
                f.write("-------ARM:" + str(i + 1) + "-------\n")
                f.write("Boolean Exp:  ")
                f.write(str(L1_tm_ts[i]))
                f.write("\n\n\n")
                f.write("DNF Exp:   ")
                f.write(str(Dnf_tm_ts[i]))
                f.write("\n\n\n\n\n")
            f.close()

        np.save(results_path + '/' + 'mean_rewards_linucb.npy', mean_rewards_linucb)
        np.save(results_path + '/' + 'mean_rewards_dt_ts.npy', mean_rewards_dt_ts)
        np.save(results_path + '/' + 'mean_rewards_tm_ts.npy', mean_rewards_tm_ts)
        np.save(results_path + '/' + 'mean_rewards_tm.npy', mean_rewards_tm)
        np.save(results_path + '/' + 'mean_rewards_lr.npy', mean_rewards_lr)
        np.save(results_path + '/' + 'mean_rewards_nn.npy', mean_rewards_nn)

        plt.plot(mean_rewards_linucb, label='Linear UCB', color='cyan')
        plt.plot(mean_rewards_dt_ts, label='Tree Bootstrap', color='purple')
        plt.plot(mean_rewards_tm_ts, label='Tsetlin Machine(TS)', color='orange')
        plt.plot(mean_rewards_tm, label='Tsetlin Machine', color='red')
        plt.plot(mean_rewards_lr, label='Logistic Regression', color='green')
        plt.plot(mean_rewards_nn, label='Neural Network', color='blue')
        plt.xlabel("Steps")
        plt.title("Cumulative Mean Rewards on " + exp_name + " dataset")
        plt.grid()
        plt.legend()
        plt.savefig(results_path + "/Performance_Comparison.pdf")
        plt.close()


'''
-----------------------------------------------------------------------RUNNING THE EXPERIMENTS------------------------------------------------------- 

'''


def run_default_experiments(name):
    name = name.lower()
    if name == 'iris':
        print("Running Default Iris Experiment")
        default_iris_experiment()
    elif name == 'breast_cancer':
        print("Running Default  Breast Cancer Experiment")
        default_breast_cancer_experiment()

    elif name == 'noisy_xor':
        print("Running  Default Noisy XOR Experiment")
        default_noisy_xor_experiment()

    elif name == 'simulated_article':
        print("Running Default Simulated Article Experiment")
        default_simulated_article_experiment()
    elif name == 'movielens':
        print('Running  Default Movielens Experiment')
        default_movielens_experiment()

    elif name == 'adult':
        print('Running Default Adult Experiment')
        default_adult_experiment()

    elif name == 'statlog_shuttle':
        print('Running  Default Statlog(Shuttle) Experiment')
        default_statlog_shuttle_experiment()
    elif name == 'mnist':
        print("Running Default Mnist Experiment")
        default_mnist_experiment()

    elif name == 'covertype':
        print("Running Default Covertype Experiment")
        default_covertype_experiment()
    else:
        raise ValueError("The name of the dataset is not defined")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments to use to launch the experiment')
    parser.add_argument('--dataset_name', help='Name of the experiment to perform', type=str, default='iris')
    parser.add_argument('--default_configuration',
                        help='Boolean value to determine whether default configuration to be used', default='False')
    parser.add_argument('--max_bits', help='maximum number of bits per feature to be used for binarization', default=10,type=int)
    parser.add_argument('--num_clauses', help='number of clauses to be used in the TM learner', default=None,type=int)
    parser.add_argument('--T', help='value of T parameter for the TM learner', default=None,type=int)
    parser.add_argument('--s', help='value of the s parameter for the TM learner', default=None,type=float)
    parser.add_argument('--num_state_bits', help='number of state bits for the TM learner', default=None,type=int)
    parser.add_argument('--interpretability', help='whether to perform interpretability for the TM learner',
                        default= 'False')
    parser.add_argument('--drop_clause_p', help='probability of dropping clause in TM learner', default=0.0,type=float)
    parser.add_argument('--num_runs', help = 'Number of independent experiment runs', default=5,type=int)
    parser.add_argument('--num_rounds', help='number of rounds for each run', default=2000, type=int)
    parser.add_argument('--exploration_probs', help='exploration probability for the epsilon greedy algorithm', type=float, default=0.1)
    parser.add_argument('--use_cuda', help='Whether to use cuda for the TM learner', default='False')
    parser.add_argument('--results_dir', help='directory name where the results will be saved', type=str, default='Results/')

    args = parser.parse_args()

    data_name = args.dataset_name.lower()

    default_val = args.default_configuration.lower()

    if default_val == 'true':

        run_default_experiments(data_name)
    elif default_val == 'false':

        max_bits = args.max_bits

        dl = DataLoader()
        print("Running experiments with ", data_name," dataset")

        if data_name == 'iris':
            X_processed, X_binarized, y, y_encoded, reward_mat = dl.load_iris_data(max_bits=max_bits)
        elif data_name =='breast_cancer':
            X_processed, X_binarized, y, y_encoded, reward_mat = dl.load_breast_cancer_data(max_bits=max_bits)
        elif data_name == 'noisy_xor':
            print("Loading  XOR data already included")
            X_processed, X_binarized, y, y_encoded, reward_mat = dl.load_noisy_xor_data(max_bits=10)
        elif data_name == 'simulated_article':
            X_processed, X_binarized, y, y_encoded, reward_mat = dl.load_simulated_article_data(max_bits=max_bits)
        elif data_name =='movielens':
            X_processed, X_binarized, y, y_encoded, reward_mat = dl.load_movielens_data(max_bits=max_bits)
        elif data_name =='adult':
            X_processed, X_binarized, y, y_encoded, reward_mat = dl.load_adult_dataset(max_bits=max_bits)
        elif data_name == 'statlog_shuttle':
            X_processed, X_binarized, y, y_encoded, reward_mat = dl.load_statlog_shuttle_dataset(max_bits=max_bits)
        elif data_name == 'mnist':
            X_processed, X_binarized, y, y_encoded, reward_mat, X_processed_cnn = dl.load_mnist_flat_dataset(max_bits=max_bits)
        elif data_name == 'covertype':
            X_processed, X_binarized, y, y_encoded, reward_mat = dl.load_covertype_dataset(max_bits=max_bits)

        else:
            raise ValueError("Invalid data name")


        # Number of independent experiment runs
        num_exp_runs = args.num_runs


        # Number of rounds for each run
        rounds = args.num_rounds

        # TM parameters
        if args.num_clauses is None:
            raise ValueError('number of clauses cannot be None')

        if args.T is None:
            raise ValueError('T cannot be None')

        if args.s is None:
            raise ValueError('s cannot be None')

        if args.num_state_bits  is None:
            raise ValueError('num_state_bits cannot be None')
        clauses = args.num_clauses
        T = args.T
        s = args.s
        drop_clause_p = args.drop_clause_p
        number_of_state_bits = args.num_state_bits

        # Exploration probability for epsilon greedy algorithm
        exploration_probability = args.exploration_probs

        # whether to use randomized context
        randomized_context = True
        # whether to use interpretability for TM
        if args.interpretability.lower() == 'true':
            interpretability = True
        elif args.interpretability.lower() == 'false':
            interpretability = False
        else:
            raise ValueError('Invalid value for argument interpretability')



        # whethet to use CUDA for TM training
        CUDA = False

        linucb = LinearUCB(X_processed, y_encoded)
        treebootstrap = TreeBootstrap(X_processed, y_encoded)
        thompsamplingtm = ThompsonSamplingTM(X_processed, X_binarized, y, y_encoded, clauses, T, s,
                                             number_of_state_bits,
                                             cuda=CUDA, drop_clause_p=drop_clause_p)
        epsilongreedytm = EpsilonGreedyTM(X_processed, X_binarized, y, y_encoded, clauses, T, s, number_of_state_bits,
                                          cuda=CUDA, drop_clause_p=drop_clause_p)
        epsilongreedylogisticregression = EpsilonGreedyLogisticRegression(X_processed, y_encoded)
        epsilongreedyneuralnetwork = EpsilonGreedyNeuralNetwork(X_processed, y_encoded)

        total_rewards_linucb = []
        total_rewards_dt_ts = []
        total_rewards_tm_ts = []
        total_rewards_tm = []
        total_rewards_lr = []
        total_rewards_nn = []
        for run in range(num_exp_runs):

            print("Running Experiment " + str(run + 1))

            rewards_linucb = linucb.train_linear_ucb(rounds, randomized_context=randomized_context,
                                                     reward_mat=reward_mat)

            rewards_dt_ts = treebootstrap.train_treebootstrap(rounds, randomized_context=randomized_context,
                                                              reward_mat=reward_mat)

            rewards_tm_ts, L1_tm_ts, Dnf_tm_ts = thompsamplingtm.train_tm_thompson_sampling(rounds,
                                                                                            randomized_context=randomized_context,
                                                                                            interpretability=interpretability,
                                                                                            reward_mat=reward_mat)

            rewards_tm, L1_tm, Dnf_tm = epsilongreedytm.train_tm_epsilon_greedy(rounds,
                                                                                exploration_probability=exploration_probability,
                                                                                randomized_context=randomized_context,
                                                                                reward_mat=reward_mat,
                                                                                interpretabiliy=interpretability)

            rewards_lr = epsilongreedylogisticregression.train_lr_epsilon_greedy(rounds,
                                                                                 exploration_probability=exploration_probability,
                                                                                 randomized_context=randomized_context,
                                                                                 reward_mat=reward_mat)

            rewards_nn = epsilongreedyneuralnetwork.train_nn_epsilon_greedy(rounds,
                                                                            exploration_probability=exploration_probability,
                                                                            randomized_context=randomized_context,
                                                                            reward_mat=reward_mat)

            total_rewards_linucb.append(rewards_linucb)
            total_rewards_dt_ts.append(rewards_dt_ts)
            total_rewards_tm_ts.append(rewards_tm_ts)
            total_rewards_tm.append(rewards_tm)
            total_rewards_lr.append(rewards_lr)
            total_rewards_nn.append(rewards_nn)

            rewards_linucb = np.mean(total_rewards_linucb, axis=0)
            rewards_dt_ts = np.mean(total_rewards_dt_ts, axis=0)
            rewards_tm_ts = np.mean(total_rewards_tm_ts, axis=0)
            rewards_tm = np.mean(total_rewards_tm, axis=0)
            rewards_lr = np.mean(total_rewards_lr, axis=0)
            rewards_nn = np.mean(total_rewards_nn, axis=0)

            mean_rewards_linucb = np.zeros_like(rewards_linucb, dtype=float)
            mean_rewards_dt_ts = np.zeros_like(rewards_dt_ts, dtype=float)
            mean_rewards_tm_ts = np.zeros_like(rewards_tm_ts, dtype=float)
            mean_rewards_tm = np.zeros_like(rewards_tm, dtype=float)
            mean_rewards_lr = np.zeros_like(rewards_lr, dtype=float)
            mean_rewards_nn = np.zeros_like(rewards_nn, dtype=float)
            for index in range(rewards_tm.shape[0]):
                mean_rewards_linucb[index] = np.mean(rewards_linucb[:index + 1])
                mean_rewards_dt_ts[index] = np.mean(rewards_dt_ts[:index + 1])
                mean_rewards_tm_ts[index] = np.mean(rewards_tm_ts[:index + 1])
                mean_rewards_tm[index] = np.mean(rewards_tm[:index + 1])
                mean_rewards_lr[index] = np.mean(rewards_lr[:index + 1])
                mean_rewards_nn[index] = np.mean(rewards_nn[:index + 1])

            path = "Results/"
            if not os.path.exists(path):
                os.makedirs(path)

            exp_name = args.dataset_name.upper()
            results_path = args.results_dir + exp_name

            if not os.path.exists(results_path):
                os.makedirs(results_path)

            if L1_tm_ts != None and Dnf_tm_ts != None:

                f = open(results_path + '/' + 'interpretability_tm_ts.txt', 'w')
                num_classifiers = y_encoded.shape[1]
                for i in range(num_classifiers):
                    f.write("-------ARM:" + str(i + 1) + "-------\n")
                    f.write("Boolean Exp:  ")
                    f.write(str(L1_tm_ts[i]))
                    f.write("\n\n\n")
                    f.write("DNF Exp:   ")
                    f.write(str(Dnf_tm_ts[i]))
                    f.write("\n\n\n\n\n")
                f.close()

            if L1_tm != None and Dnf_tm != None:

                f = open(results_path + '/' + 'interpretability_tm.txt', 'w')
                num_classifiers = y_encoded.shape[1]
                for i in range(num_classifiers):
                    f.write("-------ARM:" + str(i + 1) + "-------\n")
                    f.write("Boolean Exp:  ")
                    f.write(str(L1_tm_ts[i]))
                    f.write("\n\n\n")
                    f.write("DNF Exp:   ")
                    f.write(str(Dnf_tm_ts[i]))
                    f.write("\n\n\n\n\n")
                f.close()

            np.save(results_path + '/' + 'mean_rewards_linucb.npy', mean_rewards_linucb)
            np.save(results_path + '/' + 'mean_rewards_dt_ts.npy', mean_rewards_dt_ts)
            np.save(results_path + '/' + 'mean_rewards_tm_ts.npy', mean_rewards_tm_ts)
            np.save(results_path + '/' + 'mean_rewards_tm.npy', mean_rewards_tm)
            np.save(results_path + '/' + 'mean_rewards_lr.npy', mean_rewards_lr)
            np.save(results_path + '/' + 'mean_rewards_nn.npy', mean_rewards_nn)

            plt.plot(mean_rewards_linucb, label='Linear UCB', color='cyan')
            plt.plot(mean_rewards_dt_ts, label='Tree Bootstrap', color='purple')
            plt.plot(mean_rewards_tm_ts, label='Tsetlin Machine(TS)', color='orange')
            plt.plot(mean_rewards_tm, label='Tsetlin Machine', color='red')
            plt.plot(mean_rewards_lr, label='Logistic Regression', color='green')
            plt.plot(mean_rewards_nn, label='Neural Network', color='blue')
            plt.xlabel("Steps")
            plt.title("Cumulative Mean Rewards on " + exp_name + " dataset")
            plt.grid()
            plt.legend()
            plt.savefig(results_path + "/Performance_Comparison.pdf")
            plt.close()





    else:
        raise ValueError("Invalid argument type for default configuration")
