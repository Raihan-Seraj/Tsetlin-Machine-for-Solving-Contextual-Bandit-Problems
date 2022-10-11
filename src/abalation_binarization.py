from symtable import Class
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
from contextualbandits.online import LinUCB,LogisticUCB
from sklearn.preprocessing import KBinsDiscretizer
from copy import deepcopy
import re
import os


from data_loader import DataLoader
from contextual_bandit_algorithms import LinearUCB, TreeBootstrap, ThompsonSamplingTM, EpsilonGreedyTM, EpsilonGreedyLogisticRegression, EpsilonGreedyNeuralNetwork


'''
Performs an abalation studies of Tsetlin Machine with different binarization of the dataset
the results are saved as npy files in the folder abalation

Args:
    dataset_name: String consisting of the name of the dataset the choices are ['Iris', 'Breast Cancer', 'Simulated Article','Movielens',
    'Adult', 'Statlog(Shuttle)','Covertype']

    binarization_value: a list of binarization values 

    num_experiments: The number of experiment to report the average results over 

output: The performance of the of different levels of binarization

'''

class Abalation_Binarization(object):

    def __init__(self, dataset_name, binarization_values,num_experiments=1):

        self.dataset_name = dataset_name
        self.binarization_values = binarization_values
        self.num_experiments = num_experiments
        self.interpretability = False
        self.randomized_context = True
        self.CUDA = False
        self. dl  = DataLoader()
        self.exploration_probability = 0.1

        print("Loading params for "+ str(dataset_name)+' dataset')
        if self.dataset_name =='Iris':
          
            self.rounds = 1000
            #Parameters for TM 
            self.clauses = 1200 
            self.T = 1000
            self.s = 8.0   
            self.number_of_state_bits = 10
            self.drop_clause_p=0.0
        
        elif self.dataset_name == 'Breast Cancer':
    
            #Number of rounds for each run
            self.rounds = 1000

            #Parameters for TM
            self.clauses = 650
            self.T = 300 
            self.s = 5.0
            self.drop_clause_p=0.0
            self.number_of_state_bits = 10

        
        elif self.dataset_name == 'Movielens':
            #Number of rounds per run
            self.rounds = 1000

            #parameters for TM 
            self.clauses = 4000 
            self.T = 3000
            self.s = 8.0  
            self.number_of_state_bits = 8
            self.drop_clause_p=0.0

           

        elif self.dataset_name == 'Simulated Article':

            self.rounds = 2000
            #Parameters of TM 
            self.clauses = 2000 
            self.T = 1500
            self.s = 5.0 
            self.number_of_state_bits = 10
            self.drop_clause_p=0.3
        
        elif self.dataset_name == 'Adult':

            self.rounds=2000
            self.clauses = 1200
            self.T = 800
            self.s = 5.0
            self.drop_clause_p = 0.1
            self.number_of_state_bits = 8

        elif self.dataset_name == 'Statlog(Shuttle)':
            self.rounds = 2000
            self.clauses = 1200
            self.T = 800
            self.s = 5.0
            self.number_of_state_bits = 8
            self.drop_clause_p=0.0
        
        elif self.dataset_name == 'Covertype':
            self.rounds = 2000
            self.clauses = 1200
            self.T = 800
            self.s = 5.0
            self.drop_clause_p = 0.1
            self.number_of_state_bits = 8
        
        else:
            raise ValueError("Invalid arguement for data set ")
    

    def run_abalation(self):
        clrs = ['red','blue', 'green','cyan','purple']
        dl = DataLoader()
        fig, axs = plt.subplots(2)

        for bin_idx,bin in enumerate(self.binarization_values):

            if self.dataset_name=='Iris':
                X_processed, X_binarized, y, y_encoded,reward_mat  = dl.load_iris_data(max_bits=bin)
            
            elif self.dataset_name=='Breast Cancer':
                X_processed, X_binarized, y, y_encoded,reward_mat  = dl.load_breast_cancer_data(max_bits=bin)

            elif self.dataset_name == 'Movielens':
                X_processed, X_binarized, y, y_encoded,reward_mat  = dl.load_movielens_data(max_bits=bin)
            
            elif self.dataset_name == 'Simulated Article':
                X_processed, X_binarized, y, y_encoded,reward_mat  = dl.load_simulated_article_data(max_bits=bin)
            
            elif self.dataset_name == 'Adult':
                X_processed, X_binarized, y, y_encoded,reward_mat  = dl.load_adult_dataset(max_bits=bin)
            
            elif self.dataset_name == 'Statlog(Shuttle)':
                X_processed, X_binarized, y, y_encoded,reward_mat  = dl.load_statlog_shuttle_dataset(max_bits=bin)

            elif self.dataset_name == 'Covertype':
                X_processed, X_binarized, y, y_encoded,reward_mat  = dl.load_covertype_dataset(max_bits=bin)
        

            thompsamplingtm = ThompsonSamplingTM(X_processed,X_binarized,y,y_encoded,self.clauses,self.T,self.s,self.number_of_state_bits,cuda=self.CUDA,drop_clause_p=self.drop_clause_p)
            epsilongreedytm = EpsilonGreedyTM(X_processed,X_binarized,y,y_encoded,self.clauses,self.T,self.s,self.number_of_state_bits,cuda=self.CUDA,drop_clause_p=self.drop_clause_p)

            total_rewards_tm_ts = []
            total_rewards_tm = []

            for run in range(self.num_experiments):

                print("Running Experiment "+ str(run+1))
                
                

                
                rewards_tm_ts,L1_tm_ts,Dnf_tm_ts = thompsamplingtm.train_tm_thompson_sampling(self.rounds,randomized_context=self.randomized_context,
                interpretability=self.interpretability,reward_mat=reward_mat)
                
                rewards_tm,L1_tm,Dnf_tm = epsilongreedytm.train_tm_epsilon_greedy(self.rounds,exploration_probability=self.exploration_probability,
                randomized_context=self.randomized_context,reward_mat=reward_mat,interpretabiliy=self.interpretability)

                total_rewards_tm_ts.append(rewards_tm_ts)
                total_rewards_tm.append(rewards_tm)
     
                
                rewards_tm_ts = np.mean(total_rewards_tm_ts,axis=0)    
                rewards_tm = np.mean(total_rewards_tm,axis=0)
                
                
                mean_rewards_tm_ts = np.zeros_like(rewards_tm_ts, dtype=float)
                mean_rewards_tm = np.zeros_like(rewards_tm, dtype=float)
                
                for index in range(rewards_tm.shape[0]):
                    
                    
                    
                    mean_rewards_tm_ts[index] = np.mean(rewards_tm_ts[:index+1])
                    mean_rewards_tm[index] = np.mean(rewards_tm[:index+1])
                   


                path = "Results_Abalation/"
                if not os.path.exists(path):
                    os.makedirs(path)
                
                exp_name = self.dataset_name
                results_path = "Results_Abalation/"+exp_name

                if not os.path.exists(results_path):
                    os.makedirs(results_path)
                
                if L1_tm_ts!=None and Dnf_tm_ts!=None:

                    f = open(results_path+'/'+'interpretability_tm_ts_abalation.txt','w')
                    num_classifiers = y_encoded.shape[1]
                    for i in range(num_classifiers):
                        f.write("-------ARM:"+str(i+1)+"-------\n")
                        f.write("Boolean Exp:  ")
                        f.write(str(L1_tm_ts[i]))
                        f.write("\n\n\n")
                        f.write("DNF Exp:   ")
                        f.write(str(Dnf_tm_ts[i]))
                        f.write("\n\n\n\n\n")
                    f.close()

                if L1_tm!=None and Dnf_tm!=None:

                    f = open(results_path+'/'+'interpretability_tm_abalation.txt','w')
                    num_classifiers = y_encoded.shape[1]
                    for i in range(num_classifiers):
                        f.write("-------ARM:"+str(i+1)+"-------\n")
                        f.write("Boolean Exp:  ")
                        f.write(str(L1_tm_ts[i]))
                        f.write("\n\n\n")
                        f.write("DNF Exp:   ")
                        f.write(str(Dnf_tm_ts[i]))
                        f.write("\n\n\n\n\n")
                    f.close()


                np.save(results_path+'/'+'mean_rewards_tm_ts_abalation_bin_'+str(bin)+'.npy',mean_rewards_tm_ts)
                np.save(results_path+'/'+'mean_rewards_tm_abalation_bin_'+str(bin)+'.npy',mean_rewards_tm)
                

            
            axs[0].plot(mean_rewards_tm_ts,label='TM Thompson Sampling, Binarization='+str(bin),color=clrs[bin_idx])
            axs[1].plot(mean_rewards_tm,label='TM Epsilon-Greedy, Binarization='+str(bin),color=clrs[bin_idx])

            plt.xlabel("Steps")

            fig.suptitle("Cumulative Mean Rewards on "+ exp_name+ " dataset")
            #plt.title("Cumulative Mean Rewards on "+ exp_name+ " dataset")
            axs[0].grid()
            axs[1].grid()
            axs[0].legend()
            axs[1].legend()
        plt.savefig(results_path+"/Performance_Comparison_Binarization.pdf")
        plt.close()






'''
---------------------------------------Runnig Binarization Abalation---------------------------------------------------------
'''

if __name__ =='__main__':



    #ab = Abalation_Binarization('Iris',[4,6,8,10,12],1)
    ab = Abalation_Binarization('Iris',[4,6],3)


    ab.run_abalation()

                    





                    
                    

            
            