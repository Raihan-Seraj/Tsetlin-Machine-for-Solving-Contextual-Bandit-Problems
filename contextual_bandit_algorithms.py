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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from  pyTsetlinMachine.tm import MultiClassConvolutionalTsetlinMachine2D
from matplotlib.colors import ListedColormap
from tmu.tsetlin_machine import TMCoalescedClassifier
from copy import deepcopy
import re
import os
    

#---------------------------------------------------------LINEAR UCB-------------------------------------------------------------------------------------------
class LinearUCB(object):
    def __init__(self,X,y,alpha=0.1):

       
        self.X =X
        self.y = y
        self.alpha = alpha
        self.num_arms = self.y.shape[1]

    def train_linear_ucb(self,rounds, randomized_context=False,reward_mat = None):
        
        print("Linear UCB")
       
        
        linucb = LinUCB(nchoices = self.num_arms, beta_prior = None, alpha = self.alpha,
                    ucb_from_empty = False)  
        
        rewards_history = np.zeros(rounds)
 
        for i in tqdm(range(rounds)):

            if randomized_context:
                index = random.randint(0,self.X.shape[0]-1)


            else:
               
                index = i % self.X.shape[0]
                

        
            context = self.X[index,:]
            
            #predicted actions for each batch
            action = linucb.predict(context)[0]

            rewards_received = reward_mat[index,action] if reward_mat is not None else self.y[index,action]
            
            
            rewards_history[i] = rewards_received

            linucb.partial_fit(context, np.array([action]), np.array([rewards_received]))
            
        return rewards_history
            


#---------------------------------------------------------------------TREE BOOTSTRAP--------------------------------------------------------------------------------------------------

class TreeBootstrap(object):
    def __init__(self,X,y):
    
        self.X  = X
        self.y = y.astype(dtype=np.uint32)
        self.num_arms = y.shape[1]
        self.classifiers = [DecisionTreeClassifier() for _ in range(self.num_arms)]




    
    def train_treebootstrap(self, rounds, randomized_context=False, reward_mat=False):
        
        print("Tree Bootstrap")

        
        data = [[] for _ in range(self.num_arms) ]
        reward_history = np.zeros(rounds)
        for i in tqdm(range(rounds)):

            if randomized_context:
                #choose a randomized index
                index = random.randint(0,self.X.shape[0]-1)

            else:
                #go sequentially
                index = i%self.X.shape[0]
            
            #get context value
            context = self.X[index,:]
       
            for arm in range(self.num_arms):

                #if the data arm is not empty
                if len(data[arm])>0:
                    #get randomized indexes to be sampled with replacement
                    bootstrapped_indexes = np.random.randint(0,len(data[arm]),size=len(data[arm]))
                    #get bootstrapped data
                    bootstrapped_data = np.array(data[arm]).reshape(len(data[arm]),-1)
                    bootstrapped_data = bootstrapped_data[bootstrapped_indexes,:]

         
                    self.classifiers[arm].fit(bootstrapped_data[:,0:-1],bootstrapped_data[:,-1])

                
                else:
                    #take a randomized action
                    action = random.randint(0,self.num_arms-1)
                    #get the reward
                    reward = reward_mat[index,action] if reward_mat is not None else self.y[index,action]
                    #append reward with the context
                    rew_ctx = np.append(context,reward)
                    #add context reward pair to the observation
                    data[arm].append(rew_ctx)
                
            #choose arm with maximum probability of succress
            predictions = np.zeros(self.num_arms)
            for arm in range(self.num_arms):

                try:
                    pred = self.classifiers[arm].predict(context.reshape(1,-1))[0]
                    predictions[arm]=pred
                except NotFittedError as e:
                    pred = random.randint(0,1)
                    predictions[arm]=pred

            predictions_stabilized = predictions+1e-12
            normalized_prediction = predictions_stabilized/(predictions_stabilized.sum())
            
            #choose the arm with the most probability of success
            action = np.random.multinomial(1,normalized_prediction).argmax()

            reward = reward_mat[index,action] if reward_mat is not None else self.y[index,action]
            rew_ctx = np.append(context,reward)
            #append the new data
            data[action].append(rew_ctx)

            reward_history[i] = reward
        
  
   
        return reward_history


#---------------------------------------------------------------TM THOMPSON SAMPLING---------------------------------------------------------------------------------------------

class ThompsonSamplingTM(object):
    def __init__(self,X_original,X,y_original,y,clauses, T, s,number_of_state_bits,cuda=False,drop_clause_p=0.0,cnn=False,coalesced=False):
        self.y_original = y_original
        self.X_original = X_original
        self.X  = X.astype(dtype=np.uint32)
        self.y = y.astype(dtype=np.uint32)
        self.num_arms = y.shape[1]
        self.clauses = clauses
        self.T = T
        self.s = s
        self.number_of_state_bits = number_of_state_bits
        self.drop_clause_p = drop_clause_p
        self.cuda  = cuda
        self.cnn = cnn
        self.coalesced = coalesced

        if self.cuda and self.cnn==False:
            print("Using CUDA")
            from PyTsetlinMachineCUDA.tm import MultiClassTsetlinMachine as mctmcuda
            self.classifiers = [mctmcuda(self.clauses,self.T,self.s,number_of_state_bits=self.number_of_state_bits, boost_true_positive_feedback=0) for _ in range(self.num_arms)]
        elif self.cuda==False and self.cnn:
            self.classifiers = [MultiClassConvolutionalTsetlinMachine2D(self.clauses,self.T,self.s,(10,10),boost_true_positive_feedback=0) for _ in range(self.num_arms)]
        elif self.cuda and self.cnn:
            from PyTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D as mctm2d
            self.classifiers = [mctm2d(self.clauses, self.T,self.s,(10,10),boost_true_positive_feedback=0) for _ in range(self.num_arms)]

        else:
            self.classifiers = [MultiClassTsetlinMachine(self.clauses,self.T,self.s,number_of_state_bits=self.number_of_state_bits, boost_true_positive_feedback=0,clause_drop_p=self.drop_clause_p,weighted_clauses=True) for _ in range(self.num_arms)]
        
    
    def interpret(self, n_clauses, number_of_features):
        all_L1 = []
        all_DNF = []
        for i in range(len(self.classifiers)):
     
            L1 = []

            for j in range(0, n_clauses, 2):
                
                l1 = []
                c1 = []
                for k in range(1,number_of_features*2+1):
                    s = k-1
                    if self.classifiers[i].ta_action(0, j, s) == 1:
                        if s < number_of_features:
                            l1.append(" x%d" % (k))
                            c1.append(-k)
                        else:
                            l1.append("~x%d" % (k-number_of_features))
                            c1.append(k-number_of_features)
                L1.append(" & ".join(l1))
                #print(" & ".join(l1))
            L1 = ") | (".join(L1)
            L1 = "("+L1+")"
            
            L1 = L1.replace("| ()","")
            L1 = L1.replace("() |","")
            L1 = L1.replace("()  |","")
            DNF = to_dnf(L1)
            
            all_L1.append(L1)
            all_DNF.append(DNF)
           
        
        return all_L1, all_DNF

    
           
   




    
    def train_tm_thompson_sampling(self, rounds, randomized_context=False, interpretability=False, reward_mat=False):
        
        print("Tsetlin Machine Thompson Sampling ")
        
        if not self.coalesced:
            for tm in self.classifiers:
                for arm in range(self.y.shape[1]):
                    
                    tm.fit(self.X,self.y[:,arm],epochs=0,incremental=True)
      

        L1 = None
        Dnf = None

        #initializing data per arm which is  a list 
        data = [[] for _ in range(self.num_arms) ]
        reward_history = np.zeros(rounds)
        for i in tqdm(range(rounds)):
            
            if randomized_context:
                #choose a randomized index
                index = random.randint(0,self.X.shape[0]-1)

            else:
                #go sequentially
                index = i%self.X.shape[0]
            
            #get context value
            context = self.X[index,:]

            for arm in range(self.num_arms):
                
                #if the data arm is not empty
                if len(data[arm])>0:
                    #get randomized indexes to be sampled with replacement
                    bootstrapped_indexes = np.random.randint(0,len(data[arm]),size=len(data[arm]))
                    #get bootstrapped data
                    bootstrapped_mat = np.array(data[arm]).reshape(len(data[arm]),-1)
                    bootstrapped_data = bootstrapped_mat[bootstrapped_indexes,:]

                    # print("Data to be fitted is\n", bootstrapped_data[:,0:-1],"\n")
                    # print("y to be fitted is\n ", bootstrapped_data[:,-1])
                    if self.coalesced:
                        
                        self.classifiers[arm].fit(bootstrapped_data[:,0:-1],bootstrapped_data[:,-1])
                    else:
                        self.classifiers[arm].fit(bootstrapped_data[:,0:-1],bootstrapped_data[:,-1],epochs=1,incremental=True)


                
                else:
                    #take a randomized action
                    action = random.randint(0,self.num_arms-1)
                    #get the reward
                    reward = reward_mat[index,action] if reward_mat is not None else self.y[index,action]
                    #append reward with the context
                    rew_ctx = np.append(context,reward)
                    #add context reward pair to the observation
                    data[arm].append(rew_ctx)
                
            #choose arm with maximum probability of succress

            predictions = np.zeros(self.num_arms)
            for arm in range(self.num_arms):

                try:
           
                    pred = self.classifiers[arm].predict(context.reshape(1,-1))[0]
                    predictions[arm]=pred
                   
                except AttributeError as e:
                    pred = random.randint(0,1)
                    predictions[arm]=pred

            predictions_stabilized = predictions+1e-12

            normalized_prediction = predictions_stabilized/(predictions_stabilized.sum())
            
            #choose the arm with the most probability of success
            action = np.random.multinomial(1,normalized_prediction).argmax()

            reward = reward_mat[index,action] if reward_mat is not None else self.y[index,action]
            rew_ctx = np.append(context,reward)
            #append the new data
            data[action].append(rew_ctx)

            reward_history[i]=reward

            
       

        if interpretability:
            print("Performing Interpretability")
            L1, Dnf = self.interpret(self.clauses,self.X.shape[1])
        
        return reward_history,L1,Dnf
    


  #  -----------------------------------------------------------------TM Epsilon Greedy-------------------------------------------------------------------------------

class EpsilonGreedyTM(object):
    def __init__(self,X_original,X,y_original,y,clauses, T, s,number_of_state_bits,cuda=False,drop_clause_p=0.0,cnn=False,coalesced=False):
        self.X_original = X_original
        self.y_original=y_original
        self.X  = X.astype(dtype=np.uint32)
        self.y = y.astype(dtype=np.uint32)
        self.num_arms = y.shape[1]
        self.clauses = clauses
        self.T = T
        self.s = s
        self.number_of_state_bits = number_of_state_bits
        self.drop_clause_p = drop_clause_p
        self.cuda  = cuda
        self.cnn = cnn
        self.coalesced = coalesced

        if self.cuda and self.cnn==False:
            print("Using CUDA")
            from PyTsetlinMachineCUDA.tm import MultiClassTsetlinMachine as mctmcuda
            self.classifiers = [mctmcuda(self.clauses,self.T,self.s,number_of_state_bits=self.number_of_state_bits, boost_true_positive_feedback=0) for _ in range(self.num_arms)]
        elif self.cuda==False and self.cnn:
            self.classifiers = [MultiClassConvolutionalTsetlinMachine2D(self.clauses,self.T,self.s,(10,10),boost_true_positive_feedback=0) for _ in range(self.num_arms)]
        elif self.cuda and self.cnn:
            print("Using CUDA")
            from PyTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D as mctm2d
            self.classifiers = [mctm2d(self.clauses, self.T,self.s,(10,10),boost_true_positive_feedback=0) for _ in range(self.num_arms)]
        

        else:
            self.classifiers = [MultiClassTsetlinMachine(self.clauses,self.T,self.s,number_of_state_bits=self.number_of_state_bits, boost_true_positive_feedback=0,clause_drop_p=self.drop_clause_p,weighted_clauses=True) for _ in range(self.num_arms)]



    def interpret(self, n_clauses, number_of_features):
        all_L1 = []
        all_DNF = []
        for i in range(len(self.classifiers)):
            
            L1 = []

            for j in range(0, n_clauses, 2):
   
                l1 = []
                c1 = []
                for k in range(1,number_of_features*2+1):
                    s = k-1
                    if self.classifiers[i].ta_action(0, j, s) == 1:
                        if s < number_of_features:
                            l1.append(" x%d" % (k))
                            c1.append(-k)
                        else:
                            l1.append("~x%d" % (k-number_of_features))
                            c1.append(k-number_of_features)
                L1.append(" & ".join(l1))
                
            L1 = ") | (".join(L1)
            L1 = "("+L1+")"
           
            L1 = L1.replace("| ()","")
            L1 = L1.replace("() |","")
            L1 = L1.replace("()  |","")
            DNF = to_dnf(L1)
            
            all_L1.append(L1)
            all_DNF.append(DNF)
           
        
        return all_L1, all_DNF


    


    def action_policy_tm(self,context,rewards_data,exploration_probability):
    
        #obtain the number of arms 
       

        #choose a random number
        toss = random.uniform(0,1)

        #this condition corresponds to choosing the best action
        if toss > exploration_probability:
        
            rewards = np.zeros(self.num_arms)

            for classifier_idx, classifier in enumerate(self.classifiers):
                action_rewards = rewards_data[:,classifier_idx]
                #if there are three unique action rewards, do a classification predict 
                if len(np.unique(action_rewards.flatten()))==3:
                    try:
                        
                        action_score = classifier.predict(context.astype(dtype=np.uint32).reshape(1,-1))[0]
                    except NotFittedError as e:
                        #action score is choosen in terms of a beta distribution 
                        action_score = np.random.beta(3.0/self.num_arms,4)
                else:
                    #scoring in terms of a beta distribution 
                    action_score = np.random.beta(3.0/self.num_arms, 4)
                
                rewards[classifier_idx] = action_score
            
            max_rewards = max(rewards)
            best_actions = np.argwhere(rewards==max_rewards).flatten()
            action = np.random.choice(best_actions)
        else:

            action = random.randint(0,self.num_arms-1)
        
        return action 
    

    def fit_classifiers_tm(self,step,context_data, rewards_data) -> None:
        
        #see the context data seen so far
        context_so_far = np.array(context_data[:step]) 

        #rewards collected so far
        rewards_so_far = rewards_data[:step]

        for classifier_idx, _ in enumerate(self.classifiers):
            
            #indexing the rewards collected for the action
            action_rewards = rewards_so_far[:,classifier_idx]
            
            # finding the index which received the reward

            index = np.argwhere(action_rewards!=-1).flatten()

            action_rewards = action_rewards[index]
            
            if len(np.unique(action_rewards))==2:

                action_contexts = context_so_far[index]

               
                self.classifiers[classifier_idx].fit(action_contexts.astype(dtype=np.uint32),action_rewards.astype(dtype=np.uint32),epochs=1,incremental=True)
        


    def train_tm_epsilon_greedy(self,rounds, exploration_probability=0.1,randomized_context=False,reward_mat=None,interpretabiliy=False):
        print('Tsetlin Machine Epsilon Greedy')

        # initial fitting of the classifier ---> check
        if not self.coalesced:
            for tm in self.classifiers:
                for arm in range(self.y.shape[1]):
                    tm.fit(self.X,self.y[:,arm],epochs=0,incremental=True)
        
        #intialzing context data matrix, which is of shape (rounds x context_dim)
        context_data = []#np.zeros([rounds,self.X.shape[1]])

        #initializing rewards data to be all equal to -1 of shape (rounds x num_arms)
        rewards_data = np.full((rounds, self.num_arms),-1,dtype=float)


        rewards_history = np.zeros(rounds)

        L1 =None
        Dnf = None
        for i in tqdm(range(rounds)):
           
            #whether to choose context sequentially or randomely.
            if randomized_context:

                index = random.randint(0,self.X.shape[0]-1)
            else:
                index = i % self.X.shape[0]

            #get the context 
            context = self.X[index,:]
            
            #selecting action according to the policy
            action = self.action_policy_tm(context,rewards_data,exploration_probability)

            #getting the reward for the selected action 
            if reward_mat is not None:
                reward = reward_mat[index, action]
            else:
                reward = self.y[index,action]
            #adding reward to the history
            rewards_history[i] = reward
            
            
            context_data.append(context)
            rewards_data[i,action] = reward
           
            self.fit_classifiers_tm(i,context_data, rewards_data)
            
        if interpretabiliy==True:
            print("Performing Interpretability")
            L1, Dnf=self.interpret(self.clauses, self.X.shape[1])
        
        
        return rewards_history,L1,Dnf
    


    
#----------------------------------------------------------Logistic Regression Epsilon Greedy------------------------------------------------------------------------

class EpsilonGreedyLogisticRegression(object):
    def __init__(self, X,y):
      
        self.X = X
        sc = StandardScaler()
        self.X = sc.fit_transform(self.X)
        self.y = y
        self.num_arms = self.y.shape[1]
        self.classifiers =  [LogisticRegression(solver="lbfgs", n_jobs=-1) for _ in range(self.num_arms)]

            
    def action_policy_lr(self,context,rewards_data,exploration_probability):
        


        #choose a random number
        toss = random.uniform(0,1)

        #this condition corresponds to choosing the best action
        if toss > exploration_probability:
        
            rewards = np.zeros(self.num_arms)

            for classifier_idx, classifier in enumerate(self.classifiers):
                action_rewards = rewards_data[:,classifier_idx]
                #if there are three unique action rewards, do a classification predict 
                if len(np.unique(action_rewards.flatten()))==3:
                    try:
                        
                        action_score = classifier.predict(context.reshape(1,-1))[0]
                    except NotFittedError as e:
                        #action score is choosen in terms of a beta distribution 
                        action_score = np.random.beta(3.0/self.num_arms,4)
                else:
                    #scoring in terms of a beta distribution 
                    action_score = np.random.beta(3.0/self.num_arms, 4)
                
                rewards[classifier_idx] = action_score
            
            max_rewards = max(rewards)
            best_actions = np.argwhere(rewards==max_rewards).flatten()
            action = np.random.choice(best_actions)
        else:

            action = random.randint(0,self.num_arms-1)
        
        return action 
    

    def fit_classifiers_lr(self,step,context_data, rewards_data) -> None:
        
        #see the context data seen so far
        context_so_far = context_data[:step] 

        #rewards collected so far
        rewards_so_far = rewards_data[:step]

        for classifier_idx, _ in enumerate(self.classifiers):
            
            #indexing the rewards collected for the action
            action_rewards = rewards_so_far[:,classifier_idx]
            
            # finding the index which received the reward

            index = np.argwhere(action_rewards!=-1).flatten()

            action_rewards = action_rewards[index]
            
            if len(np.unique(action_rewards))==2:

                action_contexts = context_so_far[index]
               
                self.classifiers[classifier_idx].fit(action_contexts,action_rewards)
        
        
    

    def train_lr_epsilon_greedy(self,rounds, exploration_probability=0.1,randomized_context=False,reward_mat=None):
        print("Logistic Regression Epsilon Greedy")

        #intialzing context data matrix, which is of shape (rounds x context_dim)
        context_data = np.zeros([rounds,self.X.shape[1]])

        #initializing rewards data to be all equal to -1 of shape (rounds x num_arms)
        rewards_data = np.full((rounds, self.num_arms),-1,dtype=float)


        rewards_history = np.zeros(rounds)

        L1 =None
        Dnf = None
        for i in tqdm(range(rounds)):

            #whether to choose context sequentially or randomely.
            if randomized_context:

                index = np.random.randint(0,self.X.shape[0])
            else:
                index = i % self.X.shape[0]

            #get the context 
            context = self.X[index,:]
            
            #selecting action according to the policy
            action = self.action_policy_lr(context,rewards_data,exploration_probability)

            #getting the reward for the selected action 
            reward = reward_mat[index,action] if reward_mat is not None else self.y[index,action]
            #adding reward to the history
            rewards_history[i] = reward
            
            context_data[i] = context
            rewards_data[i,action] = reward
           
            self.fit_classifiers_lr(i,context_data, rewards_data)
      
        return rewards_history

    
 #--------------------------------------------------------------------------Neural Network Epsilon Greedy -----------------------------------------------------------------------------
    


class EpsilonGreedyNeuralNetwork(object):
    def __init__(self, X,y):
      
        self.X = X
        self.y = y
        self.num_arms = self.y.shape[1]
        self.classifiers =  [MLPClassifier(solver='adam',alpha=1e-5,max_iter=1) for _ in range(self.num_arms)]

            
    def action_policy_nn(self,context,rewards_data,exploration_probability):
        


        #choose a random number
        toss = random.uniform(0,1)

        #this condition corresponds to choosing the best action
        if toss > exploration_probability:
        
            rewards = np.zeros(self.num_arms)

            for classifier_idx, classifier in enumerate(self.classifiers):
                action_rewards = rewards_data[:,classifier_idx]
                #if there are three unique action rewards, do a classification predict 
                if len(np.unique(action_rewards.flatten()))==3:
                    try:
                        
                        action_score = classifier.predict(context.reshape(1,-1))[0]
                    except NotFittedError as e:
                        #action score is choosen in terms of a beta distribution 
                        action_score = np.random.beta(3.0/self.num_arms,4)
                else:
                    #scoring in terms of a beta distribution 
                    action_score = np.random.beta(3.0/self.num_arms, 4)
                
                rewards[classifier_idx] = action_score
            
            max_rewards = max(rewards)
            best_actions = np.argwhere(rewards==max_rewards).flatten()
            action = np.random.choice(best_actions)
        else:

            action = random.randint(0,self.num_arms-1)
        
        return action 


    def fit_classifiers_nn(self,step,context_data, rewards_data) -> None:
        
        #see the context data seen so far
        context_so_far = context_data[:step] 

        #rewards collected so far
        rewards_so_far = rewards_data[:step]

        for classifier_idx, _ in enumerate(self.classifiers):
            
            #indexing the rewards collected for the action
            action_rewards = rewards_so_far[:,classifier_idx]
            
            # finding the index which received the reward

            index = np.argwhere(action_rewards!=-1).flatten()

            action_rewards = action_rewards[index]
            
            if len(np.unique(action_rewards))==2:

                action_contexts = context_so_far[index]
                
                self.classifiers[classifier_idx].partial_fit(action_contexts,action_rewards,classes=[0,1])
        
        


    def train_nn_epsilon_greedy(self,rounds, exploration_probability=0.1,randomized_context=False,reward_mat=None):
        print("Neural Network Epsilon Greedy")

        #intialzing context data matrix, which is of shape (rounds x context_dim)
        context_data = np.zeros([rounds,self.X.shape[1]])

        #initializing rewards data to be all equal to -1 of shape (rounds x num_arms)
        rewards_data = np.full((rounds, self.num_arms),-1,dtype=float)


        rewards_history = np.zeros(rounds)

       
        for i in tqdm(range(rounds)):

            #whether to choose context sequentially or randomely.
            if randomized_context:

                index = np.random.randint(0,self.X.shape[0])
            else:
                index = i % self.X.shape[0]

            #get the context 
            context = self.X[index,:]
            
            #selecting action according to the policy
            action = self.action_policy_nn(context,rewards_data,exploration_probability)

            #getting the reward for the selected action 
            reward = reward_mat[index,action] if reward_mat is not None else self.y[index,action]
            #adding reward to the history
            rewards_history[i] = reward
            
            context_data[i] = context
            rewards_data[i,action] = reward
            
            self.fit_classifiers_nn(i,context_data, rewards_data)
        
        return rewards_history
