import numpy as np
import pandas as pd
from pyTsetlinMachine.tools import Binarizer
from tqdm import tqdm
from sklearn import datasets
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import KBinsDiscretizer
from copy import deepcopy
from keras.datasets import mnist
from sklearn.preprocessing import LabelEncoder


import re
import os


class DataLoader(object):
    def __init__(self):
        super(DataLoader, self).__init__()
    
    '''
    -----------------------------------------------------------------LOAD IRIS DATASET---------------------------------------------------------------------------------
    '''

    #function loading iris dataset
    def load_iris_data(self):
    
        #loading the iris dataset the dataset is already shuffled
        dataset = datasets.load_iris()
        #getting the features
        X_processed = dataset.data 
        #getting the labels
        y = dataset.target

        #concatenating the data and shuffling them since the dataset is not shuffled
        data_concat = np.concatenate((X_processed,y.reshape(-1,1)),axis=1)
        #shuffling the data
        np.random.shuffle(data_concat)

        #splitting the X and y values again 
        X_processed=data_concat[:,0:X_processed.shape[1]]
        y =  data_concat[:,-1]
        #binarizing the data
        b = Binarizer(max_bits_per_feature=10)
        b.fit(X_processed)
        X_binarized = b.transform(X_processed)

        #returning the labels as one hot encoder 
        y_encoded = np.zeros([y.shape[0],len(np.unique(y))])

        for i in range(len(y)):
            y_encoded[i,int(y[i])]=1
        reward_mat = None
     
        return  X_processed, X_binarized, y,y_encoded, reward_mat


    
    '''
    --------------------------------------------------------------LOAD BREAST CANCER DATASET----------------------------------------------------------------------
    '''

    def load_breast_cancer_data(self):
        #loading the breast cancer dataset
        dataset = datasets.load_breast_cancer()

        #getting the features
        X_processed = dataset.data
        #getting the labels 
        y = dataset.target
     
        #concatenating the data and shuffling them since the dataset is not shuffled
        data_concat = np.concatenate((X_processed,y.reshape(-1,1)),axis=1)
        np.random.shuffle(data_concat)

        #splitting the X and y values again 
        X_processed=data_concat[:,0:X_processed.shape[1]]
        y =  data_concat[:,-1]
        
        #Binarizing the data
        b = Binarizer(max_bits_per_feature=10)

        b.fit(X_processed)
        
        X_binarized = b.transform(X_processed)

        #returning the labels as one hot encoder 
        y_encoded = np.zeros([y.shape[0],len(np.unique(y))])

        for i in range(len(y)):
        
            y_encoded[i,int(y[i])]=1
        
        #there are no custom rewards
        reward_mat = None
        
        return X_processed, X_binarized, y,y_encoded, reward_mat
    
    '''
    -------------------------------------------------------------LOAD NOISY XOR DATASET-----------------------------------------------------------------------------
    '''

    def load_noisy_xor_data(self):
        #loading the data which is already shuffled 
        data = np.loadtxt("datasets/NoisyXORTestData.txt")
        X = data[:,0:-1]
        y = data[:,-1]

        X_processed = X

        X_binarized = X

        #returning the labels as one hot encoder 
        y_encoded = np.zeros([y.shape[0],len(np.unique(y))])

        for i in range(len(y)):
            y_encoded[i,int(y[i])]=1
        

        reward_mat = None
        
        return  X_processed, X_binarized, y, y_encoded, reward_mat 

    

    

    
    '''
    ---------------------------------------------------------------------LOAD MOVIELENS DATASET------------------------------------------------------------------------
    '''
    ##check the dataset does not look ok
    def load_movielens_data(self):

        
        NUM_USERS = 943
        MOVIE_LENS_NUM_MOVIES = 1682

        file_name = 'datasets/ml-100k/u.data'

        
        #data_matrix = dataset_utilities.load_movielens_data(file_name,delimiter='\t')
        ratings_matrix = np.zeros([NUM_USERS,MOVIE_LENS_NUM_MOVIES])

        data = pd.read_csv(file_name,delimiter='\t')
        data.columns = ['user_id','item_id','rating','timestamp']
        for i in range(len(data)):
            ratings_matrix[int(data['user_id'][i])-1, int(data['item_id'][i])-1]=float(data['rating'][i])
        
       
        num_movies = 10
        rank = 10
        ratings_matrix = ratings_matrix[:,:num_movies]
        non_zero_users=list(np.nonzero(np.sum(ratings_matrix, axis=1) > 0.0)[0])
        ratings_matrix = ratings_matrix[non_zero_users,:]
        effective_users = len(non_zero_users)

        u, s, vh = np.linalg.svd(ratings_matrix, full_matrices=False)
        u_hat = u[:, :rank] * np.sqrt(s[:rank])
        v_hat = np.transpose(np.transpose(vh[:rank, :]) * np.sqrt(s[:rank]))

        approx_ratings_matrix = np.matmul(u_hat, v_hat)

        final_ratings_matrix = np.zeros_like(approx_ratings_matrix)
        
        contexts_bin = Binarizer(max_bits_per_feature=10)
        contexts_bin.fit(u_hat)

        for i in range(final_ratings_matrix.shape[0]):
            final_ratings_matrix[i,np.argmax(approx_ratings_matrix[i,:])]=1


        X_binarized = contexts_bin.transform(u_hat)
        X_processed = u_hat
        y_encoded = final_ratings_matrix
        

    
        reward_mat = None
        y=None
        return X_processed, X_binarized, y, y_encoded, reward_mat


    
    '''
    -------------------------------------------------------------LOAD SIMULATED ARTICLE DATASET---------------------------------------------------------------------
    '''
    def load_simulated_article_data(self):
        

        rank=10

        file_name = 'datasets/SimulatedArticleData.csv'

        data = pd.read_csv(file_name,delimiter=',')
        recommendations = data['Recommendation'].astype('category').cat.codes
        recommendations = recommendations.to_numpy()
        num_actions = len(np.unique(recommendations))
        data_mat = np.zeros([len(data),num_actions])

        for i in range(data_mat.shape[0]):

            data_mat[i,recommendations[i]]=data['Reward'][i]
        

        non_zero_users=list(np.nonzero(np.sum(data_mat, axis=1) > 0.0)[0]) 
        data_mat=data_mat[non_zero_users,:]  
        u, s, vh = np.linalg.svd(data_mat, full_matrices=False)
        u_hat = u[:, :rank] * np.sqrt(s[:rank])
        v_hat = np.transpose(np.transpose(vh[:rank, :]) * np.sqrt(s[:rank]))

        approx_ratings_matrix = np.matmul(u_hat, v_hat)

        final_ratings_matrix = np.zeros_like(approx_ratings_matrix)

        contexts_bin = Binarizer(max_bits_per_feature=10)
        contexts_bin.fit(u_hat)

        for i in range(final_ratings_matrix.shape[0]):
            final_ratings_matrix[i,np.argmax(approx_ratings_matrix[i,:])]=1


        X_binarized = contexts_bin.transform(u_hat)
        X_processed = u_hat
        y_encoded = final_ratings_matrix
        
        reward_mat=None
        y=None
        
        
   
        return X_processed, X_binarized,y,y_encoded,reward_mat







    '''
    --------------------------------------------------------------LOAD ADULT DATASET----------------------------------------------------------------------------------
    '''

    def load_adult_dataset(self):
        
        raw_data = pd.read_csv('datasets/adult.data',delimiter =',')
        raw_data.columns= ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex',
        'capital-gain','capital-loss','hours-per-week','native-country','income']

        data_processed = raw_data.sample(frac=1).reset_index(drop=True)
        data_processed.dropna(inplace=True)
        #processing all categorical features
        data_processed = data_processed[data_processed.occupation!=' ?']
        data_processed = data_processed[data_processed.workclass!=' ?']
        data_processed = data_processed[data_processed.education!=' ?']
        data_processed = data_processed[data_processed['marital-status']!=' ?']
        data_processed = data_processed[data_processed.relationship!=' ?']
        data_processed = data_processed[data_processed.race!=' ?']
        data_processed = data_processed[data_processed.sex!=' ?']
        data_processed = data_processed[data_processed.income!=' ?']
        data_processed = data_processed[data_processed['native-country']!=' ?']
        data_processed = data_processed[data_processed['age']!=' ?']
        data_processed = data_processed[data_processed['fnlwgt']!=' ?']
        data_processed = data_processed[data_processed['education-num']!=' ?']
        data_processed = data_processed[data_processed['capital-gain']!=' ?']
        data_processed = data_processed[data_processed['capital-loss']!=' ?']
        data_processed = data_processed[data_processed['hours-per-week']!=' ?']
        
    
    
        data_processed['workclass']=data_processed['workclass'].astype('category').cat.codes
        data_processed['education'] = data_processed['education'].astype('category').cat.codes
        data_processed['marital-status'] = data_processed['marital-status'].astype('category').cat.codes
        data_processed['occupation'] = data_processed['occupation'].astype('category').cat.codes 
        data_processed['relationship'] = data_processed['relationship'].astype('category').cat.codes
        data_processed['race'] = data_processed['race'].astype('category').cat.codes
        data_processed['sex'] = data_processed['sex'].astype('category').cat.codes
        data_processed['native-country'] = data_processed['native-country'].astype('category').cat.codes
        data_processed['income'] = data_processed['income'].astype('category').cat.codes


        #processing all features with continuous values 

        age_discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal')
        data_processed['age'] = age_discretizer.fit_transform(data_processed['age'].to_numpy().reshape(-1,1))

        flwgt_discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal')
        data_processed['fnlwgt'] = flwgt_discretizer.fit_transform(data_processed['fnlwgt'].to_numpy().reshape(-1,1))

        
        education_num_discretizer = KBinsDiscretizer(n_bins=3,encode='ordinal')
        data_processed['education-num'] = education_num_discretizer.fit_transform(data_processed['education-num'].to_numpy().reshape(-1,1))
        
        capital_gain_discretizer = KBinsDiscretizer(n_bins=2,encode='ordinal')
        data_processed['capital-gain'] = data_processed['capital-gain'].astype('category').cat.codes
        
        capital_loss_discretizer = KBinsDiscretizer(n_bins=3,encode='ordinal')
        data_processed['capital-loss'] = data_processed['capital-loss'].astype('category').cat.codes#capital_loss_discretizer.fit_transform(data_processed['capital-loss'].to_numpy().reshape(-1,1))

        
        hour_discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal')
        data_processed['hours-per-week'] = data_processed['hours-per-week'].astype('category').cat.codes#hour_discretizer.fit_transform(data_processed['hours-per-week'].to_numpy().reshape(-1,1))
        
        
        vals=data_processed['occupation'].value_counts()
        #removing the least occuring classes
        
        #uncomment to remove least appearing examples
        # data_processed = data_processed[data_processed['occupation']!=8]
        # data_processed = data_processed[data_processed['occupation']!=1]
        
        # #converting to categorical again
        # data_processed['occupation'] = data_processed['occupation'].astype('category').cat.codes

        X_processed = data_processed.to_numpy()

        b = Binarizer(max_bits_per_feature=10)

        b.fit(X_processed)

        X_binarized = b.transform(X_processed)

        
        y_processed = data_processed['occupation'].to_numpy()

        y_encoded = np.zeros([len(y_processed),len(np.unique(y_processed))])

        for i,yval in enumerate(y_processed):
        
            y_encoded[i,int(y_processed[i])] = 1

        y = None
        
        reward_mat = None
        
        return X_processed, X_binarized.astype(dtype=np.uint32), y_processed, y_encoded, reward_mat

    '''
    ------------------------------------------------------LOAD STATLOG (SHUTTLE) DATASET-----------------------------------------------------------------------
    '''

    def load_statlog_shuttle_dataset(self):

        dataset=pd.read_csv('datasets/shuttle.trn',delimiter=' ')
        #shuffling the data

        dataset.sample(frac=1).reset_index(drop=True)

        #uncomment to remove the entries with least class (removing class 2,7,6)
        # X_processed = dataset[dataset.iloc[:,-1]!=2]
        # X_processed = dataset[dataset.iloc[:,-1]!=7]
        # X_processed = dataset[dataset.iloc[:,-1]!=6]
        X_processed=dataset
        
        y = X_processed.iloc[:,-1].astype('category').cat.codes
        y = y.to_numpy()
        X_processed = X_processed.to_numpy()

        X_processed = X_processed[:,0:-1]
        

        b = Binarizer(max_bits_per_feature=10)

        b.fit(X_processed)

        X_binarized = b.transform(X_processed)

        y_encoded =np.zeros([len(y),len(np.unique(y))])
      
        for i in range(len(y)):
            y_encoded[i,y[i]]=1
        
        reward_mat = None
       
        return X_processed, X_binarized, y, y_encoded, reward_mat

    '''
    
    ----------------------------------------LOAD MNIST FLAT DATASET---------------------------------------------------------------
    '''

    def load_mnist_flat_dataset(self):
        (X_train,y_train),(X_test,y_test) = mnist.load_data()
       
        X_train_processed_flat = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0)
        X_test_processed_flat = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0)
        X_train_processed_cnn = np.where(X_train >= 75,1,0)
        X_test_processed_cnn = np.where(X_test >= 75,1,0)
        X_processed_cnn = np.concatenate((X_train_processed_cnn,X_test_processed_cnn),axis=0)
        X_binarized_flat = np.concatenate((X_train_processed_flat,X_test_processed_flat),axis=0)
        X_raw = np.concatenate((X_train,X_test),axis=0)
        y = y = np.concatenate((y_train,y_test),axis=0)

        y_encoded = np.zeros([X_binarized_flat.shape[0],len(np.unique(y))])

        X_processed_1 = X_train.reshape(X_train.shape[0],28*28)
        X_processed_2 = X_test.reshape(X_test.shape[0],28*28)
        

        for i in range(X_binarized_flat.shape[0]):
            y_encoded[i,y[i]]=1
        
        reward_mat = None

        X_binarized = X_binarized_flat
        X_processed  = np.concatenate((X_processed_1,X_processed_2),axis=0)#X_raw.reshape(X_raw.shape[0],-1)
    
        return X_processed, X_binarized, y, y_encoded, reward_mat,X_processed_cnn

    '''
    -----------------------------------------LOAD COVERTYPE DATASET---------------------------------------------------------------------

    '''

    def load_covertype_dataset(self):

        dataset = pd.read_csv('datasets/covtype.data',delimiter=',')
        dataset_shuffled = dataset.sample(frac=1).reset_index(drop=True)
        X_vals = dataset_shuffled.iloc[:,0:-1]
        y = dataset_shuffled.iloc[:,-1]


        X_processed = X_vals.to_numpy()
        y = y.to_numpy()

        b = Binarizer(max_bits_per_feature=10)
        b.fit(X_processed)

        X_binarized = b.transform(X_processed)

        y_encoded = np.zeros([len(y),len(np.unique(y))])


        #populating y_encoded 

        for i in range(len(y)):
            y_encoded[i,y[i]-1]=1
        
        reward_mat = None
        return X_processed, X_binarized, y,y_encoded,reward_mat