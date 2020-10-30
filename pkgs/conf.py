

training_set = ["201_ChEMBL26.txt", "202_PubChem_NCATS.txt", "203_PubChem_JHICC.txt", "207_Cai.txt"]
client_list =  ['a', 'b', 'c', 'd']
# workers =  ['a', 'b', 'c', 'd','e', 'f', 'g']
batch_size = 200
init_list = ['normal', 'uniform', 'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'none']
seed_list = [0, 1, 2, 6, 8, 9]
version = 'v0.1'
init_list = ["none"]
seed_list = [0]
aggregate_epochs = [1, 5, 10, 20, 50, 100]

#1:a,b,c,d client
#   v1.1:a,b(same part as d),c d client
#   v1.3:personalization
#   v1.4:change sns style "ticks"
#2:data num federated and traditional
#3:shared data
#4:kinase
#   v4.1:kinase with embedding
#   v4.2:train multi-kinase with embedding and test only AKT1
#   v4.3:train multi-kinase with embedding and test only INSR
#   v4.4:train and val data with fixed random seed
#   v4.5:kinase without embedding only AKT1
#5:hERG
#   v5.1 cross entropy weight loss and f1_score metric
#6:disynthon
#7:test bayes
#   v7.1:add Logger and federate and fix bug of optim in traditional learning
#   v7.2:change prior when aggragate model
#   v7.4:
#   v7.5 batch400 change prior fc1 change scale
#   v7.7:rrelu_
#   v7.7:small layer and relu

#TODO: BN bayes, 负梯度
