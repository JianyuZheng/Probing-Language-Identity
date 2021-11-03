import torch
import torch.nn.functional as F     
import torch.utils.data as Data
import numpy as np
import pickle
import os
import time

start = time.time()
torch.manual_seed(1) 



files = os.listdir('./bert_embeddings')
languages = [f.strip().split('_')[0] for f in files]

with open('features2num_WALS.dat', 'rb') as fp:
    features2num_WALS = pickle.load(fp) 
features = list(features2num_WALS.keys())
feature = features[56]

with open('WALS_INFO_dict.dat', 'rb') as fp:
    WALS_INFO = pickle.load(fp) 


with open('./bert_embeddings/English_embedding.dat', 'rb') as fp:
    x_train = torch.tensor(pickle.load(fp)) 
    key = WALS_INFO['English'][feature]
    y_train = torch.ones(10000) *features2num_WALS[feature][key]
for l in languages:
    if l not in ['English','Chinese'] and feature in list(WALS_INFO[l]):
        filename = './bert_embeddings/'+l+'_embedding.dat'    
        with open(filename, 'rb') as fp:
            x0 = torch.tensor(pickle.load(fp)) 
            key = WALS_INFO[l][feature]
            y0 = torch.ones(10000) *features2num_WALS[feature][key]
        x_train = torch.cat((x_train, x0), 0)  
        y_train  = torch.cat((y_train, y0), )
x_train = x_train.type(torch.FloatTensor)
y_train = y_train.type(torch.LongTensor)     

'''
n_data = torch.ones(100, 2)         
x0 = torch.normal(2*n_data, 1)      # type0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # type0 y data (tensor), shape=(100, )
x1 = torch.normal(-2*n_data, 1)     # type1 x data (tensor), shape=(100, 1)
y1 = torch.ones(100)                # type1 y data (tensor), shape=(100, )
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)  

class Net(torch.nn.Module):   
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()    
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   
        self.out = torch.nn.Linear(n_hidden, n_output)      

    def forward(self, x):
        x = F.relu(self.hidden(x))      
        x = self.out(x)                
        return x
'''   

input_size = 768
num_hidden = 100
output_size =  len(features2num_WALS[feature].keys())
net = torch.nn.Sequential(
    torch.nn.Linear(input_size, num_hidden),
    torch.nn.Dropout(0.5),  
    torch.nn.ReLU(),
    torch.nn.Linear(num_hidden, output_size),
)
#print(net)



optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_func = torch.nn.CrossEntropyLoss()

torch_dataset = Data.TensorDataset(x_train, y_train)



BATCH_SIZE = 512
loader = Data.DataLoader(
    dataset=torch_dataset,     
    batch_size=BATCH_SIZE,      
    shuffle=True,               
    num_workers=0,              
)
for epoch in range(2):   
    for step, (batch_x, batch_y) in enumerate(loader):  
        #print(epoch,"\t", step)
        optimizer.zero_grad() 
        out = net(batch_x) 
        loss = loss_func(out, batch_y) 
        loss.backward()  
        optimizer.step()
    #print(np.sum(loss.detach().numpy()))
#torch.save(net, 'net.pkl')  


with open('./bert_embeddings/Chinese_embedding.dat', 'rb') as fp:
    x_test = torch.tensor(pickle.load(fp)) 
    key = WALS_INFO['Chinese'][feature]
    y_test = np.ones(10000)*features2num_WALS[feature][key]

net.eval()
out = net(x_test)
prediction = torch.max(F.softmax(out), 1)[1]
pred_y = prediction.data.numpy().squeeze()

print("accuracy:", np.sum(pred_y==y_test)/10000)















