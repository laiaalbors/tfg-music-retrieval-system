import sys
import torch
import pandas as pd
from datasets import TripletSegVGGishDTW
from torch.utils.data import DataLoader

# [batch_size, margin, lr, n_epochs, dropout, hidden_dim, output_dim,
#  kernel_size, nhid, embedding_size, experiment]
arg = sys.argv
batch_size = int(arg[1])
margin = float(arg[2])
lr = float(arg[3])
n_epochs = int(arg[4])
dropout = bool(int(arg[5]))
hidden_dim = int(arg[6])
output_dim = int(arg[7])
kernel_size = int(arg[8])
nhid = int(arg[9])
embedding_size = int(arg[10])
experiment = int(arg[11])
print("----------------")
print("PARAMETERS:")
print("batch_size =", batch_size)
print("margin =", margin)
print("lr =", lr)
print("n_epochs =", n_epochs)
print("dropout =", dropout)
print("hidden_dim =", hidden_dim)
print("output_dim =", output_dim)
print("kernel_size =", kernel_size)
print("nhid =", nhid)
print("embedding_size =", embedding_size)
print("experiment =", experiment)
print("----------------")

print("\n##################################\n# TRAINING PER SEGMENT VGGish DTW #\n##################################\")

print("\nCreant data loaders...")

#-------------------------------------------------#
# Llegim el fitxer csv amb el nom dels fitxers    #
# d'entrenament/validacio i el seu label/target.  #
#-------------------------------------------------#
train = pd.read_csv("../vggish/train/TRAIN.csv", header=0)
valid = pd.read_csv("../vggish/validation/VALIDATION.csv", header=0)

#-------------------------------------------------#
# Convertim aquesta informacio en diccionaris.    #
#-------------------------------------------------#
partition = {'train': list(train['name']), 'validation': list(valid['name'])}
labels_train = pd.Series(list(train['label']), index=list(train['name'])).to_dict()
labels_valid = pd.Series(valid.label.values, index=valid.name).to_dict()
labels = {};
for d in (labels_train, labels_valid):
    labels.update(d)

classes = ['NMN_BM', 'OTR_JG', 'LIT_TB', 'LMT_EP', 'SHA_LS', 'MTW_EX', 'CDA_NP', 'VAE_CS', 'OLD_TB', 'SBM_BK']
path = "./experimentsVGGish/exp-" + str(experiment) + "/"

alignment = {}
for classe in classes:
    alignment[classe] = pd.read_csv(path+classe+".csv")

#-------------------------------------------------#
# Creem els triplets d'entrenament i validacio.   #
#-------------------------------------------------#
triplet_train_dataset = TripletSegVGGishDTW(partition['train'], labels, alignment, train, train=True) # Returns triplets of arrays
triplet_valid_dataset = TripletSegVGGishDTW(partition['validation'], labels, alignment, valid, train=False)

# Parametres d'entrenament
cuda = torch.cuda.is_available()
print("\n!!torch.cuda.is_available()!!", cuda, "\n")
kwargs = {'num_workers': 2, 'pin_memory': True} if cuda else {}

triplet_train_loader = DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
triplet_valid_loader = DataLoader(triplet_valid_dataset, batch_size=batch_size, shuffle=False, **kwargs)

print("Data loaders creats.")

###################################################

import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler

from trainer import fit
from losses import TripletLoss
from networks import EmbeddingNetVGGish, TripletNet

from os import path as pth


print("\nEntrenant el model...")

# Parametres de la xarxa i d'entrenament
embedding_net = EmbeddingNetVGGish(hidden_dim, output_dim, kernel_size, nhid, embedding_size, dropout)
model = TripletNet(embedding_net)

if pth.isfile("./experimentsVGGish/exp-"+str(experiment)+"/modelSegVGGish-"+str(experiment)+".pt"):
    model.load_state_dict(torch.load("./experimentsVGGish/exp-"+str(experiment)+"/modelSegVGGish-"+str(experiment)+".pt"))
else:
    print("ERROR: El model per segment DTW's version no està entrenat!")

if cuda:
    model.cuda()
loss_fn = TripletLoss(margin)

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
log_interval = 100

#-------------------------------------------------#
# Entrenem el model amb els triplets creats.      #
#-------------------------------------------------#
fit(triplet_train_loader, triplet_valid_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

print("Model entrenat.")

#-------------------------------------------------#
# Guardem el model entrenat.                      #
#-------------------------------------------------#
torch.save(model.state_dict(), "./experimentsVGGish/exp-"+str(experiment)+"/modelSegVGGishDTW-"+str(experiment)+".pt")
print("Model guardat.")
