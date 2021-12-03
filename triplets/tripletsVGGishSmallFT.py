import sys
import torch
import pandas as pd
from datasets import TripletVGGishMeanFT
from torch.utils.data import DataLoader

# [batch_size, margin, lr, n_epochs, dropout, hidden_dim, output_dim,
#  kernel_size, nhid, embedding_size, experiment]
arg = sys.argv
batch_size = int(arg[1])
margin = float(arg[2])
lr = float(arg[3])
n_epochs1 = int(arg[4])
n_epochs2 = int(arg[5])
dropout = bool(int(arg[6]))
output_dim = int(arg[7])
kernel_size = int(arg[8])
embedding_size = int(arg[9])
experiment = int(arg[10])
print("----------------")
print("PARAMETERS:")
print("batch_size =", batch_size)
print("margin =", margin)
print("lr =", lr)
print("n_epochs1 =", n_epochs1)
print("n_epochs2 =", n_epochs2)
print("dropout =", dropout)
print("output_dim =", output_dim)
print("kernel_size =", kernel_size)
print("embedding_size =", embedding_size)
print("experiment =", experiment)
print("----------------")

print("\n####################\n# TRAINING PER SONG #\n####################")

print("\nCreant data loaders...")

#-------------------------------------------------#
# Llegim el fitxer csv amb el nom dels fitxers    #
# d'entrenament/validacio i el seu label/target.  #
#-------------------------------------------------#
train = pd.read_csv("../datasetWavfiles/train/TRAIN.csv", header=0)
valid = pd.read_csv("../datasetWavfiles/validation/VALIDATION.csv", header=0)

#-------------------------------------------------#
# Convertim aquesta informacio en diccionaris.    #
#-------------------------------------------------#
partition = {'train': list(train['name']), 'validation': list(valid['name'])}
labels_train = pd.Series(list(train['label']), index=list(train['name'])).to_dict()
labels_valid = pd.Series(valid.label.values, index=valid.name).to_dict()
labels = {};
for d in (labels_train, labels_valid):
    labels.update(d)

#-------------------------------------------------#
# Creem els triplets d'entrenament i validacio.   #
#-------------------------------------------------#
triplet_train_dataset = TripletVGGishMeanFT(partition['train'], labels, train=True) # Returns triplets of arrays
triplet_valid_dataset = TripletVGGishMeanFT(partition['validation'], labels, train=False)

# Parametres d'entrenament
cuda = torch.cuda.is_available()
#batch_size = int(arg[1]) #128
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
from networks import EmbeddingNetVGGishSmallFT, TripletNet

from os import path as pth


print("\nEntrenant el model...")

#---------------------------------------------------------------#
# Comencem només entrenaant la nova capai deixem els paràmetres #
# del model congelats.                                          #
#---------------------------------------------------------------#

pretrained_model = torch.hub.load('harritaylor/torchvggish', 'vggish')
if cuda:
    pretrained_model.cuda()
feature_extractor = pretrained_model.features

for layer in feature_extractor:
    for param in layer.parameters():
        param.requires_grad = False

embedding_extractor = pretrained_model.embeddings

for layer in embedding_extractor:
    for param in layer.parameters():
        param.requires_grad = False

embedding_net = EmbeddingNetVGGishSmallFT(feature_extractor, embedding_extractor, output_dim, kernel_size, embedding_size, dropout)
model = TripletNet(embedding_net)

improve = False
#if pth.isfile("./exp-"+str(experiment)+"/modelVGGishFT-"+str(experiment)+".pt"):
#    model.load_state_dict(torch.load("./exp-"+str(experiment)+"/modelVGGishFT-"+str(experiment)+".pt"))
#    improve = True

if cuda:
    model.cuda()

loss_fn = TripletLoss(margin)
if cuda:
    loss_fn.cuda()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
log_interval = 100

#-------------------------------------------------#
# Entrenem el model amb els triplets creats.      #
#-------------------------------------------------#
#model.load_state_dict(torch.load("./exp-"+str(experiment)+"/modelVGGishFT-"+str(experiment)+".pt"))
fit(triplet_train_loader, triplet_valid_loader, model, loss_fn, optimizer, scheduler, n_epochs1, cuda, log_interval)
torch.save(model.state_dict(), "./experimentsVGGishSmallFT/exp-"+str(experiment)+"/modelVGGishSmallFT-"+str(experiment)+".pt")

print("\nPrimera fase entrenada.\n")

#------------------------#
# Ara fem el fine-tuning #
#------------------------#

# Decidim quines capes fine-tune i quines congelar
feature_extractor = embedding_net.feature_extractor

for layer in feature_extractor[:13]:  # Freeze layers 0 to 12
    for param in layer.parameters():
        param.requires_grad = False

for layer in feature_extractor[13:]:  # Train layers 13 to 15
    for param in layer.parameters():
        param.requires_grad = True

embedding_extractor = embedding_net.embedding_extractor

for layer in embedding_extractor:  # Train all layers (0 a 5)
    for param in layer.parameters():
        param.requires_grad = True

#if cuda:
#    model.cuda()
#loss_fn = TripletLoss(margin)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
log_interval = 100

fit(triplet_train_loader, triplet_valid_loader, model, loss_fn, optimizer, scheduler, n_epochs2, cuda, log_interval)


print("Model entrenat.")

#-------------------------------------------------#
# Guardem el model entrenat.                      #
#-------------------------------------------------#
if improve:
    torch.save(model.state_dict(), "./experimentsVGGishSmallFT/exp-"+str(experiment)+"/modelVGGishSmallFT2-"+str(experiment)+".pt")
else:
    torch.save(model.state_dict(), "./experimentsVGGishSmallFT/exp-"+str(experiment)+"/modelVGGishSmallFT-"+str(experiment)+".pt")
print("Model guardat.")
