import sys
import torch
import matplotlib
import numpy as np
import pandas as pd
from scipy import spatial
from datasets import SingularVGGishMeanFT
from collections import Counter
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize

# [k, dropout, hidden_dim, output_dim, kernel_size, nhid, embedding_size, experiment]
arg = sys.argv
k = int(arg[1])
experiment = int(arg[2])
improve = bool(int(arg[3]))
print("----------------")
print("PARAMETERS:")
print("k =", k)
print("experiment =", experiment)
print("improve =", improve)
print("----------------")

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
# Creem els datasets i els dataloaders per        #
# visualitzar els embeddings.                     #
#-------------------------------------------------#
train_dataset = SingularVGGishMeanFT(partition['train'], labels, train=True)
valid_dataset = SingularVGGishMeanFT(partition['validation'], labels, train=False)

#cuda = torch.cuda.is_available()
cuda = False
batch_size = 256
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, **kwargs)

print("Data loaders creats.")

###################################################

#-------------------------------------------------#
# Funcions per obtenir els embeddings i           #
# visualitzar-los.                                #
#-------------------------------------------------#
classes = ['NMN_BM', 'OTR_JG', 'LIT_TB', 'LMT_EP', 'SHA_LS', 'MTW_EX', 'CDA_NP', 'VAE_CS', 'OLD_TB', 'SBM_BK']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

def plot_embeddings(embeddings, targets, name, xlim=None, ylim=None):
    plt.figure(figsize=(10,10))
    for i,tar in enumerate(classes):
        inds = np.where(targets==tar)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(classes)
    if improve:
        plt.savefig("./exp-"+str(experiment)+'/'+name+"VGGishFT2-"+str(experiment)+'.png')
    else:
        plt.savefig("./exp-"+str(experiment)+'/'+name+"VGGishFT-"+str(experiment)+'.png')

def extract_embeddings(dataloader, model, size):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), size))
        labels = np.empty(len(dataloader.dataset),dtype=object)
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            if size == 2:
                embeddings[k:k+len(images)] = model.get_embedding2D(images).data.cpu().numpy()
            else:
                embeddings[k:k+len(images)] = model.get_embeddingFT(images).data.cpu().numpy()
            labels[k:k+len(images)] = target #.numpy()
            k += len(images)
    return embeddings, labels

def compute_accuracy(embeddings_train, targets_train, embeddings_valid, targets_valid, k=10):
    tree = spatial.cKDTree(embeddings_train)
    knn = tree.query(embeddings_valid, k)[1] # Closest indices
    l = [[targets_train[i] for i in e] for e in knn]
    #print(targets_valid.shape, len(targets_valid))

    count, count2 = 0, 0
    for i,e in enumerate(l):
        c = Counter(e)
        label,_ = c.most_common()[0]
        if len(c.most_common())>1:
            label2,_ = c.most_common()[1]
            count2 += (label2 == targets_valid[i])
        #print(type(e), e)
        #print(i, label, targets_valid[i], label == targets_valid[i])
        count += (label == targets_valid[i])
        count2 += (label == targets_valid[i])

    accuracy = count/len(targets_valid)*100
    accuracy2 = count2/len(targets_valid)*100
    return accuracy, accuracy2

def compute_accuracy_new(embeddings_train, targets_train, embeddings_valid, targets_valid, k=10):
    normed_embeddings_train = normalize(embeddings_train, axis=1, norm='l2')
    normed_embeddings_valid = normalize(embeddings_valid, axis=1, norm='l2')
    similarity_matrix = np.matmul(normed_embeddings_valid, np.transpose(normed_embeddings_train)) # valid_size X train_size

    l = []
    for v_emb_sim in similarity_matrix:
        indices = np.argpartition(v_emb_sim, -k)[-k:]
        l.append(targets_train[indices])

    count, count2 = 0, 0
    for i,e in enumerate(l):
        c = Counter(e)
        label,_ = c.most_common()[0]
        if len(c.most_common())>1:
            label2,_ = c.most_common()[1]
            count2 += (label2 == targets_valid[i])
        count += (label == targets_valid[i])
        count2 += (label == targets_valid[i])

    accuracy = count/len(targets_valid)*100
    accuracy2 = count2/len(targets_valid)*100
    return accuracy, accuracy2


###################################################

from networks import EmbeddingNetVGGishMeanFT, TripletNet

#-------------------------------------------------#
# Llegim el model entrenat i visualitzem els      #
# embeddings.                                     #
#-------------------------------------------------#
print("\nObtenint el model...")

#device = torch.device("cuda")
pretrained_model = torch.hub.load('harritaylor/torchvggish', 'vggish')
#pretrained_model.to(device)
feature_extractor = pretrained_model.features
embedding_extractor = pretrained_model.embeddings

embedding_net = EmbeddingNetVGGishMeanFT(feature_extractor, embedding_extractor)
model = TripletNet(embedding_net)
if improve:
    print("   Taking the improved model.")
    model.load_state_dict(torch.load("./exp-"+str(experiment)+"/modelVGGishFT2-"+str(experiment)+".pt", map_location=torch.device('cpu')))
else:
    model.load_state_dict(torch.load("./exp-"+str(experiment)+"/modelVGGishFT-"+str(experiment)+".pt", map_location=torch.device('cpu')))

if cuda:
    print("    Enviem el model a GPU.")
    model.cuda()
else:
    model.cpu()
#model.eval()

#train_embeddings_tl_2D, train_labels_tl_2D = extract_embeddings(train_loader, model, 2)
#plot_embeddings(train_embeddings_tl_2D, train_labels_tl_2D, "train")
#val_embeddings_tl_2D, val_labels_tl_2D = extract_embeddings(valid_loader, model, 2)
#plot_embeddings(val_embeddings_tl_2D, val_labels_tl_2D, "valid")

print("Model obtingut.")

#-------------------------------------------------#
# Calculem l'accuracy de validaci??.               #
#-------------------------------------------------#

print("\nCalculant l'accuracy de validacio...")
embedding_size = 128

train_embeddings_tl, train_labels_tl = extract_embeddings(train_loader, model, embedding_size)
print("   First done.")
val_embeddings_tl, val_labels_tl = extract_embeddings(valid_loader, model, embedding_size)
print("   Second done.")

accuracy,accuracy2 = compute_accuracy(train_embeddings_tl, train_labels_tl, val_embeddings_tl, val_labels_tl, k)
print("ACCURACY = ", accuracy, "%")
print("ACCURACY 2 primers = ", accuracy2, "%")

print("--- NEW VERSIONS ---")
accuracy,accuracy2 = compute_accuracy_new(train_embeddings_tl, train_labels_tl, val_embeddings_tl, val_labels_tl, k)
print("ACCURACY = ", accuracy, "%")
print("ACCURACY 2 primers = ", accuracy2, "%")

print("Accuracy de validacio calculada.")
