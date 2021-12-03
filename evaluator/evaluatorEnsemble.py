import sys
import torch
import matplotlib
import numpy as np
import pandas as pd
from scipy import spatial
from datasets import Singular, SingularVGGish
from collections import Counter
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize

# [k, dropout, hidden_dim, output_dim, kernel_size, nhid, embedding_size, experiment]
arg = sys.argv
k = int(arg[1])
# MODEL 1
hidden_dim_1 = int(arg[2])
output_dim_1= int(arg[3])
kernel_size_1 = int(arg[4])
nhid_1 = int(arg[5])
embedding_size_1 = int(arg[6])
dropout_1 = bool(int(arg[7]))
experiment1 = int(arg[8])
# MODEL 2
hidden_dim_2 = int(arg[9])
output_dim_2 = int(arg[10])
kernel_size_2 = int(arg[11])
nhid_2 = int(arg[12])
embedding_size_2 = int(arg[13])
dropout_2 = bool(int(arg[14]))
experiment2 = int(arg[15])
print("---------------------")
print("k =", k)
print("PARAMETERS MODEL 1:")
print("dropout =", dropout_1)
print("hidden_dim =", hidden_dim_1)
print("output_dim =", output_dim_1)
print("kernel_size =", kernel_size_1)
print("nhid =", nhid_1)
print("embedding_size =", embedding_size_1)
print("experiment1 =", experiment1)
print("---------------------")
print("PARAMETERS MODEL 2:")
print("dropout =", dropout_2)
print("hidden_dim =", hidden_dim_2)
print("output_dim =", output_dim_2)
print("kernel_size =", kernel_size_2)
print("nhid =", nhid_2)
print("embedding_size =", embedding_size_2)
print("experiment2 =", experiment2)
print("---------------------")

print("\nCreant data loaders...")

#-------------------------------------------------#
# Llegim el fitxer csv amb el nom dels fitxers    #
# d'entrenament/validacio i el seu label/target.  #
#-------------------------------------------------#
train = pd.read_csv("../dataset/train/TRAIN.csv", header=0)
valid = pd.read_csv("../dataset/validation/VALIDATION.csv", header=0)

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
# MODEL 1
train_dataset_1 = Singular(partition['train'], labels, train=True)
valid_dataset_1 = Singular(partition['validation'], labels, train=False)
# MODEL 2
train_dataset_2 = SingularVGGish(partition['train'], labels, train=True)
valid_dataset_2 = SingularVGGish(partition['validation'], labels, train=False)

#cuda = torch.cuda.is_available()
cuda = False
batch_size = 256
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

# MODEL 1
train_loader_1 = DataLoader(train_dataset_1, batch_size=batch_size, shuffle=True, **kwargs)
valid_loader_1 = DataLoader(valid_dataset_1, batch_size=batch_size, shuffle=False, **kwargs)
# MODEL 2
train_loader_2 = DataLoader(train_dataset_2, batch_size=batch_size, shuffle=True, **kwargs)
valid_loader_2 = DataLoader(valid_dataset_2, batch_size=batch_size, shuffle=False, **kwargs)

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

def plot_embeddings(embeddings, targets, name, experiment, xlim=None, ylim=None):
    """AQUESTA FUNCIÓ S'HA D'ARREGLAR"""
    plt.figure(figsize=(10,10))
    for i,tar in enumerate(classes):
        inds = np.where(targets==tar)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(classes)
    plt.savefig("./experiments/exp-"+str(experiment)+'/'+name+"SegDTW-"+str(experiment)+'.png')

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
                embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target #.numpy()
            k += len(images)
    return embeddings, labels

def compute_accuracy(embeddings_train_1, targets_train_1, embeddings_valid_1,
                     targets_valid_1, embeddings_train_2, targets_train_2,
                     embeddings_valid_2, targets_valid_2, k=100):
    # MODEL 1
    tree_1 = spatial.cKDTree(embeddings_train_1)
    knn_1 = tree_1.query(embeddings_valid_1, k)[1] # Closest indices
    l_1 = [[targets_train_1[i] for i in e] for e in knn_1]
    #print(targets_valid.shape, len(targets_valid))

    # MODEL 2
    tree_2 = spatial.cKDTree(embeddings_train_2)
    knn_2 = tree_2.query(embeddings_valid_2, k)[1] # Closest indices
    l_2 = [[targets_train_2[i] for i in e] for e in knn_2]

    l = [x + y for x, y in zip(l_1, l_2)]
    count, count2 = 0, 0
    for i,e in enumerate(l):
        c = Counter(e)
        label,_ = c.most_common()[0]
        if len(c.most_common())>1:
            label2,_ = c.most_common()[1]
            count2 += (label2 == targets_valid_1[i])
        #print(type(e), e)
        #print(i, label, targets_valid[i], label == targets_valid[i])
        count += (label == targets_valid_1[i])
        count2 += (label == targets_valid_1[i])

    accuracy = count/len(targets_valid_1)*100
    accuracy2 = count2/len(targets_valid_1)*100
    return accuracy, accuracy2

def compute_accuracy_new(embeddings_train_1, targets_train_1, embeddings_valid_1,
                         targets_valid_1, embeddings_train_2, targets_train_2,
                         embeddings_valid_2, targets_valid_2, k=100):
    # MODEL 1
    normed_embeddings_train_1 = normalize(embeddings_train_1, axis=1, norm='l2')
    normed_embeddings_valid_1 = normalize(embeddings_valid_1, axis=1, norm='l2')
    similarity_matrix_1 = np.matmul(normed_embeddings_valid_1, np.transpose(normed_embeddings_train_1)) # valid_size X train_size

    # MODEL 2
    normed_embeddings_train_2 = normalize(embeddings_train_2, axis=1, norm='l2')
    normed_embeddings_valid_2 = normalize(embeddings_valid_2, axis=1, norm='l2')
    similarity_matrix_2 = np.matmul(normed_embeddings_valid_2, np.transpose(normed_embeddings_train_2))

    l_1 = []
    for v_emb_sim in similarity_matrix_1:
        indices = np.argpartition(v_emb_sim, -k)[-k:]
        l_1.append(targets_train_1[indices])

    l_2 = []
    for v_emb_sim in similarity_matrix_2:
        indices = np.argpartition(v_emb_sim, -k)[-k:]
        l_2.append(targets_train_2[indices])

    l = [x + y for x, y in zip(l_1, l_2)]
    count, count2 = 0, 0
    for i,e in enumerate(l):
        c = Counter(e)
        label,_ = c.most_common()[0]
        if len(c.most_common())>1:
            label2,_ = c.most_common()[1]
            count2 += (label2 == targets_valid_1[i])
        count += (label == targets_valid_1[i])
        count2 += (label == targets_valid_1[i])

    accuracy = count/len(targets_valid_1)*100
    accuracy2 = count2/len(targets_valid_1)*100
    return accuracy, accuracy2


###################################################

from networks import EmbeddingNet, EmbeddingNetVGGish, TripletNet

#-------------------------------------------------#
# Llegim el model entrenat i visualitzem els      #
# embeddings.                                     #
#-------------------------------------------------#
print("\nObtenint embeddings dels dos models...")

print("   MODEL 1")
embedding_net_1 = EmbeddingNet(hidden_dim_1, output_dim_1, kernel_size_1, nhid_1, embedding_size_1, dropout_1)
model_1 = TripletNet(embedding_net_1)
model_1.load_state_dict(torch.load("./experiments/exp-"+str(experiment1)+"/model-"+str(experiment1)+".pt", map_location=torch.device('cpu')))

if cuda:
    model_1.cuda()

print("   MODEL 2")
embedding_net_2 = EmbeddingNetVGGish(hidden_dim_2, output_dim_2, kernel_size_2, nhid_2, embedding_size_2, dropout_2)
model_2 = TripletNet(embedding_net_2)
model_2.load_state_dict(torch.load("./experimentsVGGish/exp-"+str(experiment2)+"/modelVGGish-"+str(experiment2)+".pt", map_location=torch.device('cpu')))

if cuda:
    model_2.cuda()

#train_embeddings_tl_2D, train_labels_tl_2D = extract_embeddings(train_loader, model, 2)
#plot_embeddings(train_embeddings_tl_2D, train_labels_tl_2D, "train", experiment1)
#val_embeddings_tl_2D, val_labels_tl_2D = extract_embeddings(valid_loader, model, 2)
#plot_embeddings(val_embeddings_tl_2D, val_labels_tl_2D, "valid", experiment1)

print("Embeddings obtinguts.")

#-------------------------------------------------#
# Calculem l'accuracy de validació.               #
#-------------------------------------------------#

print("\nCalculant l'accuracy de validacio...")

print("   MODEL 1")
train_embeddings_1, train_labels_1 = extract_embeddings(train_loader_1, model_1, embedding_size_1)
val_embeddings_1, val_labels_1 = extract_embeddings(valid_loader_1, model_1, embedding_size_1)

print("   MODEL 2")
train_embeddings_2, train_labels_2 = extract_embeddings(train_loader_2, model_2, embedding_size_2)
val_embeddings_2, val_labels_2 = extract_embeddings(valid_loader_2, model_2, embedding_size_2)

accuracy,accuracy2 = compute_accuracy(train_embeddings_1, train_labels_1,
                                      val_embeddings_1, val_labels_1,
                                      train_embeddings_2, train_labels_2,
                                      val_embeddings_2, val_labels_2, k)
print("ACCURACY = ", accuracy, "%")
print("ACCURACY 2 primers = ", accuracy2, "%")

print("--- NEW VERSIONS ---")
accuracy,accuracy2 = compute_accuracy_new(train_embeddings_1, train_labels_1,
                                      val_embeddings_1, val_labels_1,
                                      train_embeddings_2, train_labels_2,
                                      val_embeddings_2, val_labels_2, k)
print("ACCURACY = ", accuracy, "%")
print("ACCURACY 2 primers = ", accuracy2, "%")

print("Accuracy de validacio calculada.")
