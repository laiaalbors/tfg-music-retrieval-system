import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from vggish_input import wavfile_to_examples

##############
# SONG LEVEL #
##############

#---------#
# MELODIA #
#---------#
class Triplet(Dataset):

  def __init__(self, list_IDs, labels, train):
    'Initialization'
    self.labels = labels     # diccionari {id: label}
    self.list_IDs = list_IDs # llista [id1,id2...]
    self.train = train

    if self.train:
      self.train_labels = {id: self.labels[id] for id in self.list_IDs} # diccionari {id: label}
      self.labels_set = set(self.train_labels.values())                 # set (label1,label2...)
      self.label_to_indices = {label: [self.list_IDs.index(k) for k,v in self.train_labels.items() if v == label]
                               for label in self.labels_set}            # diccionari {label: [idx1,idx2...]}
    else:
      self.valid_labels = {id: self.labels[id] for id in self.list_IDs} # diccionari {id: label}
      # generate fixed triplets for validation
      self.labels_set = set(self.valid_labels.values())                 # set (label1,label2...)
      self.label_to_indices = {label: [self.list_IDs.index(k) for k,v in self.valid_labels.items() if v == label]
                               for label in self.labels_set}            # diccionari {label: [idx1,idx2...]}

      random_state = np.random.RandomState(29)

      triplets = [[i,
                    random_state.choice(self.label_to_indices[self.valid_labels[self.list_IDs[i]]]),
                    random_state.choice(self.label_to_indices[
                                            np.random.choice(
                                                list(self.labels_set - set([self.valid_labels[self.list_IDs[i]]]))
                                            )
                                        ])
                    ]
                  for i in range(len(self.list_IDs))]
      self.valid_triplets = triplets

  def __len__(self):
    'Denotes the total number of samples'
    return len(self.list_IDs)

  def __getitem__(self, index):
    'Generates one sample of data'
    # Select sample
    if self.train:
      ref_id = self.list_IDs[index]
      label_ref = self.train_labels[ref_id]
      positive_index = index
      while positive_index == index:
          positive_index = np.random.choice(self.label_to_indices[label_ref])
      label_neg = np.random.choice(list(self.labels_set - set([label_ref])))
      negative_index = np.random.choice(self.label_to_indices[label_neg])
      pos_id = self.list_IDs[positive_index]
      neg_id = self.list_IDs[negative_index]
      path = "train/"
    else:
      ref_id = self.list_IDs[self.valid_triplets[index][0]]
      pos_id = self.list_IDs[self.valid_triplets[index][1]]
      neg_id = self.list_IDs[self.valid_triplets[index][2]]
      path = "validation/"

    ID = self.list_IDs[index]

    # Load data
    ref = pd.read_csv("../datasetBig/"+ path + str(ref_id) + ".csv", names=("time","value"))
    Ref = torch.tensor([np.array(ref['value'])], dtype=torch.float32)
    pos = pd.read_csv("../datasetBig/"+ path + str(pos_id) + ".csv", names=("time","value"))
    Pos = torch.tensor([np.array(pos['value'])], dtype=torch.float32)
    neg = pd.read_csv("../datasetBig/"+ path + str(neg_id) + ".csv", names=("time","value"))
    Neg = torch.tensor([np.array(neg['value'])], dtype=torch.float32)

    return (Ref, Pos, Neg), []

class Singular(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels, train):
        'Initialization'
        self.labels = labels     # diccionari {id: label}
        self.list_IDs = list_IDs # llista [id1,id2...]
        self.train = train       # True or False

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        if self.train:
          path = "train/"
        else:
          path = "validation/"

        # Load data and get label
        x = pd.read_csv("../datasetBig/"+ path + str(ID) + ".csv", names=("time","value"))
        X = torch.tensor([np.array(x['value'])], dtype=torch.float32)
        y = self.labels[ID]

        return X, y

#-------#
# PITCH #
#-------#
class TripletPitch(Dataset):

  def __init__(self, list_IDs, labels, train):
    'Initialization'
    self.labels = labels     # diccionari {id: label}
    self.list_IDs = list_IDs # llista [id1,id2...]
    self.train = train

    if self.train:
      self.train_labels = {id: self.labels[id] for id in self.list_IDs} # diccionari {id: label}
      self.labels_set = set(self.train_labels.values())                 # set (label1,label2...)
      self.label_to_indices = {label: [self.list_IDs.index(k) for k,v in self.train_labels.items() if v == label]
                               for label in self.labels_set}            # diccionari {label: [idx1,idx2...]}
    else:
      self.valid_labels = {id: self.labels[id] for id in self.list_IDs} # diccionari {id: label}
      # generate fixed triplets for validation
      self.labels_set = set(self.valid_labels.values())                 # set (label1,label2...)
      self.label_to_indices = {label: [self.list_IDs.index(k) for k,v in self.valid_labels.items() if v == label]
                               for label in self.labels_set}            # diccionari {label: [idx1,idx2...]}

      random_state = np.random.RandomState(29)

      triplets = [[i,
                    random_state.choice(self.label_to_indices[self.valid_labels[self.list_IDs[i]]]),
                    random_state.choice(self.label_to_indices[
                                            np.random.choice(
                                                list(self.labels_set - set([self.valid_labels[self.list_IDs[i]]]))
                                            )
                                        ])
                    ]
                  for i in range(len(self.list_IDs))]
      self.valid_triplets = triplets

  def __len__(self):
    'Denotes the total number of samples'
    return len(self.list_IDs)

  def __getitem__(self, index):
    'Generates one sample of data'
    # Select sample
    if self.train:
      ref_id = self.list_IDs[index]
      label_ref = self.train_labels[ref_id]
      positive_index = index
      while positive_index == index:
          positive_index = np.random.choice(self.label_to_indices[label_ref])
      label_neg = np.random.choice(list(self.labels_set - set([label_ref])))
      negative_index = np.random.choice(self.label_to_indices[label_neg])
      pos_id = self.list_IDs[positive_index]
      neg_id = self.list_IDs[negative_index]
      path = "train/"
    else:
      ref_id = self.list_IDs[self.valid_triplets[index][0]]
      pos_id = self.list_IDs[self.valid_triplets[index][1]]
      neg_id = self.list_IDs[self.valid_triplets[index][2]]
      path = "validation/"

    ID = self.list_IDs[index]

    # Load data
    ref = pd.read_csv("/home/usuaris/veu/laia.albors/datasetPitches/"+ path + str(ref_id) + ".csv", names=["value"])
    Ref = torch.tensor([np.array(ref['value'])], dtype=torch.float32)
    pos = pd.read_csv("/home/usuaris/veu/laia.albors/datasetPitches/"+ path + str(pos_id) + ".csv", names=["value"])
    Pos = torch.tensor([np.array(pos['value'])], dtype=torch.float32)
    neg = pd.read_csv("/home/usuaris/veu/laia.albors/datasetPitches/"+ path + str(neg_id) + ".csv", names=["value"])
    Neg = torch.tensor([np.array(neg['value'])], dtype=torch.float32)

    return (Ref, Pos, Neg), []

class SingularPitch(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels, train):
        'Initialization'
        self.labels = labels     # diccionari {id: label}
        self.list_IDs = list_IDs # llista [id1,id2...]
        self.train = train       # True or False

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        if self.train:
          path = "train/"
        else:
          path = "validation/"

        # Load data and get label
        x = pd.read_csv("/home/usuaris/veu/laia.albors/datasetPitches/"+ path + str(ID) + ".csv", names=["value"])
        X = torch.tensor([np.array(x['value'])], dtype=torch.float32)
        y = self.labels[ID]

        return X, y

#--------#
# VGGish #
#--------#
class TripletVGGish(Dataset):

  def __init__(self, list_IDs, labels, train):
    'Initialization'
    self.labels = labels     # diccionari {id: label}
    self.list_IDs = list_IDs # llista [id1,id2...]
    self.train = train

    if self.train:
      self.train_labels = {id: self.labels[id] for id in self.list_IDs} # diccionari {id: label}
      self.labels_set = set(self.train_labels.values())                 # set (label1,label2...)
      self.label_to_indices = {label: [self.list_IDs.index(k) for k,v in self.train_labels.items() if v == label]
                               for label in self.labels_set}            # diccionari {label: [idx1,idx2...]}
    else:
      self.valid_labels = {id: self.labels[id] for id in self.list_IDs} # diccionari {id: label}
      # generate fixed triplets for validation
      self.labels_set = set(self.valid_labels.values())                 # set (label1,label2...)
      self.label_to_indices = {label: [self.list_IDs.index(k) for k,v in self.valid_labels.items() if v == label]
                               for label in self.labels_set}            # diccionari {label: [idx1,idx2...]}

      random_state = np.random.RandomState(29)

      triplets = [[i,
                    random_state.choice(self.label_to_indices[self.valid_labels[self.list_IDs[i]]]),
                    random_state.choice(self.label_to_indices[
                                            np.random.choice(
                                                list(self.labels_set - set([self.valid_labels[self.list_IDs[i]]]))
                                            )
                                        ])
                    ]
                  for i in range(len(self.list_IDs))]
      self.valid_triplets = triplets

  def __len__(self):
    'Denotes the total number of samples'
    return len(self.list_IDs)

  def __getitem__(self, index):
    'Generates one sample of data'
    # Select sample
    if self.train:
      ref_id = self.list_IDs[index]
      label_ref = self.train_labels[ref_id]
      positive_index = index
      while positive_index == index:
          positive_index = np.random.choice(self.label_to_indices[label_ref])
      label_neg = np.random.choice(list(self.labels_set - set([label_ref])))
      negative_index = np.random.choice(self.label_to_indices[label_neg])
      pos_id = self.list_IDs[positive_index]
      neg_id = self.list_IDs[negative_index]
      path = "train/"
    else:
      ref_id = self.list_IDs[self.valid_triplets[index][0]]
      pos_id = self.list_IDs[self.valid_triplets[index][1]]
      neg_id = self.list_IDs[self.valid_triplets[index][2]]
      path = "validation/"

    # Load data
    f_ref = open("/home/usuaris/veu/laia.albors/vggishBig/"+ path + str(ref_id) + ".npy", 'rb')
    ref = np.load(f_ref, allow_pickle=True); Ref = torch.tensor([ref], dtype=torch.float32)
    f_pos = open("/home/usuaris/veu/laia.albors/vggishBig/"+ path + str(pos_id) + ".npy", 'rb')
    pos = np.load(f_pos, allow_pickle=True); Pos = torch.tensor([pos], dtype=torch.float32)
    f_neg = open("/home/usuaris/veu/laia.albors/vggishBig/"+ path + str(neg_id) + ".npy", 'rb')
    neg = np.load(f_neg, allow_pickle=True); Neg = torch.tensor([neg], dtype=torch.float32)

    return (Ref, Pos, Neg), []

class SingularVGGish(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels, train):
        'Initialization'
        self.labels = labels     # diccionari {id: label}
        self.list_IDs = list_IDs # llista [id1,id2...]
        self.train = train       # True or False

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        if self.train:
          path = "train/"
        else:
          path = "validation/"

        # Load data and get label
        f = open("/home/usuaris/veu/laia.albors/vggishBig/"+ path + str(ID) + ".npy", 'rb')
        x = np.load(f, allow_pickle=True); X = torch.tensor([x], dtype=torch.float32)
        y = self.labels[ID]

        return X, y

class SingularVGGish2(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels, train):
        'Initialization'
        self.labels = labels     # diccionari {id: label}
        self.list_IDs = list_IDs # llista [id1,id2...]
        self.train = train       # True or False

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        if self.train:
          path = "train/"
        else:
          path = "validation/"

        # Load data and get label
        f = open("/home/usuaris/veu/laia.albors/vggish/"+ path + str(ID) + ".npy", 'rb')
        x = np.load(f); X = torch.tensor([x], dtype=torch.float32)
        y = self.labels[ID]

        return X, y


#-------------#
# VGGish Mean #
#-------------#
class TripletVGGishMean(Dataset):

  def __init__(self, list_IDs, labels, train):
    'Initialization'
    self.labels = labels     # diccionari {id: label}
    self.list_IDs = list_IDs # llista [id1,id2...]
    self.train = train

    if self.train:
      self.train_labels = {id: self.labels[id] for id in self.list_IDs} # diccionari {id: label}
      self.labels_set = set(self.train_labels.values())                 # set (label1,label2...)
      self.label_to_indices = {label: [self.list_IDs.index(k) for k,v in self.train_labels.items() if v == label]
                               for label in self.labels_set}            # diccionari {label: [idx1,idx2...]}
    else:
      self.valid_labels = {id: self.labels[id] for id in self.list_IDs} # diccionari {id: label}
      # generate fixed triplets for validation
      self.labels_set = set(self.valid_labels.values())                 # set (label1,label2...)
      self.label_to_indices = {label: [self.list_IDs.index(k) for k,v in self.valid_labels.items() if v == label]
                               for label in self.labels_set}            # diccionari {label: [idx1,idx2...]}

      random_state = np.random.RandomState(29)

      triplets = [[i,
                    random_state.choice(self.label_to_indices[self.valid_labels[self.list_IDs[i]]]),
                    random_state.choice(self.label_to_indices[
                                            np.random.choice(
                                                list(self.labels_set - set([self.valid_labels[self.list_IDs[i]]]))
                                            )
                                        ])
                    ]
                  for i in range(len(self.list_IDs))]
      self.valid_triplets = triplets

  def __len__(self):
    'Denotes the total number of samples'
    return len(self.list_IDs)

  def __getitem__(self, index):
    'Generates one sample of data'
    # Select sample
    if self.train:
      ref_id = self.list_IDs[index]
      label_ref = self.train_labels[ref_id]
      positive_index = index
      while positive_index == index:
          positive_index = np.random.choice(self.label_to_indices[label_ref])
      label_neg = np.random.choice(list(self.labels_set - set([label_ref])))
      negative_index = np.random.choice(self.label_to_indices[label_neg])
      pos_id = self.list_IDs[positive_index]
      neg_id = self.list_IDs[negative_index]
      path = "train/"
    else:
      ref_id = self.list_IDs[self.valid_triplets[index][0]]
      pos_id = self.list_IDs[self.valid_triplets[index][1]]
      neg_id = self.list_IDs[self.valid_triplets[index][2]]
      path = "validation/"

    # Load data
    f_ref = open("/home/usuaris/veu/laia.albors/vggish/"+ path + str(ref_id) + ".npy", 'rb')
    ref = np.load(f_ref); Ref = torch.tensor([ref], dtype=torch.float32)
    Ref = torch.mean(Ref, 1); Ref = Ref.view(128)
    f_pos = open("/home/usuaris/veu/laia.albors/vggish/"+ path + str(pos_id) + ".npy", 'rb')
    pos = np.load(f_pos); Pos = torch.tensor([pos], dtype=torch.float32)
    Pos = torch.mean(Pos, 1); Pos = Pos.view(128)
    f_neg = open("/home/usuaris/veu/laia.albors/vggish/"+ path + str(neg_id) + ".npy", 'rb')
    neg = np.load(f_neg); Neg = torch.tensor([neg], dtype=torch.float32)
    Neg = torch.mean(Neg, 1); Neg = Neg.view(128)

    return (Ref, Pos, Neg), []

class SingularVGGishMean(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels, train):
        'Initialization'
        self.labels = labels     # diccionari {id: label}
        self.list_IDs = list_IDs # llista [id1,id2...]
        self.train = train       # True or False

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        if self.train:
          path = "train/"
        else:
          path = "validation/"

        # Load data and get label
        f = open("/home/usuaris/veu/laia.albors/vggish/"+ path + str(ID) + ".npy", 'rb')
        x = np.load(f); X = torch.tensor([x], dtype=torch.float32)
        X = torch.mean(X, 1); X = X.view(128)
        y = self.labels[ID]

        return X, y


#---------------------#
# VGGish fine-tuning #
#---------------------#
class TripletVGGishFT(Dataset):

  def __init__(self, list_IDs, labels, train):
    'Initialization'
    self.labels = labels     # diccionari {id: label}
    self.list_IDs = list_IDs # llista [id1,id2...]
    self.train = train

    if self.train:
      self.train_labels = {id: self.labels[id] for id in self.list_IDs} # diccionari {id: label}
      self.labels_set = set(self.train_labels.values())                 # set (label1,label2...)
      self.label_to_indices = {label: [self.list_IDs.index(k) for k,v in self.train_labels.items() if v == label]
                               for label in self.labels_set}            # diccionari {label: [idx1,idx2...]}
    else:
      self.valid_labels = {id: self.labels[id] for id in self.list_IDs} # diccionari {id: label}
      # generate fixed triplets for validation
      self.labels_set = set(self.valid_labels.values())                 # set (label1,label2...)
      self.label_to_indices = {label: [self.list_IDs.index(k) for k,v in self.valid_labels.items() if v == label]
                               for label in self.labels_set}            # diccionari {label: [idx1,idx2...]}

      random_state = np.random.RandomState(29)

      triplets = [[i,
                    random_state.choice(self.label_to_indices[self.valid_labels[self.list_IDs[i]]]),
                    random_state.choice(self.label_to_indices[
                                            np.random.choice(
                                                list(self.labels_set - set([self.valid_labels[self.list_IDs[i]]]))
                                            )
                                        ])
                    ]
                  for i in range(len(self.list_IDs))]
      self.valid_triplets = triplets

  def __len__(self):
    'Denotes the total number of samples'
    return len(self.list_IDs)

  def __getitem__(self, index):
    'Generates one sample of data'
    # Select sample
    if self.train:
      ref_id = self.list_IDs[index]
      label_ref = self.train_labels[ref_id]
      positive_index = index
      while positive_index == index:
          positive_index = np.random.choice(self.label_to_indices[label_ref])
      label_neg = np.random.choice(list(self.labels_set - set([label_ref])))
      negative_index = np.random.choice(self.label_to_indices[label_neg])
      pos_id = self.list_IDs[positive_index]
      neg_id = self.list_IDs[negative_index]
      path = "train/"
    else:
      ref_id = self.list_IDs[self.valid_triplets[index][0]]
      pos_id = self.list_IDs[self.valid_triplets[index][1]]
      neg_id = self.list_IDs[self.valid_triplets[index][2]]
      path = "validation/"

    # Load data
    f_ref = open("/home/usuaris/veu/laia.albors/datasetWavfiles/"+ path + str(ref_id) + ".npy", 'rb')
    ref = np.load(f_ref); Ref = np.mean(ref, 0); Ref = torch.tensor([Ref], dtype=torch.float32)
    f_pos = open("/home/usuaris/veu/laia.albors/datasetWavfiles/"+ path + str(pos_id) + ".npy", 'rb')
    pos = np.load(f_pos); Pos = np.mean(pos, 0); Pos = torch.tensor([Pos], dtype=torch.float32)
    f_neg = open("/home/usuaris/veu/laia.albors/datasetWavfiles/"+ path + str(neg_id) + ".npy", 'rb')
    neg = np.load(f_neg); Neg = np.mean(neg, 0); Neg = torch.tensor([Neg], dtype=torch.float32)

    #ref = wavfile_to_examples("/home/usuaris/veu/laia.albors/datasetChunks/"+ path + str(ref_id) + ".wav", return_tensor=False)
    #Ref = np.mean(ref, 0); Ref = torch.tensor([Ref], dtype=torch.float32)

    return (Ref, Pos, Neg), []

class SingularVGGishFT(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels, train):
        'Initialization'
        self.labels = labels     # diccionari {id: label}
        self.list_IDs = list_IDs # llista [id1,id2...]
        self.train = train       # True or False

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        if self.train:
          path = "train/"
        else:
          path = "validation/"

        # Load data and get label
        f = open("/home/usuaris/veu/laia.albors/datasetWavfiles/"+ path + str(ID) + ".npy", 'rb')
        x = np.load(f); X = np.mean(x, 0); X = torch.tensor([X], dtype=torch.float32)

        #x = wavfile_to_examples("/home/usuaris/veu/laia.albors/datasetChunks/"+ path + str(ID) + ".wav", return_tensor=False)
        #X = np.mean(x, 0); X = torch.tensor([X], dtype=torch.float32)

        y = self.labels[ID]

        return X, y

class TripletVGGishMeanFT(Dataset):

  def __init__(self, list_IDs, labels, train):
    'Initialization'
    self.labels = labels     # diccionari {id: label}
    self.list_IDs = list_IDs # llista [id1,id2...]
    self.train = train

    if self.train:
      self.train_labels = {id: self.labels[id] for id in self.list_IDs} # diccionari {id: label}
      self.labels_set = set(self.train_labels.values())                 # set (label1,label2...)
      self.label_to_indices = {label: [self.list_IDs.index(k) for k,v in self.train_labels.items() if v == label]
                               for label in self.labels_set}            # diccionari {label: [idx1,idx2...]}
    else:
      self.valid_labels = {id: self.labels[id] for id in self.list_IDs} # diccionari {id: label}
      # generate fixed triplets for validation
      self.labels_set = set(self.valid_labels.values())                 # set (label1,label2...)
      self.label_to_indices = {label: [self.list_IDs.index(k) for k,v in self.valid_labels.items() if v == label]
                               for label in self.labels_set}            # diccionari {label: [idx1,idx2...]}

      random_state = np.random.RandomState(29)

      triplets = [[i,
                    random_state.choice(self.label_to_indices[self.valid_labels[self.list_IDs[i]]]),
                    random_state.choice(self.label_to_indices[
                                            np.random.choice(
                                                list(self.labels_set - set([self.valid_labels[self.list_IDs[i]]]))
                                            )
                                        ])
                    ]
                  for i in range(len(self.list_IDs))]
      self.valid_triplets = triplets

  def __len__(self):
    'Denotes the total number of samples'
    return len(self.list_IDs)

  def __getitem__(self, index):
    'Generates one sample of data'
    # Select sample
    if self.train:
      ref_id = self.list_IDs[index]
      label_ref = self.train_labels[ref_id]
      positive_index = index
      while positive_index == index:
          positive_index = np.random.choice(self.label_to_indices[label_ref])
      label_neg = np.random.choice(list(self.labels_set - set([label_ref])))
      negative_index = np.random.choice(self.label_to_indices[label_neg])
      pos_id = self.list_IDs[positive_index]
      neg_id = self.list_IDs[negative_index]
      path = "train/"
    else:
      ref_id = self.list_IDs[self.valid_triplets[index][0]]
      pos_id = self.list_IDs[self.valid_triplets[index][1]]
      neg_id = self.list_IDs[self.valid_triplets[index][2]]
      path = "validation/"

    # Load data
    f_ref = open("/home/usuaris/veu/laia.albors/datasetWavfiles/"+ path + str(ref_id) + ".npy", 'rb')
    Ref = np.load(f_ref); #Ref = np.mean(ref, 0);
    Ref = torch.tensor(Ref, dtype=torch.float32)
    f_pos = open("/home/usuaris/veu/laia.albors/datasetWavfiles/"+ path + str(pos_id) + ".npy", 'rb')
    Pos = np.load(f_pos); #Pos = np.mean(pos, 0);
    Pos = torch.tensor(Pos, dtype=torch.float32)
    f_neg = open("/home/usuaris/veu/laia.albors/datasetWavfiles/"+ path + str(neg_id) + ".npy", 'rb')
    Neg = np.load(f_neg); #Neg = np.mean(neg, 0);
    Neg = torch.tensor(Neg, dtype=torch.float32)

    return (Ref, Pos, Neg), []

class SingularVGGishMeanFT(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels, train):
        'Initialization'
        self.labels = labels     # diccionari {id: label}
        self.list_IDs = list_IDs # llista [id1,id2...]
        self.train = train       # True or False

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        if self.train:
          path = "train/"
        else:
          path = "validation/"

        # Load data and get label
        f = open("/home/usuaris/veu/laia.albors/datasetWavfiles/"+ path + str(ID) + ".npy", 'rb')
        X = np.load(f); #X = np.mean(x, 0);
        X = torch.tensor(X, dtype=torch.float32)

        y = self.labels[ID]

        return X, y


#################
# SEGMENT LEVEL #
#################

#---------#
# MELODIA #
#---------#
class TripletSeg(Dataset):

  def __init__(self, list_IDs, labels, train):
    'Initialization'
    self.labels = labels     # diccionari {id: [label, numSeg, max]}
    self.list_IDs = list_IDs # llista [id1,id2...]
    self.train = train

    if self.train:
      self.train_labels = {id: self.labels[id] for id in self.list_IDs}         # diccionari {id: [label, numSeg, max]}
      self.labels_set = set(np.array(list(self.train_labels.values()))[:,0])    # set (label1,label2...)
      self.label_to_indices = {label: [self.list_IDs.index(k) for k,v in self.train_labels.items() if v[0] == label]
                               for label in self.labels_set}                    # diccionari {label: [idx1,idx2...]}
    else:
      self.valid_labels = {id: self.labels[id] for id in self.list_IDs}         # diccionari {id: label}
      # generate fixed triplets for validation
      self.labels_set = set(np.array(list(self.valid_labels.values()))[:,0])    # set (label1,label2...)
      self.label_to_indices = {label: [self.list_IDs.index(k) for k,v in self.valid_labels.items() if v[0] == label]
                               for label in self.labels_set}                    # diccionari {label: [idx1,idx2...]}

    self.candidates = {}                                                        # diccionari {idx: [idx1, idx2...]}
    for id in self.list_IDs: # id = train-00000
        idx = self.list_IDs.index(id) # idx = 0
        #print("\n",id, idx, self.labels[id][0])
        self.candidates[idx] = [] # {0: []}
        for idx2 in self.label_to_indices[self.labels[id][0]]: # idx2 = 1
            id2 = self.list_IDs[idx2]
            #print("   ", id2, idx2)
            max1 = self.labels[id][2]; max2 = self.labels[id2][2]
            numSeg = round(max2/max1*self.labels[id][1])
            #print("      ", numSeg)
            if (self.labels[id2][1] in [numSeg-1, numSeg, numSeg+1]) and (id != id2):
                self.candidates[idx].append(idx2)

    #print(len(self.list_IDs), len(self.candidates[0]))

    if not self.train:
        random_state = np.random.RandomState(29)

        triplets = [[i,
                      random_state.choice(self.candidates[i]),
                      random_state.choice(list( set(range(len(self.list_IDs))) - set(self.candidates[i]) - set(self.label_to_indices[self.valid_labels[self.list_IDs[i]][0]]) ))
                     ]
                    for i in range(len(self.list_IDs))]
        self.valid_triplets = triplets

  def __len__(self):
    'Denotes the total number of samples'
    return len(self.list_IDs)

  def __getitem__(self, index):
    'Generates one sample of data'
    # Select sample
    if self.train:
      ref_id = self.list_IDs[index]
      label_ref = self.train_labels[ref_id][0]
      positive_index = np.random.choice(self.candidates[index])
      negative_index = np.random.choice(list(set(range(len(self.list_IDs))) - set(self.candidates[index]) - set(self.label_to_indices[label_ref])))
      pos_id = self.list_IDs[positive_index]
      neg_id = self.list_IDs[negative_index]
      path = "train/"
    else:
      ref_id = self.list_IDs[self.valid_triplets[index][0]]
      pos_id = self.list_IDs[self.valid_triplets[index][1]]
      neg_id = self.list_IDs[self.valid_triplets[index][2]]
      path = "validation/"

    ID = self.list_IDs[index]

    # Load data
    ref = pd.read_csv("../dataset/"+ path + str(ref_id) + ".csv", names=("time","value"))
    Ref = torch.tensor([np.array(ref['value'])], dtype=torch.float32)
    pos = pd.read_csv("../dataset/"+ path + str(pos_id) + ".csv", names=("time","value"))
    Pos = torch.tensor([np.array(pos['value'])], dtype=torch.float32)
    neg = pd.read_csv("../dataset/"+ path + str(neg_id) + ".csv", names=("time","value"))
    Neg = torch.tensor([np.array(neg['value'])], dtype=torch.float32)

    return (Ref, Pos, Neg), []

#-----------------------------#
# MELODIA using DTW alignment #
#-----------------------------#
class TripletSegDTW(Dataset):

  def positive_index(self, index):
    random_state = np.random.RandomState(29)

    ref_id = self.list_IDs[index]
    label_ref = self.valid_labels[ref_id]
    line = self.info[self.info.name == ref_id]
    ref_numSeg = int(line.numSeg)
    ref_trans = int(line.trans)
    dataFrame = self.alignment[label_ref]
    ref_column_name = "valid_7_" + str(ref_trans)
    v = list(dataFrame[ref_column_name])
    ref_row_num = random_state.choice([i for i, elem in enumerate(v) if str(ref_numSeg) in str(elem).split()])

    pos_column_name = ref_column_name
    while ref_column_name == pos_column_name:
        pos_trans = random_state.choice(range(121))
        pos_column_name = "valid_7_" + str(pos_trans)

    #print(label_ref, pos_column_name, ref_row_num, type(dataFrame[pos_column_name][ref_row_num]))
    pos_numSeg = int(random_state.choice(str(dataFrame[pos_column_name][ref_row_num]).split()))
    I = self.info[self.info.label == label_ref]
    I = I[I.trans == pos_trans]
    I = I[I.numSeg == pos_numSeg]
    pos_id = list(I.name)[0]

    return self.list_IDs.index(pos_id)

  def __init__(self, list_IDs, labels, alignment, info, train):
    'Initialization'
    self.labels = labels       # diccionari {id: label}
    self.list_IDs = list_IDs   # llista [id1,id2...]
    self.train = train
    self.alignment = alignment # diccionari {song1: df1, song2: df2...}
    self.info = info           # Data Frame amb [name, label, numSong, type, trans, numSeg]

    if self.train:
      self.train_labels = {id: self.labels[id] for id in self.list_IDs} # diccionari {id: label}
      self.labels_set = set(self.train_labels.values())                 # set (label1,label2...)
      self.label_to_indices = {label: [self.list_IDs.index(k) for k,v in self.train_labels.items() if v == label]
                               for label in self.labels_set}            # diccionari {label: [idx1,idx2...]}
    else:
      self.valid_labels = {id: self.labels[id] for id in self.list_IDs} # diccionari {id: label}
      # generate fixed triplets for validation
      self.labels_set = set(self.valid_labels.values())                 # set (label1,label2...)
      self.label_to_indices = {label: [self.list_IDs.index(k) for k,v in self.valid_labels.items() if v == label]
                               for label in self.labels_set}            # diccionari {label: [idx1,idx2...]}

      random_state = np.random.RandomState(29)

      triplets = [[i,
                    self.positive_index(i),
                    random_state.choice(self.label_to_indices[
                                            np.random.choice(
                                                list(self.labels_set - set([self.valid_labels[self.list_IDs[i]]]))
                                            )
                                        ])
                    ]
                  for i in range(len(self.list_IDs))]
      self.valid_triplets = triplets

  def __len__(self):
    'Denotes the total number of samples'
    return len(self.list_IDs)

  def __getitem__(self, index):
    'Generates one sample of data'
    # Select sample
    if self.train:
      ref_id = self.list_IDs[index]
      label_ref = self.train_labels[ref_id]
      line = self.info[self.info.name == ref_id]
      ref_numSong = int(line.numSong)
      ref_numSeg = int(line.numSeg)
      ref_trans = int(line.trans)
      dataFrame = self.alignment[label_ref]
      ref_column_name = "train_" + str(ref_numSong) + "_" + str(ref_trans)
      v = list(dataFrame[ref_column_name])
      ref_row_num = np.random.choice([i for i, elem in enumerate(v) if str(ref_numSeg) in str(elem).split()])

      pos_column_name = ref_column_name
      while ref_column_name == pos_column_name:
          pos_numSong = np.random.choice([1,2])
          pos_trans = np.random.choice(range(121))
          pos_column_name = "train_" + str(pos_numSong) + "_" + str(pos_trans)

      pos_numSeg = int(np.random.choice(str(dataFrame[pos_column_name][ref_row_num]).split()))
      I = self.info[self.info.label == label_ref]
      I = I[I.numSong == pos_numSong]
      I = I[I.trans == pos_trans]
      I = I[I.numSeg == pos_numSeg]
      pos_id = list(I.name)[0]

      label_neg = np.random.choice(list(self.labels_set - set([label_ref])))
      negative_index = np.random.choice(self.label_to_indices[label_neg])
      neg_id = self.list_IDs[negative_index]
      path = "train/"
    else:
      ref_id = self.list_IDs[self.valid_triplets[index][0]]
      pos_id = self.list_IDs[self.valid_triplets[index][1]]
      neg_id = self.list_IDs[self.valid_triplets[index][2]]
      path = "validation/"

    # Load data
    ref = pd.read_csv("../dataset/"+ path + str(ref_id) + ".csv", names=("time","value"))
    Ref = torch.tensor([np.array(ref['value'])], dtype=torch.float32)
    pos = pd.read_csv("../dataset/"+ path + str(pos_id) + ".csv", names=("time","value"))
    Pos = torch.tensor([np.array(pos['value'])], dtype=torch.float32)
    neg = pd.read_csv("../dataset/"+ path + str(neg_id) + ".csv", names=("time","value"))
    Neg = torch.tensor([np.array(neg['value'])], dtype=torch.float32)

    return (Ref, Pos, Neg), []


#--------#
# VGGISH #
#--------#
class TripletSegVGGish(Dataset):

  def __init__(self, list_IDs, labels, train):
    'Initialization'
    self.labels = labels     # diccionari {id: [label, numSeg, max]}
    self.list_IDs = list_IDs # llista [id1,id2...]
    self.train = train

    if self.train:
      self.train_labels = {id: self.labels[id] for id in self.list_IDs}         # diccionari {id: [label, numSeg, max]}
      self.labels_set = set(np.array(list(self.train_labels.values()))[:,0])    # set (label1,label2...)
      self.label_to_indices = {label: [self.list_IDs.index(k) for k,v in self.train_labels.items() if v[0] == label]
                               for label in self.labels_set}                    # diccionari {label: [idx1,idx2...]}
    else:
      self.valid_labels = {id: self.labels[id] for id in self.list_IDs}         # diccionari {id: label}
      # generate fixed triplets for validation
      self.labels_set = set(np.array(list(self.valid_labels.values()))[:,0])    # set (label1,label2...)
      self.label_to_indices = {label: [self.list_IDs.index(k) for k,v in self.valid_labels.items() if v[0] == label]
                               for label in self.labels_set}                    # diccionari {label: [idx1,idx2...]}

    self.candidates = {}                                                        # diccionari {idx: [idx1, idx2...]}
    for id in self.list_IDs: # id = train-00000
        idx = self.list_IDs.index(id) # idx = 0
        self.candidates[idx] = [] # {0: []}
        for idx2 in self.label_to_indices[self.labels[id][0]]: # idx2 = 1
            id2 = self.list_IDs[idx2]
            max1 = self.labels[id][2]; max2 = self.labels[id2][2]
            numSeg = round(max2/max1*self.labels[id][1])
            if (self.labels[id2][1] in [numSeg-1, numSeg, numSeg+1]) and (id != id2):
                self.candidates[idx].append(idx2)

    if not self.train:
        random_state = np.random.RandomState(29)

        triplets = [[i,
                      random_state.choice(self.candidates[i]),
                      random_state.choice(list( set(range(len(self.list_IDs))) - set(self.candidates[i]) - set(self.label_to_indices[self.valid_labels[self.list_IDs[i]][0]]) ))
                     ]
                    for i in range(len(self.list_IDs))]
        self.valid_triplets = triplets

  def __len__(self):
    'Denotes the total number of samples'
    return len(self.list_IDs)

  def __getitem__(self, index):
    'Generates one sample of data'
    # Select sample
    if self.train:
      ref_id = self.list_IDs[index]
      label_ref = self.train_labels[ref_id][0]
      positive_index = np.random.choice(self.candidates[index])
      negative_index = np.random.choice(list(set(range(len(self.list_IDs))) - set(self.candidates[index]) - set(self.label_to_indices[label_ref])))
      pos_id = self.list_IDs[positive_index]
      neg_id = self.list_IDs[negative_index]
      path = "train/"
    else:
      ref_id = self.list_IDs[self.valid_triplets[index][0]]
      pos_id = self.list_IDs[self.valid_triplets[index][1]]
      neg_id = self.list_IDs[self.valid_triplets[index][2]]
      path = "validation/"

    ID = self.list_IDs[index]

    # Load data
    f_ref = open("/home/usuaris/veu/laia.albors/vggish/"+ path + str(ref_id) + ".npy", 'rb')
    ref = np.load(f_ref); Ref = torch.tensor([ref], dtype=torch.float32)
    f_pos = open("/home/usuaris/veu/laia.albors/vggish/"+ path + str(pos_id) + ".npy", 'rb')
    pos = np.load(f_pos); Pos = torch.tensor([pos], dtype=torch.float32)
    f_neg = open("/home/usuaris/veu/laia.albors/vggish/"+ path + str(neg_id) + ".npy", 'rb')
    neg = np.load(f_neg); Neg = torch.tensor([neg], dtype=torch.float32)

    return (Ref, Pos, Neg), []

#----------------------------#
# VGGish using DTW alignment #
#----------------------------#
class TripletSegVGGishDTW(Dataset):

  def positive_index(self, index):
    random_state = np.random.RandomState(29)

    ref_id = self.list_IDs[index]
    label_ref = self.valid_labels[ref_id]
    line = self.info[self.info.name == ref_id]
    ref_numSeg = int(line.numSeg)
    ref_trans = int(line.trans)
    dataFrame = self.alignment[label_ref]
    ref_column_name = "valid_7_" + str(ref_trans)
    v = list(dataFrame[ref_column_name])
    ref_row_num = random_state.choice([i for i, elem in enumerate(v) if str(ref_numSeg) in str(elem).split()])

    pos_column_name = ref_column_name
    while ref_column_name == pos_column_name:
        pos_trans = random_state.choice(range(121))
        pos_column_name = "valid_7_" + str(pos_trans)

    #print(label_ref, pos_column_name, ref_row_num, type(dataFrame[pos_column_name][ref_row_num]))
    pos_numSeg = int(random_state.choice(str(dataFrame[pos_column_name][ref_row_num]).split()))
    I = self.info[self.info.label == label_ref]
    I = I[I.trans == pos_trans]
    I = I[I.numSeg == pos_numSeg]
    pos_id = list(I.name)[0]

    return self.list_IDs.index(pos_id)

  def __init__(self, list_IDs, labels, alignment, info, train):
    'Initialization'
    self.labels = labels       # diccionari {id: label}
    self.list_IDs = list_IDs   # llista [id1,id2...]
    self.train = train
    self.alignment = alignment # diccionari {song1: df1, song2: df2...}
    self.info = info           # Data Frame amb [name, label, numSong, type, trans, numSeg]

    if self.train:
      self.train_labels = {id: self.labels[id] for id in self.list_IDs} # diccionari {id: label}
      self.labels_set = set(self.train_labels.values())                 # set (label1,label2...)
      self.label_to_indices = {label: [self.list_IDs.index(k) for k,v in self.train_labels.items() if v == label]
                               for label in self.labels_set}            # diccionari {label: [idx1,idx2...]}
    else:
      self.valid_labels = {id: self.labels[id] for id in self.list_IDs} # diccionari {id: label}
      # generate fixed triplets for validation
      self.labels_set = set(self.valid_labels.values())                 # set (label1,label2...)
      self.label_to_indices = {label: [self.list_IDs.index(k) for k,v in self.valid_labels.items() if v == label]
                               for label in self.labels_set}            # diccionari {label: [idx1,idx2...]}

      random_state = np.random.RandomState(29)

      triplets = [[i,
                    self.positive_index(i),
                    random_state.choice(self.label_to_indices[
                                            np.random.choice(
                                                list(self.labels_set - set([self.valid_labels[self.list_IDs[i]]]))
                                            )
                                        ])
                    ]
                  for i in range(len(self.list_IDs))]
      self.valid_triplets = triplets

  def __len__(self):
    'Denotes the total number of samples'
    return len(self.list_IDs)

  def __getitem__(self, index):
    'Generates one sample of data'
    # Select sample
    if self.train:
      ref_id = self.list_IDs[index]
      label_ref = self.train_labels[ref_id]
      line = self.info[self.info.name == ref_id]
      ref_numSong = int(line.numSong)
      ref_numSeg = int(line.numSeg)
      ref_trans = int(line.trans)
      dataFrame = self.alignment[label_ref]
      ref_column_name = "train_" + str(ref_numSong) + "_" + str(ref_trans)
      v = list(dataFrame[ref_column_name])
      ref_row_num = np.random.choice([i for i, elem in enumerate(v) if str(ref_numSeg) in str(elem).split()])

      pos_column_name = ref_column_name
      while ref_column_name == pos_column_name:
          pos_numSong = np.random.choice([1,2])
          pos_trans = np.random.choice(range(121))
          pos_column_name = "train_" + str(pos_numSong) + "_" + str(pos_trans)

      pos_numSeg = int(np.random.choice(str(dataFrame[pos_column_name][ref_row_num]).split()))
      I = self.info[self.info.label == label_ref]
      I = I[I.numSong == pos_numSong]
      I = I[I.trans == pos_trans]
      I = I[I.numSeg == pos_numSeg]
      pos_id = list(I.name)[0]

      label_neg = np.random.choice(list(self.labels_set - set([label_ref])))
      negative_index = np.random.choice(self.label_to_indices[label_neg])
      neg_id = self.list_IDs[negative_index]
      path = "train/"
    else:
      ref_id = self.list_IDs[self.valid_triplets[index][0]]
      pos_id = self.list_IDs[self.valid_triplets[index][1]]
      neg_id = self.list_IDs[self.valid_triplets[index][2]]
      path = "validation/"

    # Load data
    f_ref = open("/home/usuaris/veu/laia.albors/vggish/"+ path + str(ref_id) + ".npy", 'rb')
    ref = np.load(f_ref); Ref = torch.tensor([ref], dtype=torch.float32)
    f_pos = open("/home/usuaris/veu/laia.albors/vggish/"+ path + str(pos_id) + ".npy", 'rb')
    pos = np.load(f_pos); Pos = torch.tensor([pos], dtype=torch.float32)
    f_neg = open("/home/usuaris/veu/laia.albors/vggish/"+ path + str(neg_id) + ".npy", 'rb')
    neg = np.load(f_neg); Neg = torch.tensor([neg], dtype=torch.float32)

    return (Ref, Pos, Neg), []
