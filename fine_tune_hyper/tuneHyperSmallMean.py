from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import pandas as pd
from datasets import TripletVGGishMean, SingularVGGishMean

from losses import TripletLoss
from networks import EmbeddingNetVGGishSmallMean, TripletNet

from collections import Counter
from sklearn.preprocessing import normalize


def load_data(data_dir="../vggish", tr=True):
    train = pd.read_csv(data_dir+"/train/TRAIN.csv", header=0)
    valid = pd.read_csv(data_dir+"/validation/VALIDATION.csv", header=0)

    partition = {'train': list(train['name']), 'validation': list(valid['name'])}
    labels_train = pd.Series(list(train['label']), index=list(train['name'])).to_dict()
    labels_valid = pd.Series(valid.label.values, index=valid.name).to_dict()
    labels = {};
    for d in (labels_train, labels_valid):
        labels.update(d)

    if tr:
        triplet_train_dataset = TripletVGGishMean(partition['train'], labels, train=True)
        triplet_valid_dataset = TripletVGGishMean(partition['validation'], labels, train=False)
    else:
        triplet_train_dataset = SingularVGGishMean(partition['train'], labels, train=True)
        triplet_valid_dataset = SingularVGGishMean(partition['validation'], labels, train=False)

    return triplet_train_dataset, triplet_valid_dataset


def extract_embeddings(dataloader, model, size):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), size))
        labels = np.empty(len(dataloader.dataset),dtype=object)
        k = 0
        for images, target in dataloader:
            if torch.cuda.is_available():
                images = images.cuda()
            if size == 2:
                embeddings[k:k+len(images)] = model.get_embedding2D(images).data.cpu().numpy()
            else:
                embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target #.numpy()
            k += len(images)
    return embeddings, labels


def compute_accuracy(embeddings_train, targets_train, embeddings_valid, targets_valid, k=10):
    normed_embeddings_train = normalize(embeddings_train, axis=1, norm='l2')
    normed_embeddings_valid = normalize(embeddings_valid, axis=1, norm='l2')
    similarity_matrix = np.matmul(normed_embeddings_valid, np.transpose(normed_embeddings_train)) # valid_size X train_size

    l = []
    for v_emb_sim in similarity_matrix:
        indices = np.argpartition(v_emb_sim, -k)[-k:]
        l.append(targets_train[indices])

    count = 0
    for i,e in enumerate(l):
        c = Counter(e)
        label,_ = c.most_common()[0]
        count += (label == targets_valid[i])

    accuracy = count/len(targets_valid)*100
    return accuracy


def train_triplets(config, checkpoint_dir=None, data_dir=None):
    embedding_net = EmbeddingNetVGGishSmallMean(config['embedding_size'])
    net = TripletNet(embedding_net)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    loss_fn = TripletLoss(config['margin'])
    optimizer = optim.Adam(net.parameters(), lr=config['lr'])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainset, testset = load_data(data_dir)
    train_subset, val_subset = trainset, testset

    train_subset_sing, val_subset_sing = load_data(data_dir, False)

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        **kwargs)
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        **kwargs)

    trainloader_sing = torch.utils.data.DataLoader(
        train_subset_sing,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        **kwargs)
    valloader_sing = torch.utils.data.DataLoader(
        val_subset_sing,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        **kwargs)

    for epoch in range(20):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        net.train()
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            #print("INPUT:", inputs[0].shape)
            labels = labels if len(labels) > 0 else None
            if not type(inputs) in (tuple, list):
                inputs = (inputs,)
            inputs = tuple(d.to(device) for d in inputs)
            #inputs = inputs.to(device)
            if labels is not None:
              labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(*inputs)
            #print("OUTPUT:", outputs[0].shape)
            if type(outputs) not in (tuple, list):
              outputs = (outputs,)

            loss_inputs = outputs
            if labels is not None:
                labels = (labels,)
                loss_inputs += labels

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        net.eval()
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                labels = labels if len(labels) > 0 else None
                if not type(inputs) in (tuple, list):
                    inputs = (inputs,)
                inputs = tuple(d.to(device) for d in inputs)
                #inputs = inputs.to(device)
                if labels is not None:
                  labels  = labels.to(device)

                outputs = net(*inputs)

                loss_inputs = outputs
                if labels is not None:
                    labels = (labels,)
                    loss_inputs += labels

                loss_outputs = loss_fn(*loss_inputs)
                loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs

                val_loss += loss.cpu().numpy()
                val_steps += 1

        train_embeddings, train_labels = extract_embeddings(trainloader_sing, net, config['embedding_size'])
        val_embeddings, val_labels = extract_embeddings(valloader_sing, net, config['embedding_size'])
        accuracy = compute_accuracy(train_embeddings, train_labels, val_embeddings, val_labels, config['k'])

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=accuracy)
    print("Finished Training")


def main(num_samples=10, max_num_epochs=30, gpus_per_trial=2):
    data_dir = os.path.abspath("/home/usuaris/veu/laia.albors/vggish/")
    train_subset, val_subset = load_data(data_dir, False)
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=256,
        shuffle=True,
        **kwargs)
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=256,
        shuffle=False,
        **kwargs)

    config = {
        "batch_size": tune.choice([64, 128, 256]),
        "margin": tune.choice([1, 3, 5]),
        "lr": tune.choice([1e-3, 1e-4, 1e-5]),
        "embedding_size": tune.choice([32, 64, 128]),
        "k": 100
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(train_triplets, data_dir=data_dir),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="vggishSmallMean")

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    embedding_net = EmbeddingNetVGGishSmallMean(best_trial.config['embedding_size'])
    best_trained_model = TripletNet(embedding_net)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    train_embeddings, train_labels = extract_embeddings(trainloader, best_trained_model, best_trial.config['embedding_size'])
    val_embeddings, val_labels = extract_embeddings(valloader, best_trained_model, best_trial.config['embedding_size'])
    test_acc = compute_accuracy(train_embeddings, train_labels, val_embeddings, val_labels, config['k'])
    print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=30, max_num_epochs=30, gpus_per_trial=1)
