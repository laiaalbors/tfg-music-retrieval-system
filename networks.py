import torch.nn as nn
import torch.nn.functional as F


class EmbeddingNet(nn.Module):
    def __init__(self, hidden_dim=32, output_dim=64, kernel_size=5, nhid=2048, embedding_size=256, dropout=True):
        super(EmbeddingNet, self).__init__()
        s = int(((3442 - 2*int(kernel_size/2))/2 - 2*int(kernel_size/2))/2)
        if dropout:
            self.embedding_extractor = nn.Sequential(nn.Conv1d(1, hidden_dim, kernel_size), nn.PReLU(),
                                                     nn.MaxPool1d(2, stride=2),
                                                     nn.Conv1d(hidden_dim, output_dim, kernel_size), nn.PReLU(),
                                                     nn.MaxPool1d(2, stride=2),
                                                     nn.Flatten(),
                                                     nn.Dropout(p=0.5),
                                                     nn.Linear(output_dim * s, nhid),
                                                     nn.PReLU(),
                                                     nn.Linear(nhid, embedding_size))
        else:
            self.embedding_extractor = nn.Sequential(nn.Conv1d(1, hidden_dim, kernel_size), nn.PReLU(),
                                                     nn.MaxPool1d(2, stride=2),
                                                     nn.Conv1d(hidden_dim, output_dim, kernel_size), nn.PReLU(),
                                                     nn.MaxPool1d(2, stride=2),
                                                     nn.Flatten(),
                                                     nn.Linear(output_dim * s, nhid),
                                                     nn.PReLU(),
                                                     nn.Linear(nhid, embedding_size))
        self.fc = nn.Sequential(nn.PReLU(),
                                nn.Linear(embedding_size, 2)
                                )

    def forward(self, x):
        output = self.embedding_extractor(x)
        output = self.fc(output)
        #output = output.view(output.size()[0], -1)
        return output

    def get_embedding2D(self, x):
        return self.forward(x)

    def get_embedding(self, x):
        output = self.embedding_extractor(x)
        return output


class EmbeddingNetPitch(nn.Module):
    def __init__(self, hidden_dim=32, output_dim=64, kernel_size=5, nhid=2048, embedding_size=256, dropout=True):
        super(EmbeddingNetPitch, self).__init__()
        s = int(((999 - 2*int(kernel_size/2))/2 - 2*int(kernel_size/2))/2)
        if dropout:
            self.embedding_extractor = nn.Sequential(nn.Conv1d(1, hidden_dim, kernel_size), nn.PReLU(),
                                                     nn.MaxPool1d(2, stride=2),
                                                     nn.Conv1d(hidden_dim, output_dim, kernel_size), nn.PReLU(),
                                                     nn.MaxPool1d(2, stride=2),
                                                     nn.Flatten(),
                                                     nn.Dropout(p=0.5),
                                                     nn.Linear(output_dim * s, nhid),
                                                     nn.PReLU(),
                                                     nn.Linear(nhid, embedding_size))
        else:
            self.embedding_extractor = nn.Sequential(nn.Conv1d(1, hidden_dim, kernel_size), nn.PReLU(),
                                                     nn.MaxPool1d(2, stride=2),
                                                     nn.Conv1d(hidden_dim, output_dim, kernel_size), nn.PReLU(),
                                                     nn.MaxPool1d(2, stride=2),
                                                     nn.Flatten(),
                                                     nn.Linear(output_dim * s, nhid),
                                                     nn.PReLU(),
                                                     nn.Linear(nhid, embedding_size))
        self.fc = nn.Sequential(nn.PReLU(),
                                nn.Linear(embedding_size, 2)
                                )

    def forward(self, x):
        output = self.embedding_extractor(x)
        output = self.fc(output)
        #output = output.view(output.size()[0], -1)
        return output

    def get_embedding2D(self, x):
        return self.forward(x)

    def get_embedding(self, x):
        output = self.embedding_extractor(x)
        return output


class EmbeddingNetVGGish(nn.Module):
    def __init__(self, hidden_dim=32, output_dim=64, kernel_size=5, nhid=2048, embedding_size=256, dropout=True):
        super(EmbeddingNetVGGish, self).__init__()
        s1 = int(((10 - 2*int(kernel_size/2))/2 - 2*int(kernel_size/2))/2)
        s2 = int(((128 - 2*int(kernel_size/2))/2 - 2*int(kernel_size/2))/2)
        if dropout:
            self.embedding_extractor = nn.Sequential(nn.Conv2d(1, hidden_dim, kernel_size), nn.PReLU(),
                                                     nn.MaxPool2d(2, stride=2),
                                                     nn.Conv2d(hidden_dim, output_dim, kernel_size), nn.PReLU(),
                                                     nn.MaxPool2d(2, stride=2),
                                                     nn.Flatten(),
                                                     nn.Dropout(p=0.5),
                                                     nn.Linear(output_dim * s1 * s2, nhid),
                                                     nn.PReLU(),
                                                     nn.Linear(nhid, embedding_size))
        else:
            self.embedding_extractor = nn.Sequential(nn.Conv2d(1, hidden_dim, kernel_size), nn.PReLU(),
                                                     nn.MaxPool2d(2, stride=2),
                                                     nn.Conv2d(hidden_dim, output_dim, kernel_size), nn.PReLU(),
                                                     nn.MaxPool2d(2, stride=2),
                                                     nn.Flatten(),
                                                     nn.Linear(output_dim * s1 * s2, nhid),
                                                     nn.PReLU(),
                                                     nn.Linear(nhid, embedding_size))
        self.fc = nn.Sequential(nn.PReLU(),
                                nn.Linear(embedding_size, 2)
                                )

    def forward(self, x):
        output = self.embedding_extractor(x)
        output = self.fc(output)
        return output

    def get_embedding2D(self, x):
        return self.forward(x)

    def get_embedding(self, x):
        output = self.embedding_extractor(x)
        return output

class EmbeddingNetVGGishSmall(nn.Module):
    def __init__(self, output_dim=32, kernel_size=5, embedding_size=256, dropout=True):
        super(EmbeddingNetVGGishSmall, self).__init__()
        s1 = int((10 - 2*int(kernel_size/2))/2)
        s2 = int((128 - 2*int(kernel_size/2))/2)
        if dropout:
            self.embedding_extractor = nn.Sequential(nn.Conv2d(1, output_dim, kernel_size), nn.PReLU(),
                                                     nn.MaxPool2d(2, stride=2),
                                                     nn.Flatten(),
                                                     nn.Dropout(p=0.5),
                                                     nn.Linear(output_dim * s1 * s2, embedding_size),
                                                     nn.PReLU())
        else:
            self.embedding_extractor = nn.Sequential(nn.Conv2d(1, output_dim, kernel_size), nn.PReLU(),
                                                     nn.MaxPool2d(2, stride=2),
                                                     nn.Flatten(),
                                                     nn.Linear(output_dim * s1 * s2, embedding_size),
                                                     nn.PReLU())
        self.fc = nn.Sequential(nn.Linear(embedding_size, 2))

    def forward(self, x):
        output = self.embedding_extractor(x)
        output = self.fc(output)
        return output

    def get_embedding2D(self, x):
        return self.forward(x)

    def get_embedding(self, x):
        output = self.embedding_extractor(x)
        return output

class EmbeddingNetVGGishSmallMean(nn.Module):
    def __init__(self, embedding_size=256):
        super(EmbeddingNetVGGishSmallMean, self).__init__()
        self.embedding_extractor = nn.Sequential(nn.Linear(128, embedding_size),
                                                 nn.PReLU())
        self.fc = nn.Sequential(nn.Linear(embedding_size, 2))

    def forward(self, x):
        output = self.embedding_extractor(x)
        output = self.fc(output)
        return output

    def get_embedding2D(self, x):
        return self.forward(x)

    def get_embedding(self, x):
        output = self.embedding_extractor(x)
        return output


class EmbeddingNetVGGishSmallSmall(nn.Module):
    def __init__(self, embedding_size=256):
        super(EmbeddingNetVGGishSmallSmall, self).__init__()
        #self.flat = nn.Flatten();
        self.embedding_extractor = nn.Sequential(nn.Linear(10*128, embedding_size),
                                                 nn.PReLU())
        self.fc = nn.Sequential(nn.Linear(embedding_size, 2))

    def forward(self, x):
        #print("MIDA X:", x.shape)
        #output = self.flat(x)
        output = x.view(x.shape[0], -1)
        #print("MIDA OUTPUT:", output.shape)
        output = self.embedding_extractor(output)
        output = self.fc(output)
        return output

    def get_embedding2D(self, x):
        return self.forward(x)

    def get_embedding(self, x):
        output = x.view(x.shape[0], -1)
        output = self.embedding_extractor(output)
        return output


class EmbeddingNetVGGishFT(nn.Module):
    def __init__(self, feature_extractor, embedding_extractor):
        super(EmbeddingNetVGGishFT, self).__init__()
        self.feature_extractor = feature_extractor
        self.flat = nn.Flatten()
        self.embedding_extractor = embedding_extractor
        self.fc = nn.Sequential(nn.Linear(128, 2))

    def forward(self, x):
        output = self.feature_extractor(x)
        output = self.flat(output)
        output = self.embedding_extractor(output)
        output = self.fc(output)
        #output = output.view(output.size()[0], -1)
        return output

    def get_embedding2D(self, x):
        return self.forward(x)

    def get_embedding(self, x):
        output = self.feature_extractor(x)
        output = self.flat(output)
        output = self.embedding_extractor(output)
        return output

class EmbeddingNetVGGishMeanFT(nn.Module):
    def __init__(self, feature_extractor, embedding_extractor):
        super(EmbeddingNetVGGishMeanFT, self).__init__()
        self.feature_extractor = feature_extractor
        self.flat = nn.Flatten()
        self.embedding_extractor = embedding_extractor
        self.fc = nn.Sequential(nn.Linear(128, 2))

    def forward(self, x):
        output = x.view(-1, 1, x.shape[2], x.shape[3])
        output = self.feature_extractor(output)
        output = self.flat(output)
        output = self.embedding_extractor(output)
        output = output.view(x.shape[0], x.shape[1], -1)
        output = output.mean(dim=1)
        output = self.fc(output)
        #output = output.view(output.size()[0], -1)
        return output

    def get_embedding2D(self, x):
        return self.forward(x)

    def get_embedding(self, x):
        output = x.view(-1, 1, x.shape[2], x.shape[3])
        output = self.feature_extractor(output)
        output = self.flat(output)
        output = self.embedding_extractor(output)
        output = output.view(x.shape[0], x.shape[1], -1)
        output = output.mean(dim=1)
        return output

class EmbeddingNetVGGishSmallFT(nn.Module):
    def __init__(self, feature_extractor, embedding_extractor, output_dim=32, kernel_size=5, embedding_size=256, dropout=True):
        super(EmbeddingNetVGGishSmallFT, self).__init__()
        s1 = int((10 - 2*int(kernel_size/2))/2)
        s2 = int((128 - 2*int(kernel_size/2))/2)
        self.feature_extractor = feature_extractor
        self.flat = nn.Flatten()
        self.embedding_extractor = embedding_extractor
        if dropout:
            self.extra_layers = nn.Sequential(nn.Conv2d(1, output_dim, kernel_size), nn.PReLU(),
                                              nn.MaxPool2d(2, stride=2),
                                              nn.Flatten(),
                                              nn.Dropout(p=0.5),
                                              nn.Linear(output_dim * s1 * s2, embedding_size),
                                              nn.PReLU())
        else:
            self.extra_layers = nn.Sequential(nn.Conv2d(1, output_dim, kernel_size), nn.PReLU(),
                                              nn.MaxPool2d(2, stride=2),
                                              nn.Flatten(),
                                              nn.Linear(output_dim * s1 * s2, embedding_size),
                                              nn.PReLU())
        self.fc = nn.Sequential(nn.Linear(embedding_size, 2))

    def forward(self, x):
        output = x.view(-1, 1, x.shape[2], x.shape[3])
        output = self.feature_extractor(output)
        output = self.flat(output)
        output = self.embedding_extractor(output)
        output = output.view(x.shape[0], 1, x.shape[1], -1)
        output = self.extra_layers(output)
        output = self.fc(output)
        return output

    def get_embedding2D(self, x):
        return self.forward(x)

    def get_embedding(self, x):
        output = x.view(-1, 1, x.shape[2], x.shape[3])
        output = self.feature_extractor(output)
        output = self.flat(output)
        output = self.embedding_extractor(output)
        output = output.view(x.shape[0], 1, x.shape[1], -1)
        output = self.extra_layers(output)
        return output

class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding2D(self, x):
        return self.embedding_net(x)

    def get_embedding(self, x):
        return self.embedding_net.embedding_extractor(x)

    def get_embeddingSmall(self, x):
        output = x.view(x.shape[0], -1)
        return self.embedding_net.embedding_extractor(output)

    def get_embeddingFT(self, x):
        output = x.view(-1, 1, x.shape[2], x.shape[3])
        output = self.embedding_net.feature_extractor(output)
        output = self.embedding_net.flat(output)
        output = self.embedding_net.embedding_extractor(output)
        output = output.view(x.shape[0], x.shape[1], -1)
        output = output.mean(dim=1)
        return output
