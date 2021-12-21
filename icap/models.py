import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class Encoder(nn.Module):
    def __init__(self, embed_size, train_cnn=False, resnet_model=models.resnet50):
        super(Encoder, self).__init__()
        self.train_cnn = train_cnn  # fine tune or not
        resnet = resnet_model(pretrained=True)  # pretrained resnet
        resnet = self._fine_tune(resnet)  # fine tune or not
        modules = list(resnet.children())[:-1]  # remove the last fc layer

        self.resnet = nn.Sequential(*modules)  # resnet without the last fc layer
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)  # 2048 * embed_size
        self.dropout = nn.Dropout(0.5)  # dropout
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)  # BN

    def _fine_tune(self, resnet):
        if self.train_cnn:
            for param in resnet.parameters():
                param.requires_grad_(True)  # fine tune
        else:
            for param in resnet.parameters():
                param.requires_grad_(False)  # not fine tune

        return resnet

    def forward(self, images):
        features = self.resnet(images)  # (batch_size, 2048, H, W)
        features = features.view(features.size(0), -1)  # (batch_size, 2048 * H * W)
        embed = self.bn(self.embed(features))  # (batch_size, embed_size)
        return embed


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, embed_size)  # embed_size * vocab_size
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers, batch_first=True
        )  # embed_size * hidden_size
        self.linear = nn.Linear(hidden_size, vocab_size)  # vocab_size * hidden_size

        self._init_weights()

    def _init_weights(self):
        torch.nn.init.xavier_uniform_(self.linear.weight)  # Xavier uniform
        torch.nn.init.xavier_uniform_(self.embed.weight)  # Xavier uniform

    def forward(self, features, captions, lengths):
        features = features.unsqueeze(dim=1)  # (batch_size, 1, embed_size)

        embeddings = self.embed(captions)  # (batch_size, caption_length, embed_size)
        embeddings = torch.cat(
            (features, embeddings), dim=1
        )  # (batch_size, caption_length + 1, embed_size)
        packed = pack_padded_sequence(
            embeddings, lengths, batch_first=True
        )  
        hiddens, _ = self.lstm(packed)  # (batch_size, caption_length , hidden_size)
        outputs = self.linear(
            hiddens[0]
        )  # (batch_size, caption_length , vocab_size)
        return outputs


class ImageCaptionNet(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(ImageCaptionNet, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.encoder = Encoder(embed_size)
        self.decoder = Decoder(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions, lengths):
        features = self.encoder(images)  # (batch_size, embed_size)
        outputs = self.decoder(
            features, captions, lengths
        )  # (batch_size, caption_length, vocab_size)
        return outputs
