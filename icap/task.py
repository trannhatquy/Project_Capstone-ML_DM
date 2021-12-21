import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence


import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torchmetrics import Accuracy


class ImageCaptionTask(pl.LightningModule):
    def __init__(
        self, model, optimizer, criterion, vocab_size, scheduler=None, batch_first=True
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.vocab_size = vocab_size
        self.batch_first = batch_first
        self.metric = Accuracy()

    def forward(self, imgs, captions):
        outputs = self.model(imgs, captions[:-1]) # (batch_size, caption_length, vocab_size)
        return outputs

    def shared_step(self, batch, batch_idx):
        imgs, captions, lengths = batch # (batch_size, 3, 224, 224) (batch_size, caption_length) (batch_size)
        packed = pack_padded_sequence(captions, lengths, batch_first=self.batch_first) 
        targets = packed[0] 

        outputs = self.model(imgs, captions, lengths) 
        loss = self.criterion(outputs, targets) 
        acc = self.metric(outputs, targets) 
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        # self.log("train_acc", acc, prog_bar=True)
        return loss 

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        # self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        if self.scheduler:
            return [self.optimizer], [self.scheduler]
        return self.optimizer
