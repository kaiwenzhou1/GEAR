
import torch
import tracemalloc
import pytorch_lightning as pl
from torchmetrics import MeanMetric
from .models.sasrec import SASRec
from .models.GEAR import GEAR
from .utils import recalls_and_ndcgs_for_ks


class RecModel(pl.LightningModule):
    def __init__(self,
            backbone: GEAR,
            alpha: float = 0.5,
        ):
        super().__init__()

        self.backbone = backbone
        self.max_len = backbone.max_len
        self.alpha = alpha

        self.loss = torch.nn.CrossEntropyLoss(ignore_index=0)

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.average_loss = MeanMetric()
    
    def forward(self, input_ids, b_seq, time_bias):
        return self.backbone(input_ids, b_seq, time_bias)
    
    def predict(self, input_ids, b_seq, time_bias, candidates):
        return self.backbone.predict(input_ids, b_seq, time_bias, candidates)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        b_seq = batch['behaviors']
        user_id = batch['user_id']
        time_bias = batch['time_bias']
        
        interleaved = torch.empty(len(input_ids), self.max_len * 2, dtype=input_ids.dtype, device=input_ids.device)
        interleaved[:, 0::2] = b_seq
        interleaved[:, 1::2] = input_ids

        logits_item, logits_behavior = self(input_ids[:, :-1], b_seq, time_bias)

        # logits: [batch_size, seq_len, num_items + 1] -> [batch_size * seq_len, num_items + 1]
        logits_item = logits_item.reshape(-1, logits_item.size(-1))
        # labels: [batch_size, seq_len] -> [batch_size * seq_len]
        labels_item = input_ids
        labels_item = labels_item.reshape(-1)

        logits_behavior = logits_behavior.reshape(-1, logits_behavior.size(-1))
        labels_behavior = b_seq[:, 1:]
        labels_behavior = labels_behavior.reshape(-1)

        loss = self.loss(logits_item, labels_item) + self.alpha * self.loss(logits_behavior, labels_behavior)
        loss = loss.unsqueeze(0)
        self.average_loss.update(loss)
        return {'loss':loss}

    def on_train_epoch_end(self):
        # loss = torch.cat([o['loss'] for o in self.training_step_outputs], 0).mean()
        self.log('train_loss', self.average_loss)
        # self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        b_seq = batch['behaviors']
        user_id = batch['user_id']
        time_bias = batch['time_bias']
        candidates = batch['candidates'].squeeze() # B x C

        logits = self.predict(input_ids[:, :-1], b_seq, time_bias, candidates)
        labels = batch['labels'].squeeze()

        metrics = recalls_and_ndcgs_for_ks(logits, labels, [1, 5, 10, 20, 50])
        self.validation_step_outputs.append(metrics)
        return metrics
    
    def on_validation_epoch_end(self):
        keys = self.validation_step_outputs[0].keys()
        for k in keys:
            tmp = []
            for o in self.validation_step_outputs:
                tmp.append(o[k])
            self.log(f'Val:{k}', torch.Tensor(tmp).mean())
        self.validation_step_outputs.clear()
        