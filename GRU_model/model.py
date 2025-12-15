import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

class GRURecommender(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, num_layers=1, dropout=0.1, lr=0.001, batch_size=256, max_len=20):
        super().__init__()
        self.save_hyperparameters()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self.lr = lr
        self.vocab_size = vocab_size
        
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"val_recall_at_6": 0, "val_loss": 10, "train_loss": 10})
        
    def forward(self, x, lengths):
        # x: [batch, seq_len]
        emb = self.embedding(x) # [batch, seq_len, emb_dim]
        emb = self.dropout(emb)
        
        # Pack padded sequence
        # lengths must be on cpu for pack_padded_sequence
        packed_emb = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        packed_out, _ = self.gru(packed_emb)
        
        # Unpack
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        # out: [batch, seq_len, hidden_dim]
        
        logits = self.fc(out) # [batch, seq_len, vocab_size]
        return logits
    
    def training_step(self, batch, batch_idx):
        seqs = batch['sequence']
        lengths = batch['lengths']
        
        # We need at least 2 items to train (input -> target)
        # Filter out sequences with length < 2
        valid_mask = lengths > 1
        if not valid_mask.any():
            return None
            
        seqs = seqs[valid_mask]
        lengths = lengths[valid_mask]
        
        # Input: seq[:, :-1]
        # Target: seq[:, 1:]
        input_seq = seqs[:, :-1]
        target_seq = seqs[:, 1:]
        input_lengths = lengths - 1
        
        logits = self(input_seq, input_lengths) # [batch, seq_len-1, vocab]
        
        # Flatten for loss
        # We need to mask out padding in the loss
        # target_seq has 0s for padding. CrossEntropyLoss(ignore_index=0) handles this.
        
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), target_seq.reshape(-1), ignore_index=0)
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        seqs = batch['sequence']
        lengths = batch['lengths']
        
        valid_mask = lengths > 1
        if not valid_mask.any():
            return
            
        seqs = seqs[valid_mask]
        lengths = lengths[valid_mask]
        
        input_seq = seqs[:, :-1]
        target_seq = seqs[:, 1:]
        input_lengths = lengths - 1
        
        logits = self(input_seq, input_lengths) # [batch, seq_len-1, vocab]
        
        # We only care about the prediction for the last actual item in the input
        # The last actual item in input is at index input_lengths - 1
        
        batch_size = logits.size(0)
        last_logits = logits[torch.arange(batch_size), input_lengths - 1] # [batch, vocab]
        last_targets = target_seq[torch.arange(batch_size), input_lengths - 1] # [batch]
        
        # Calculate Recall@6
        # Get top 6
        _, top_k = torch.topk(last_logits, k=6, dim=1) # [batch, 6]
        
        # Check if target is in top_k
        hits = (top_k == last_targets.unsqueeze(1)).any(dim=1).float()
        recall = hits.mean()
        
        # Calculate Validation Loss (same as training)
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), target_seq.reshape(-1), ignore_index=0)
        
        self.log('val_recall_at_6', recall, prog_bar=True)
        self.log('val_loss', loss, prog_bar=True)
        return recall

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-8)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_recall_at_6"
            }
        }
