import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

class HRNNRecommender(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=1, dropout=0.3, lr=0.001, batch_size=256, max_len=20, loss_type='top1'):
        super().__init__()
        self.save_hyperparameters()
        self.vocab_size = vocab_size
        self.loss_type = loss_type
        
        # Shared Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 1. Session Encoder (Inner GRU)
        # Encodes a sequence of items into a session vector
        self.session_gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        
        # 2. User GRU (Outer GRU)
        # Encodes a sequence of session vectors into a user state
        self.user_gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        
        # 3. Main GRU (Current Session)
        # Initialized by User GRU state
        self.main_gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def get_session_representation(self, seqs, lengths):
        # seqs: [batch, len]
        # lengths: [batch]
        embedded = self.embedding(seqs)
        
        # Pack
        # Handle 0 length sequences (padding sessions)
        clean_lengths = lengths.clone()
        clean_lengths[clean_lengths == 0] = 1
        
        packed = nn.utils.rnn.pack_padded_sequence(embedded, clean_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.session_gru(packed) # hidden: [layers, batch, dim]
        
        # Extract last layer
        final_state = hidden[-1] # [batch, dim]
        
        # Mask out empty sessions
        mask = (lengths > 0).float().unsqueeze(1).to(self.device)
        final_state = final_state * mask
        
        return final_state

    def forward(self, curr_seq, curr_lengths, history, history_lengths, history_count):
        # 1. Encode History Sessions
        # history: [batch, max_hist, max_len]
        batch_size, max_hist, max_len = history.size()
        
        # Flatten to process all past sessions in parallel
        flat_hist = history.view(-1, max_len) # [batch*max_hist, max_len]
        flat_lens = history_lengths.view(-1)  # [batch*max_hist]
        
        # Encode all past sessions
        flat_sess_reps = self.get_session_representation(flat_hist, flat_lens) # [batch*max_hist, dim]
        
        # Reshape back to [batch, max_hist, dim]
        user_seq = flat_sess_reps.view(batch_size, max_hist, -1)
        
        # 2. Run User GRU
        # We only want to run this on valid history counts
        # Handle 0 history count
        clean_hist_counts = history_count.clone()
        clean_hist_counts[clean_hist_counts == 0] = 1
        
        packed_user = nn.utils.rnn.pack_padded_sequence(user_seq, clean_hist_counts.cpu(), batch_first=True, enforce_sorted=False)
        _, user_hidden = self.user_gru(packed_user) # [layers, batch, dim]
        
        # If history_count was 0, user_hidden should be 0
        user_mask = (history_count > 0).float().view(1, batch_size, 1).to(self.device)
        user_hidden = user_hidden * user_mask
        
        # 3. Run Main GRU initialized with User State
        curr_embedded = self.embedding(curr_seq)
        curr_embedded = self.dropout(curr_embedded)
        
        # Pack current
        clean_curr_lens = curr_lengths.clone()
        clean_curr_lens[clean_curr_lens == 0] = 1
        
        packed_curr = nn.utils.rnn.pack_padded_sequence(curr_embedded, clean_curr_lens.cpu(), batch_first=True, enforce_sorted=False)
        
        # Pass user_hidden as initial state (h_0)
        output, _ = self.main_gru(packed_curr, user_hidden)
        
        # Unpack
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=curr_seq.size(1))
        
        logits = self.fc(output)
        return logits
    
    def bpr_loss(self, logits, targets):
        """
        Bayesian Personalized Ranking Loss with In-Batch Negative Sampling
        loss = -log(sigmoid(pos_score - neg_score))
        """
        # Flatten
        logits = logits.reshape(-1, self.vocab_size)
        targets = targets.reshape(-1)
        
        # Mask padding
        mask = targets > 0
        logits = logits[mask]
        targets = targets[mask]
        
        if targets.size(0) == 0: return torch.tensor(0.0, requires_grad=True, device=self.device)
        
        # Positive scores: gather the logit corresponding to the target item
        pos_scores = logits.gather(1, targets.unsqueeze(1)).squeeze()
        
        # Negative scores: Sample from the batch's targets (In-Batch Negative Sampling)
        # This ensures we sample "popular" items as negatives, which is harder and better for learning.
        neg_indices = targets[torch.randperm(targets.size(0), device=self.device)]
        
        # Handle collisions: if neg == pos, sample random from vocab
        collision_mask = (neg_indices == targets)
        if collision_mask.any():
            random_neg = torch.randint(1, self.vocab_size, (collision_mask.sum(),), device=self.device)
            neg_indices[collision_mask] = random_neg
            
        neg_scores = logits.gather(1, neg_indices.unsqueeze(1)).squeeze()
        
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8)
        return loss.mean()

    def top1_loss(self, logits, targets):
        """
        TOP1 Loss from Hidasi et al. (2015) with In-Batch Negative Sampling
        loss = sigmoid(neg_score - pos_score) + sigmoid(neg_score^2)
        """
        # Flatten and mask (same as BPR)
        logits = logits.reshape(-1, self.vocab_size)
        targets = targets.reshape(-1)
        mask = targets > 0
        logits = logits[mask]
        targets = targets[mask]
        
        if targets.size(0) == 0: return torch.tensor(0.0, requires_grad=True, device=self.device)
        
        pos_scores = logits.gather(1, targets.unsqueeze(1)).squeeze()
        
        # Negative scores: Sample from the batch's targets (In-Batch Negative Sampling)
        neg_indices = targets[torch.randperm(targets.size(0), device=self.device)]
        
        # Handle collisions
        collision_mask = (neg_indices == targets)
        if collision_mask.any():
            random_neg = torch.randint(1, self.vocab_size, (collision_mask.sum(),), device=self.device)
            neg_indices[collision_mask] = random_neg
            
        neg_scores = logits.gather(1, neg_indices.unsqueeze(1)).squeeze()
        
        # The paper adds a regularization term sigmoid(neg_scores ** 2)
        loss = torch.sigmoid(neg_scores - pos_scores) + torch.sigmoid(neg_scores ** 2)
        return loss.mean()

    def training_step(self, batch, batch_idx):
        curr_seq = batch['sequence']
        curr_lengths = batch['lengths']
        history = batch['history']
        history_lengths = batch['history_lengths']
        history_count = batch['history_count']
        
        # We need at least 2 items to train (input -> target)
        # Filter out sequences with length < 2
        valid_mask = curr_lengths > 1
        if not valid_mask.any():
            return None
            
        curr_seq = curr_seq[valid_mask]
        curr_lengths = curr_lengths[valid_mask]
        history = history[valid_mask]
        history_lengths = history_lengths[valid_mask]
        history_count = history_count[valid_mask]
        
        # Input: seq[:, :-1]
        # Target: seq[:, 1:]
        input_seq = curr_seq[:, :-1]
        target_seq = curr_seq[:, 1:]
        input_lengths = curr_lengths - 1
        
        logits = self(input_seq, input_lengths, history, history_lengths, history_count) # [batch, seq_len-1, vocab]
        
        # Flatten for loss
        # We need to mask out padding in the loss
        # target_seq has 0s for padding. CrossEntropyLoss(ignore_index=0) handles this.
        
        if self.loss_type == 'bpr':
            loss = self.bpr_loss(logits, target_seq)
        elif self.loss_type == 'top1':
            loss = self.top1_loss(logits, target_seq)
        else:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), target_seq.reshape(-1), ignore_index=0)
        
        self.log('train_loss', loss, prog_bar=True, batch_size=curr_seq.size(0))
        return loss
    
    def validation_step(self, batch, batch_idx):
        curr_seq = batch['sequence']
        curr_lengths = batch['lengths']
        history = batch['history']
        history_lengths = batch['history_lengths']
        history_count = batch['history_count']
        
        valid_mask = curr_lengths > 1
        if not valid_mask.any():
            return
            
        curr_seq = curr_seq[valid_mask]
        curr_lengths = curr_lengths[valid_mask]
        history = history[valid_mask]
        history_lengths = history_lengths[valid_mask]
        history_count = history_count[valid_mask]
        
        input_seq = curr_seq[:, :-1]
        target_seq = curr_seq[:, 1:]
        input_lengths = curr_lengths - 1
        
        logits = self(input_seq, input_lengths, history, history_lengths, history_count) # [batch, seq_len-1, vocab]
        
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
        if self.loss_type == 'bpr':
            loss = self.bpr_loss(logits, target_seq)
        elif self.loss_type == 'top1':
            loss = self.top1_loss(logits, target_seq)
        else:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), target_seq.reshape(-1), ignore_index=0)
        
        self.log('val_recall_at_6', recall, prog_bar=True, batch_size=batch_size)
        self.log('val_loss', loss, prog_bar=True, batch_size=batch_size)
        return recall
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_recall_at_6"}}
