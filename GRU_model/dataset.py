import torch
from torch.utils.data import Dataset
import pickle
import os
import numpy as np

class SessionDataset(Dataset):
    def __init__(self, sessions, vocab_map, max_len=20):
        self.sessions = sessions
        self.vocab_map = vocab_map
        self.max_len = max_len
        self.pad_idx = 0

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        session_data = self.sessions[idx]
        seq_raw = session_data['sequence']
        
        # Map product IDs to indices
        seq = [self.vocab_map[pid] for pid in seq_raw if pid in self.vocab_map]
        
        # Truncate
        if len(seq) > self.max_len:
            seq = seq[-self.max_len:]
            
        length = len(seq)
        
        # Pad
        seq_tensor = torch.full((self.max_len,), self.pad_idx, dtype=torch.long)
        if length > 0:
            seq_tensor[:length] = torch.tensor(seq, dtype=torch.long)
            
        return {
            'sequence': seq_tensor,
            'lengths': torch.tensor(length, dtype=torch.long),
            'visit_id': str(session_data['visit_id']) # Keep visit_id for inference
        }

def load_processed_data(processed_dir):
    print("Loading processed data...")
    
    # Load Product IDs
    with open(os.path.join(processed_dir, 'product_ids.pkl'), 'rb') as f:
        product_ids = pickle.load(f)
        
    # Create Mappings
    # 0 is padding
    valid_product_ids = sorted(list(product_ids))
    id2idx = {pid: i+1 for i, pid in enumerate(valid_product_ids)}
    idx2id = {i+1: pid for i, pid in enumerate(valid_product_ids)}
    vocab_size = len(valid_product_ids) + 1
    
    # Load Training Sessions
    with open(os.path.join(processed_dir, 'train_sessions.pkl'), 'rb') as f:
        train_sessions_raw = pickle.load(f)
        
    # Sort by time
    # Assuming start_time is a string, lexicographical sort works for ISO format
    # Filter out sessions without start_time if any
    train_sessions_raw = [s for s in train_sessions_raw if 'start_time' in s]
    train_sessions_raw.sort(key=lambda x: x.get('start_time', ''))
    
    # Split Train/Val (Last 10%)
    split_idx = int(len(train_sessions_raw) * 0.9)
    train_data = train_sessions_raw[:split_idx]
    val_data = train_sessions_raw[split_idx:]
    
    # Load Test Sessions
    with open(os.path.join(processed_dir, 'test_sessions.pkl'), 'rb') as f:
        test_data = pickle.load(f)
        
    print(f"Vocab Size: {vocab_size}")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    return train_data, val_data, test_data, id2idx, idx2id, vocab_size

def collate_fn(batch):
    # Pad sequences
    sequences = [item['sequence'] for item in batch]
    lengths = [item['lengths'] for item in batch]
    visit_ids = [item['visit_id'] for item in batch]
    
    # Stack
    padded_seqs = torch.stack(sequences)
    lengths_tensor = torch.stack(lengths)
    
    return {
        'visit_id': visit_ids,
        'sequence': padded_seqs,
        'lengths': lengths_tensor
    }
