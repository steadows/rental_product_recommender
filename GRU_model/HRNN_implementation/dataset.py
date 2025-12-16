import torch
from torch.utils.data import Dataset
import pickle
import os
import numpy as np
import copy

class SessionDataset(Dataset):
    def __init__(self, sessions, vocab_map, max_len=20, max_history=5, augment=False):
        self.vocab_map = vocab_map
        self.max_len = max_len
        self.max_history = max_history
        self.pad_idx = 0
        
        if augment:
            self.sessions = []
            for s_data in sessions:
                seq = s_data['sequence']
                # Generate all sub-sessions with at least 2 items (1 input, 1 target)
                for i in range(2, len(seq) + 1):
                    new_entry = s_data.copy()
                    new_entry['sequence'] = seq[:i]
                    self.sessions.append(new_entry)
        else:
            self.sessions = sessions

    def __len__(self):
        return len(self.sessions)

    def process_seq(self, seq_raw):
        # Helper to map and pad a single sequence
        seq = [self.vocab_map[pid] for pid in seq_raw if pid in self.vocab_map]
        if len(seq) > self.max_len:
            seq = seq[-self.max_len:]
        return seq

    def __getitem__(self, idx):
        session_data = self.sessions[idx]
        
        # 1. Current Session
        curr_seq = self.process_seq(session_data['sequence'])
        curr_len = len(curr_seq)
        curr_tensor = torch.full((self.max_len,), self.pad_idx, dtype=torch.long)
        if curr_len > 0:
            curr_tensor[:curr_len] = torch.tensor(curr_seq, dtype=torch.long)
            
        # 2. History Sessions (List of Tensors)
        history_raw = session_data.get('history', [])
        history_lens = []
        
        # We need a fixed size tensor for batching: [max_history, max_len]
        hist_tensor_block = torch.full((self.max_history, self.max_len), self.pad_idx, dtype=torch.long)
        
        valid_hist_count = min(len(history_raw), self.max_history)
        
        # We take the LAST K sessions (most recent)
        # history_raw is already sorted [oldest ... newest] by link_sessions
        # So we take history_raw[-max_history:]
        
        relevant_history = history_raw[-self.max_history:] if valid_hist_count > 0 else []
        
        for i, raw_s in enumerate(relevant_history):
            proc_s = self.process_seq(raw_s)
            l = len(proc_s)
            
            if l > 0:
                hist_tensor_block[i, :l] = torch.tensor(proc_s, dtype=torch.long)
                history_lens.append(l)
            else:
                history_lens.append(0)
                
        # Pad history_lens to max_history size
        while len(history_lens) < self.max_history:
            history_lens.append(0)

        return {
            'sequence': curr_tensor,
            'lengths': torch.tensor(curr_len, dtype=torch.long),
            'history': hist_tensor_block, # [K, L]
            'history_lengths': torch.tensor(history_lens, dtype=torch.long), # [K]
            'history_count': torch.tensor(valid_hist_count, dtype=torch.long),
            'visit_id': str(session_data['visit_id']),
            'traffic_source': str(session_data.get('traffic_source', 'unknown'))
        }

def collate_fn(batch):
    return {
        'sequence': torch.stack([item['sequence'] for item in batch]),
        'lengths': torch.stack([item['lengths'] for item in batch]),
        'history': torch.stack([item['history'] for item in batch]),
        'history_lengths': torch.stack([item['history_lengths'] for item in batch]),
        'history_count': torch.stack([item['history_count'] for item in batch]),
        'visit_id': [item['visit_id'] for item in batch],
        'traffic_source': [item['traffic_source'] for item in batch]
    }

def link_sessions(all_sessions, max_history=5):
    """Links sessions to their previous K sessions based on client_id and time."""
    print(f"Linking user sessions (History={max_history})...")
    # Group by client_id
    user_sessions = {}
    for s in all_sessions:
        cid = s.get('client_id', 'unknown')
        if cid == 'unknown': continue
        if cid not in user_sessions:
            user_sessions[cid] = []
        user_sessions[cid].append(s)
    
    count = 0
    # Sort and link
    for cid, u_sess in user_sessions.items():
        # Sort by start_time
        u_sess.sort(key=lambda x: x.get('start_time', ''))
        
        for i in range(len(u_sess)):
            # Get past K sessions
            start_idx = max(0, i - max_history)
            past_sessions = u_sess[start_idx:i]
            
            # Extract sequences for history
            history_seqs = [ps['sequence'] for ps in past_sessions]
            u_sess[i]['history'] = history_seqs
            
            if len(history_seqs) > 0:
                count += 1
            
    print(f"Linked {count} sessions to previous history.")
    return all_sessions

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
        
    # Load Test Sessions
    with open(os.path.join(processed_dir, 'test_sessions.pkl'), 'rb') as f:
        test_data = pickle.load(f)
        
    # --- FIX FOR DATA LEAKAGE ---
    # 1. Link Train sessions independently. 
    # This ensures Train data NEVER sees Test data as history.
    print("Linking Train sessions...")
    link_sessions(train_sessions_raw)
    
    # 2. Create a deep copy of Train to serve as context for Test.
    # We use deepcopy so that the subsequent linking doesn't mess up the clean Train set.
    print("Creating Train copy for Test context...")
    train_copy = copy.deepcopy(train_sessions_raw)
    
    # 3. Combine Train Copy + Test for linking Test history.
    # Test sessions can see Train sessions and previous Test sessions.
    combined_test_context = train_copy + test_data
    print("Linking Test sessions (with Train context)...")
    link_sessions(combined_test_context)
    
    # The `test_data` list elements have been modified in-place by link_sessions
    # because they were part of `combined_test_context`.
    test_data_linked = test_data
    
    # The `train_sessions_raw` list elements are UNTOUCHED by the second linking
    # because we used `train_copy`.
    train_sessions_linked = train_sessions_raw
    
    # -----------------------------
    
    # Now sort Train by time for splitting
    train_sessions_linked.sort(key=lambda x: x.get('start_time', ''))
    
    # Split Train/Val (Last 10%)
    split_idx = int(len(train_sessions_linked) * 0.9)
    train_data = train_sessions_linked[:split_idx]
    val_data = train_sessions_linked[split_idx:]
    
    print(f"Vocab Size: {vocab_size}")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data_linked)}")
    
    return train_data, val_data, test_data_linked, id2idx, idx2id, vocab_size
