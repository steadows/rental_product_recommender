import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pandas as pd
import pickle
import os
from tqdm import tqdm
from collections import Counter
import glob

from dataset import load_processed_data, SessionDataset, collate_fn
from model import GRURecommender

def get_popular_items(train_sessions, k=50):
    """Calculate top K popular items from training sessions."""
    print("Calculating popular items...")
    cnt = Counter()
    for session in train_sessions:
        seq = session['sequence']
        cnt.update(seq)
    
    # Return list of product IDs (integers)
    popular = [pid for pid, count in cnt.most_common(k)]
    return popular

def predict():
    PROCESSED_DIR = 'processed_data'
    CHECKPOINT_DIR = 'checkpoints'
    
    # 1. Load Data
    # We need train data to calculate popularity
    train_data, _, test_data, id2idx, idx2id, vocab_size = load_processed_data(PROCESSED_DIR)
    
    # Calculate Global Popularity (Top 20 is enough to fill 6 spots)
    popular_items = get_popular_items(train_data, k=20)
    print(f"Top 5 popular items: {popular_items[:5]}")
    
    # Prepare Test Loader
    # We need to handle empty sessions carefully. 
    # The dataset class handles empty sequences by returning length 0.
    test_dataset = SessionDataset(test_data, id2idx, max_len=20)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    # 2. Load Model
    # Find best checkpoint
    if not os.path.exists(CHECKPOINT_DIR):
        print("Checkpoints directory not found. Run train.py first.")
        return

    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.ckpt')]
    if not checkpoints:
        print("No checkpoints found.")
        return
        
    # Sort by val_recall (assuming filename format has metric at end)
    # Filename: gru-recommender-epoch=XX-val_recall_at_6=0.XXXX.ckpt
    try:
        best_ckpt = sorted(checkpoints, key=lambda x: float(x.split('=')[-1].replace('.ckpt', '')), reverse=True)[0]
    except:
        # Fallback to modification time if filename parsing fails
        best_ckpt = sorted(checkpoints, key=lambda x: os.path.getmtime(os.path.join(CHECKPOINT_DIR, x)), reverse=True)[0]
        
    ckpt_path = os.path.join(CHECKPOINT_DIR, best_ckpt)
    print(f"Loading checkpoint: {ckpt_path}")
    
    model = GRURecommender.load_from_checkpoint(ckpt_path)
    model.eval()
    model.freeze()
    
    results = []
    
    print("Generating predictions...")
    for batch in tqdm(test_loader):
        seqs = batch['sequence']
        lengths = batch['lengths']
        visit_ids = batch['visit_id']
        
        # Identify non-empty sequences
        # pack_padded_sequence requires length > 0
        valid_mask = lengths > 0
        
        # Initialize predictions with empty lists
        batch_preds = [[] for _ in range(len(visit_ids))]
        
        if valid_mask.any():
            valid_seqs = seqs[valid_mask]
            valid_lengths = lengths[valid_mask]
            
            # Forward pass
            logits = model(valid_seqs, valid_lengths) # [batch_valid, seq_len, vocab]
            
            # Get last step logits
            batch_size_valid = logits.size(0)
            last_idxs = valid_lengths - 1
            last_logits = logits[torch.arange(batch_size_valid), last_idxs]
            
            # Get top 20 candidates (to ensure we have enough after filtering)
            _, top_k = torch.topk(last_logits, k=20, dim=1)
            top_k_np = top_k.cpu().numpy()
            
            # Map back to original indices in the batch
            valid_indices = torch.nonzero(valid_mask).squeeze().cpu().numpy()
            if valid_indices.ndim == 0:
                valid_indices = [valid_indices.item()]
                
            for i, batch_idx in enumerate(valid_indices):
                candidates = top_k_np[i]
                # Convert to product IDs
                pred_pids = []
                for idx in candidates:
                    if idx in idx2id:
                        pred_pids.append(idx2id[idx])
                batch_preds[batch_idx] = pred_pids
        
        # Fill with popular items and format
        for i, vid in enumerate(visit_ids):
            preds = batch_preds[i]
            
            # Fill with popular items if needed
            for pop_item in popular_items:
                if len(preds) >= 6:
                    break
                if pop_item not in preds:
                    preds.append(pop_item)
            
            # Truncate to exactly 6
            final_preds = preds[:6]
            
            results.append({
                'visit_id': vid,
                'product_ids': ' '.join(map(str, final_preds))
            })
            
    # Save submission
    df = pd.DataFrame(results)
    # Ensure columns are in correct order
    df = df[['visit_id', 'product_ids']]
    df.to_csv('submission.csv', index=False)
    print(f"Submission saved to submission.csv with {len(df)} rows.")

if __name__ == '__main__':
    predict()
