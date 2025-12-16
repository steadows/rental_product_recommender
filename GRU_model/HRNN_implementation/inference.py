import torch
from torch.utils.data import DataLoader
from dataset import load_processed_data, SessionDataset, collate_fn
from model import HRNNRecommender
import pandas as pd
from tqdm import tqdm
import glob
import os
from collections import Counter

def get_global_popularity(train_data, idx2id, top_k=6):
    """Calculates the top K most popular items from training data."""
    print("Calculating global popularity...")
    all_items = []
    for session in train_data:
        all_items.extend(session['sequence'])
    
    counts = Counter(all_items)
    # Get top K Product IDs (NOT indices)
    top_pids = [pid for pid, _ in counts.most_common(top_k)]
    
    # Convert to strings
    top_ids = [str(pid) for pid in top_pids]
    
    # Pad if we somehow don't have enough
    if len(top_ids) < top_k:
        all_ids = list(idx2id.values())
        for item in all_ids:
            if str(item) not in top_ids:
                top_ids.append(str(item))
                if len(top_ids) == top_k:
                    break
        
    return top_ids

def get_source_popularity(train_data, idx2id, top_k=6):
    """Calculates the top K most popular items per traffic source."""
    print("Calculating source-based popularity...")
    source_items = {}
    
    for session in train_data:
        source = str(session.get('traffic_source', 'unknown'))
        if source not in source_items:
            source_items[source] = []
        source_items[source].extend(session['sequence'])
        
    source_pop = {}
    for source, items in source_items.items():
        counts = Counter(items)
        top_pids = [pid for pid, _ in counts.most_common(top_k)]
        top_ids = [str(pid) for pid in top_pids]
        source_pop[source] = top_ids
        
    return source_pop

def main():
    PROCESSED_DIR = 'processed_data'
    MAX_HISTORY = 5
    
    # 1. Load Data
    train_data, _, test_data, id2idx, idx2id, vocab_size = load_processed_data(PROCESSED_DIR)
    
    # 2. Calculate Fallbacks
    global_fallback = get_global_popularity(train_data, idx2id, top_k=6)
    source_fallback = get_source_popularity(train_data, idx2id, top_k=6)
    
    print(f"Global Popularity Fallback: {global_fallback}")
    print(f"Found {len(source_fallback)} traffic sources.")
    
    # 3. Prepare Test Loader
    # Note: augment=False is default, which is what we want for testing
    test_dataset = SessionDataset(test_data, id2idx, max_len=20, max_history=MAX_HISTORY)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # 4. Load Model
    ckpts = glob.glob('checkpoints/*.ckpt')
    if not ckpts:
        print("No checkpoint found. Please train first.")
        return
    
    # Sort by modification time
    ckpts.sort(key=os.path.getmtime)
    best_ckpt = ckpts[-1]
    print(f"Loading checkpoint: {best_ckpt}")
    
    model = HRNNRecommender.load_from_checkpoint(best_ckpt)
    model.eval()
    model.freeze()
    
    # Move model to MPS if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
        print("Using MPS for inference")
    else:
        device = torch.device("cpu")
        print("Using CPU for inference")

    results = []
    
    print("Running inference...")
    for batch in tqdm(test_loader):
        seqs = batch['sequence']
        lengths = batch['lengths']
        visit_ids = batch['visit_id']
        traffic_sources = batch['traffic_source']
        history = batch['history']
        history_lengths = batch['history_lengths']
        history_count = batch['history_count']
        
        # Move to device
        seqs = seqs.to(device)
        lengths = lengths.to(device)
        history = history.to(device)
        history_lengths = history_lengths.to(device)
        history_count = history_count.to(device)
        
        # Handle 0-length sequences (Cold Start)
        valid_mask = lengths > 0
        
        # Placeholder for predictions
        batch_preds = [[] for _ in range(len(visit_ids))]
        
        if valid_mask.any():
            # Filter valid inputs
            valid_seqs = seqs[valid_mask]
            valid_lengths = lengths[valid_mask]
            valid_history = history[valid_mask]
            valid_hist_lens = history_lengths[valid_mask]
            valid_hist_count = history_count[valid_mask]
            
            # Forward pass for valid sequences
            logits = model(valid_seqs, valid_lengths, valid_history, valid_hist_lens, valid_hist_count) # [valid_batch, seq_len, vocab]
            
            # Get predictions
            batch_size_valid = logits.size(0)
            gather_idxs = (valid_lengths - 1).clamp(min=0)
            last_logits = logits[torch.arange(batch_size_valid), gather_idxs]
            
            _, top_k = torch.topk(last_logits, k=20, dim=1)
            top_k_np = top_k.cpu().numpy()
            
            # Map back to full batch
            valid_indices = torch.nonzero(valid_mask).squeeze().cpu().numpy()
            if valid_indices.ndim == 0:
                valid_indices = [valid_indices.item()]
                
            for idx, preds in zip(valid_indices, top_k_np):
                candidates = [str(idx2id[p_idx]) for p_idx in preds if p_idx in idx2id]
                batch_preds[idx] = candidates

        # Fill with fallback and format
        for i, vid in enumerate(visit_ids):
            pred_ids = []
            
            # Add model predictions
            for item in batch_preds[i]:
                if item not in pred_ids:
                    pred_ids.append(item)
                    if len(pred_ids) == 6:
                        break
            
            # Fill with Contextual Fallback (Source Popularity)
            if len(pred_ids) < 6:
                source = str(traffic_sources[i])
                if source in source_fallback:
                    for item in source_fallback[source]:
                        if item not in pred_ids:
                            pred_ids.append(item)
                            if len(pred_ids) == 6:
                                break
            
            # Fill with Global Popularity (if still needed)
            if len(pred_ids) < 6:
                for item in global_fallback:
                    if item not in pred_ids:
                        pred_ids.append(item)
                        if len(pred_ids) == 6:
                            break
            
            results.append({
                'visit_id': vid,
                'product_ids': " ".join(pred_ids)
            })
            
    # Save submission
    sub_df = pd.DataFrame(results)
    sub_df.to_csv('submission.csv', index=False)
    print(f"Saved submission.csv with {len(sub_df)} rows.")

if __name__ == '__main__':
    main()
