import torch
from torch.utils.data import DataLoader
from dataset import load_processed_data, SessionDataset, collate_fn
from model import GRURecommender
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
        all_items.extend(session['sequence']) # These are Product IDs
    
    counts = Counter(all_items)
    # Get top K Product IDs
    top_pids = [pid for pid, _ in counts.most_common(top_k)]
    
    # Convert to strings
    top_ids = [str(pid) for pid in top_pids]
    
    # Pad if we somehow don't have enough
    while len(top_ids) < top_k:
        if idx2id:
            # Pad with random valid item
            top_ids.append(str(list(idx2id.values())[0]))
        else:
            break
        
    return top_ids

def main():
    PROCESSED_DIR = 'processed_data'
    
    # 1. Load Data
    print("Loading data...")
    train_data, _, test_data, id2idx, idx2id, vocab_size = load_processed_data(PROCESSED_DIR)
    
    # 2. Calculate Fallback (Global Popularity)
    popular_fallback = get_global_popularity(train_data, idx2id, top_k=6)
    print(f"Global Popularity Fallback: {popular_fallback}")
    
    # 3. Prepare Test Loader
    test_dataset = SessionDataset(test_data, id2idx, max_len=20)
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
    
    model = GRURecommender.load_from_checkpoint(best_ckpt)
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
        
        # Move to device
        seqs = seqs.to(device)
        lengths = lengths.to(device)
        
        # Handle 0-length sequences (Cold Start)
        # We can only pass non-zero length sequences to the model
        valid_mask = lengths > 0
        
        # Initialize predictions with fallback
        batch_preds = []
        
        if valid_mask.any():
            # Filter valid inputs
            valid_seqs = seqs[valid_mask]
            valid_lengths = lengths[valid_mask]
            
            # Forward pass for valid sequences
            logits = model(valid_seqs, valid_lengths) # [valid_batch, seq_len, vocab]
            
            # Get predictions
            batch_size_valid = logits.size(0)
            gather_idxs = (valid_lengths - 1).clamp(min=0)
            last_logits = logits[torch.arange(batch_size_valid), gather_idxs]
            
            _, top_k = torch.topk(last_logits, k=6, dim=1)
            top_k_np = top_k.cpu().numpy()
            
            # Map back to full batch
            valid_indices = torch.nonzero(valid_mask).squeeze().cpu().numpy()
            if valid_indices.ndim == 0:
                valid_indices = [valid_indices.item()]
                
            # Store model predictions in a dict for easy lookup
            model_preds_map = {idx: preds for idx, preds in zip(valid_indices, top_k_np)}
        else:
            model_preds_map = {}
            
        lengths_np = lengths.cpu().numpy()
        
        for i, vid in enumerate(visit_ids):
            # Check for Cold Start / Empty Session
            if lengths_np[i] == 0:
                # Use Fallback
                pred_str = " ".join(popular_fallback)
            else:
                # Use Model Prediction
                if i in model_preds_map:
                    top_k_preds = model_preds_map[i]
                    pred_ids = [str(idx2id[idx]) for idx in top_k_preds if idx in idx2id]
                else:
                    # Should not happen if logic is correct
                    pred_ids = []
                
                # Fill with popular items if model predicts < 6 unique valid items (rare)
                if len(pred_ids) < 6:
                    for p in popular_fallback:
                        if p not in pred_ids:
                            pred_ids.append(p)
                            if len(pred_ids) == 6:
                                break
                                
                pred_str = " ".join(pred_ids[:6])
            
            results.append({
                'visit_id': vid,
                'product_ids': pred_str
            })
            
    # Save submission
    sub_df = pd.DataFrame(results)
    sub_df.to_csv('submission.csv', index=False)
    print(f"Saved submission.csv with {len(sub_df)} rows.")

if __name__ == '__main__':
    main()
