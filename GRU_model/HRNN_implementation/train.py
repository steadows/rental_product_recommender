import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import os

from dataset import load_processed_data, SessionDataset, collate_fn
from model import HRNNRecommender

def train():
    # Hyperparameters
    BATCH_SIZE = 512        # Smaller batch size for stability
    EMBEDDING_DIM = 128     # Reverted to 128
    HIDDEN_DIM = 256        # Reverted to 256
    NUM_LAYERS = 1
    DROPOUT = 0.3           # Standard dropout
    LR = 0.01              # Standard LR
    MAX_EPOCHS = 30
    MAX_LEN = 20
    MAX_HISTORY = 5         # New hyperparameter
    
    DATA_DIR = 'processed_data'
    
    # Load Data
    train_data, val_data, test_data, id2idx, idx2id, vocab_size = load_processed_data(DATA_DIR)
    
    # Enable Augmentation for Training
    print("Initializing datasets (with augmentation)...")
    train_dataset = SessionDataset(train_data, id2idx, max_len=MAX_LEN, max_history=MAX_HISTORY, augment=True)
    val_dataset = SessionDataset(val_data, id2idx, max_len=MAX_LEN, max_history=MAX_HISTORY, augment=False)
    
    print(f"Augmented Train Size: {len(train_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # Model
    model = HRNNRecommender(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        lr=LR,
        batch_size=BATCH_SIZE,
        max_len=MAX_LEN,
        loss_type='top1'
    )
    
    # Logger
    logger = pl.loggers.TensorBoardLogger("tb_logs", name="hrnn_model_top1", default_hp_metric=False)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_recall_at_6',
        dirpath='checkpoints',
        filename='hrnn-aug-{epoch:02d}-{val_recall_at_6:.4f}',
        save_top_k=1,
        mode='max'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_recall_at_6',
        patience=3,
        min_delta=0.001,
        mode='max'
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator='mps',
        devices=1,
        logger=logger
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    print(f"Best model path: {checkpoint_callback.best_model_path}")
    return checkpoint_callback.best_model_path

if __name__ == '__main__':
    train()
