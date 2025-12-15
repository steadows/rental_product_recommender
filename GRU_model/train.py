import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import os

from dataset import load_processed_data, SessionDataset, collate_fn
from model import GRURecommender

def train():
    # Hyperparameters
    BATCH_SIZE = 512
    EMBEDDING_DIM = 64
    HIDDEN_DIM = 128
    NUM_LAYERS = 1
    DROPOUT = 0.1
    LR = 0.0001
    MAX_EPOCHS = 20
    MAX_LEN = 20
    
    DATA_DIR = 'processed_data'
    
    # Load Data
    train_data, val_data, test_data, id2idx, idx2id, vocab_size = load_processed_data(DATA_DIR)
    
    train_dataset = SessionDataset(train_data, id2idx, max_len=MAX_LEN)
    val_dataset = SessionDataset(val_data, id2idx, max_len=MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # Model
    model = GRURecommender(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        lr=LR,
        batch_size=BATCH_SIZE,
        max_len=MAX_LEN
    )
    
    # Logger
    logger = pl.loggers.TensorBoardLogger("tb_logs", name="gru_model", default_hp_metric=False)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_recall_at_6',
        dirpath='checkpoints',
        filename='gru-recommender-{epoch:02d}-{val_recall_at_6:.4f}',
        save_top_k=1,
        mode='max'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_recall_at_6',
        patience=3,
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
