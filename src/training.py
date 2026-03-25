import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt

class ImitationTrainer:
    def __init__(self, model, train_dataloader, val_dataloader, static_graph, config, use_wandb=True):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.use_wandb = use_wandb
        
        # Move static graph to device exactly once
        self.static_graph = {k: v.to(self.device) for k, v in static_graph.items()}
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 1e-3))
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.epochs = config.get('epochs', 10)
        
        # Internal loss tracking
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
        if use_wandb:
            wandb.init(
                project=config.get('project_name', "sidequest-imitation-learning"),
                name=config.get('run_name', model.__class__.__name__),
                config=config
            )
            wandb.watch(self.model, log="all", log_freq=10)

    def _process_batch(self, batch):
        """Helper to process a single batch to keep train/val loops DRY."""
        seed_mask_batch, y_batch, eval_mask_batch, event_feat_batch = batch
        
        seed_mask_batch = seed_mask_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        eval_mask_batch = eval_mask_batch.to(self.device)
        event_feat_batch = event_feat_batch.to(self.device)
        
        B, N, _ = seed_mask_batch.shape
        expanded_features = self.static_graph['x_static'].unsqueeze(0).expand(B, N, -1)
        x_batch = torch.cat([seed_mask_batch, expanded_features], dim=-1)
        
        logits = self.model(x_batch, self.static_graph, event_feat_batch) 
        
        loss = self.criterion(logits, y_batch)
        masked_loss = (loss * eval_mask_batch).sum() / eval_mask_batch.sum().clamp(min=1e-6)
        
        return masked_loss

    def train(self):
        print(f"Starting training on {self.device}...")
        
        for epoch in range(self.epochs):
            # --- Training Phase ---
            self.model.train()
            train_loss_accum = 0.0
            
            for batch in self.train_dataloader:
                self.optimizer.zero_grad()
                loss = self._process_batch(batch)
                loss.backward()
                self.optimizer.step()
                train_loss_accum += loss.item()
                
            avg_train_loss = train_loss_accum / len(self.train_dataloader)
            self.history['train_loss'].append(avg_train_loss)
            
            # --- Validation Phase ---
            self.model.eval()
            val_loss_accum = 0.0
            
            with torch.no_grad():
                for batch in self.val_dataloader:
                    loss = self._process_batch(batch)
                    val_loss_accum += loss.item()
            
            avg_val_loss = val_loss_accum / len(self.val_dataloader)
            self.history['val_loss'].append(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{self.epochs} | Train BCE: {avg_train_loss:.4f} | Val BCE: {avg_val_loss:.4f}")
            if self.use_wandb:
                wandb.log({
                    "train_loss": avg_train_loss, 
                    "val_loss": avg_val_loss, 
                    "epoch": epoch + 1
                })
        
        if self.use_wandb:
            wandb.finish()
        print("Training complete.")
        self.plot_losses()

    def plot_losses(self, save_path="loss_curve.png"):
        """Plots and saves the training and validation loss curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='Train Loss', marker='o')
        plt.plot(self.history['val_loss'], label='Validation Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Masked BCE Loss')
        plt.title('Training and Validation Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        print(f"Loss curve saved to {save_path}")



import pickle
import torch
from sklearn.model_selection import train_test_split
from preprocess import build_production_dataloader

def load_and_prepare_training_data(data_dir="data", batch_size=32, val_split=0.2, random_seed=42):
    """
    Loads the precomputed MCMC imitation data, splits it, and initializes the dataloader pipelines.
    """
    print("Loading imitation data from disk...")
    with open(f"{data_dir}/imitation_data.pkl", "rb") as f:
        imitation_dataset = pickle.load(f)

    print("Loading index mappings...")
    with open(f"{data_dir}/user_idx.pkl", "rb") as f:
        user_idx = pickle.load(f)
        
    with open(f"{data_dir}/event_idx.pkl", "rb") as f:
        event_idx = pickle.load(f)

    print(f"Splitting data with {val_split*100}% validation ratio...")
    train_data, val_data = train_test_split(
        imitation_dataset, 
        test_size=val_split, 
        random_state=random_seed
    )

    print("Building PyTorch Train DataLoader...")
    train_dataloader, static_graph = build_production_dataloader(
        imitation_dataset=train_data,
        user_idx=user_idx,
        event_idx=event_idx,
        data_dir=data_dir,
        batch_size=batch_size,
        shuffle=True
    )
    
    print("Building PyTorch Validation DataLoader...")
    # Shuffle is False for validation
    val_dataloader, _ = build_production_dataloader(
        imitation_dataset=val_data,
        user_idx=user_idx,
        event_idx=event_idx,
        data_dir=data_dir,
        batch_size=batch_size,
        shuffle=False
    )
    
    # static_graph from the val_dataloader is discarded as it should be identical 
    # to the one generated by the train_dataloader.
    return train_dataloader, val_dataloader, static_graph