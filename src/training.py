import torch
import torch.nn as nn
import wandb

class ImitationTrainer:
    def __init__(self, model, dataloader, static_graph, config):
        self.model = model
        self.dataloader = dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Move static graph to device exactly once
        self.static_graph = {k: v.to(self.device) for k, v in static_graph.items()}
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 1e-3))
        
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.epochs = config.get('epochs', 10)
        
        wandb.init(
            project=config.get('project_name', "sidequest-imitation-learning"),
            name=config.get('run_name', model.__class__.__name__),
            config=config
        )
        wandb.watch(self.model, log="all", log_freq=10)

    def train(self):
        print(f"Starting training on {self.device}...")
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            
            # Unpack the 4-tuple yielded by the new collate function
            for batch_idx, (seed_mask_batch, y_batch, eval_mask_batch, event_feat_batch) in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                
                seed_mask_batch = seed_mask_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                eval_mask_batch = eval_mask_batch.to(self.device)
                event_feat_batch = event_feat_batch.to(self.device)
                
                # Construct [B, N, F+1] input
                B, N, _ = seed_mask_batch.shape
                expanded_features = self.static_graph['x_static'].unsqueeze(0).expand(B, N, -1)
                x_batch = torch.cat([seed_mask_batch, expanded_features], dim=-1)
                
                # Forward pass now includes event features
                logits = self.model(x_batch, self.static_graph, event_feat_batch) 
                
                # Masked Loss: Only evaluate nodes the MCMC simulator actually rolled out
                loss = self.criterion(logits, y_batch)
                masked_loss = (loss * eval_mask_batch).sum() / eval_mask_batch.sum().clamp(min=1e-6)
                
                masked_loss.backward()
                self.optimizer.step()
                
                epoch_loss += masked_loss.item()
                
            avg_loss = epoch_loss / len(self.dataloader)
            print(f"Epoch {epoch+1}/{self.epochs} | Masked BCE Loss: {avg_loss:.4f}")
            wandb.log({"train_loss": avg_loss, "epoch": epoch + 1})
            
        wandb.finish()
        print("Training complete.")



import pickle
import torch
from preprocess import build_production_dataloader

def load_and_prepare_training_data(data_dir="data", batch_size=32):
    """
    Loads the precomputed MCMC imitation data and initializes the dataloader pipeline.
    """
    print("Loading imitation data from disk...")
    with open(f"{data_dir}/imitation_data.pkl", "rb") as f:
        imitation_dataset = pickle.load(f)

    print("Loading index mappings...")
    with open(f"{data_dir}/user_idx.pkl", "rb") as f:
        user_idx = pickle.load(f)
        
    with open(f"{data_dir}/event_idx.pkl", "rb") as f:
        event_idx = pickle.load(f)

    print("Building PyTorch DataLoader...")
    dataloader, static_graph = build_production_dataloader(
        imitation_dataset=imitation_dataset,
        user_idx=user_idx,
        event_idx=event_idx,
        data_dir=data_dir,
        batch_size=batch_size,
        shuffle=True
    )
    
    return dataloader, static_graph