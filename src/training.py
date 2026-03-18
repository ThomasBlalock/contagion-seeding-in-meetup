# imitation_pretraining.py
# TODO: boilerplate code - fix as needed

import torch
import torch.nn as nn
import wandb

class ImitationTrainer:
    def __init__(self, model, dataloader, config):
        """
        config should contain: lr, epochs, project_name, run_name
        """
        self.model = model
        self.dataloader = dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 1e-3))
        
        # KLDivLoss is mathematically correct for fitting probability distributions
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        
        self.epochs = config.get('epochs', 10)
        
        # Automatically spin up W&B
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
            
            for batch_idx, (state, target_dist) in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                
                # Move all multiplex tensors to device
                node_feats = state["node_features"].to(self.device)
                edge_index_1 = state["edge_index_1"].to(self.device)
                edge_attr_1 = state["edge_attr_1"].to(self.device)
                edge_index_2 = state["edge_index_2"].to(self.device)
                edge_attr_2 = state["edge_attr_2"].to(self.device)
                event_feats = state["event_features"].to(self.device)
                seed_mask = state["seed_mask"].to(self.device)
                
                target = target_dist.to(self.device)
                
                # Forward pass using the new multiplex API contract
                log_probs = self.model(
                    node_features=node_feats,
                    edge_index_1=edge_index_1,
                    edge_attr_1=edge_attr_1,
                    edge_index_2=edge_index_2,
                    edge_attr_2=edge_attr_2,
                    event_features=event_feats,
                    seed_mask=seed_mask
                )
                
                # Calculate loss and backprop
                loss = self.criterion(log_probs, target)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(self.dataloader)
            print(f"Epoch {epoch+1}/{self.epochs} | KL Loss: {avg_loss:.4f}")
            wandb.log({"train_loss": avg_loss, "epoch": epoch + 1})
            
        wandb.finish()
        print("Training complete.")