# train.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SRDataset
from model import SRCNN
from utils import calculate_psnr, save_checkpoint
from tqdm import tqdm
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

class EarlyStopping:
    """Early stops the training if validation PSNR doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation PSNR improved.
            verbose (bool): If True, prints a message for each validation PSNR improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_psnr = -float('inf')

    def __call__(self, val_psnr):
        score = val_psnr

        if self.best_score is None:
            self.best_score = score
            if self.verbose:
                print(f'Initial validation PSNR set to {self.best_score:.4f} dB')
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'No improvement in PSNR for {self.counter} consecutive epochs.')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print('Early stopping triggered.')
        else:
            if self.verbose:
                print(f'Validation PSNR improved from {self.best_score:.4f} dB to {score:.4f} dB. Resetting counter.')
            self.best_score = score
            self.counter = 0

def parse_args():
    parser = argparse.ArgumentParser(description='Train SRCNN for Image Super-Resolution')
    parser.add_argument('--train_dir', type=str, default='DIV2K_train_HR/', help='Directory with training HR images')
    parser.add_argument('--valid_dir', type=str, default='DIV2K_valid_HR/', help='Directory with validation HR images')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')  # Limited to 100
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--scale_factor', type=int, default=2, help='Scale factor for super-resolution')
    parser.add_argument('--patch_size', type=int, default=32, help='Patch size for training')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/', help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (cuda or cpu)')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--delta', type=float, default=0.0, help='Minimum change in PSNR to qualify as improvement')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output for early stopping')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create checkpoint directory if not exists
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Device configuration
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data transformations
    transform = ToTensor()
    
    # Datasets and Dataloaders
    train_dataset = SRDataset(
        hr_dir=args.train_dir, 
        patch_size=args.patch_size, 
        scale_factor=args.scale_factor, 
        transform=transform
    )
    valid_dataset = SRDataset(
        hr_dir=args.valid_dir, 
        patch_size=args.patch_size, 
        scale_factor=args.scale_factor, 
        transform=transform
    )
    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    # Model, Loss, Optimizer
    model = SRCNN(num_channels=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=args.patience, verbose=args.verbose, delta=args.delta)
    
    # Training variables
    train_psnr_history = []
    valid_psnr_history = []
    
    # Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_psnr = 0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{args.epochs}', unit='batch') as pbar:
            for batch in train_loader:
                lr, hr = batch
                lr = lr.to(device)
                hr = hr.to(device)
                
                # Forward pass
                sr = model(lr)
                loss = criterion(sr, hr)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_psnr += calculate_psnr(sr, hr)
                
                pbar.set_postfix({'Loss': loss.item(), 'PSNR': calculate_psnr(sr, hr)})
                pbar.update(1)
        
        avg_loss = epoch_loss / len(train_loader)
        avg_psnr = epoch_psnr / len(train_loader)
        train_psnr_history.append(avg_psnr)
        
        # Validation
        model.eval()
        val_psnr = 0
        with torch.no_grad():
            for batch in valid_loader:
                lr, hr = batch
                lr = lr.to(device)
                hr = hr.to(device)
                
                sr = model(lr)
                psnr = calculate_psnr(sr, hr)
                val_psnr += psnr
        
        avg_val_psnr = val_psnr / len(valid_loader)
        valid_psnr_history.append(avg_val_psnr)
        
        print(f'Epoch [{epoch}/{args.epochs}] - Train PSNR: {avg_psnr:.2f} dB - Val PSNR: {avg_val_psnr:.2f} dB')
        
        # Check Early Stopping
        early_stopping(avg_val_psnr)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break
        
        # Save checkpoint if it's the best so far
        if avg_val_psnr > early_stopping.best_psnr:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'srcnn_best.pth')
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=checkpoint_path)
            print(f'Best model saved at epoch {epoch} with Val PSNR: {avg_val_psnr:.2f} dB')
    
    # Plot PSNR history
    plt.figure(figsize=(10,5))
    plt.plot(range(1, len(train_psnr_history) +1), train_psnr_history, label='Train PSNR')
    plt.plot(range(1, len(valid_psnr_history) +1), valid_psnr_history, label='Validation PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.title('Training and Validation PSNR over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.checkpoint_dir, 'psnr_history.png'))
    plt.show()
    
    print('Training Completed.')

if __name__ == '__main__':
    main()
