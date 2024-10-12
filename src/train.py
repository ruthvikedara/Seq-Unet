import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import argparse

from model import UNet
from dataloader import MultiData, get_dataloaders
from losses import bce_ssim_loss, custom_loss

def train(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    train_dataloader, val_dataloader, _ = get_dataloaders(args.datapath, args.batch_size, args.block_rate)

    train_loss_list = []
    val_loss_list = []
    block_rate = args.block_rate
    alpha = args.alpha

    start_time = time.time()

    for epoch in range(args.epochs):
        print(f'Epoch: {epoch}')
        
        # Training
        model.train()
        training_loss = 0
        for data in tqdm(train_dataloader):
            image_file, mask_file = data
            image_file = image_file.to(device)
            mask_file = mask_file.float().to(device)

            optimizer.zero_grad()
            output = model(image_file)
            loss = bce_ssim_loss(output, mask_file.unsqueeze(1))
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        train_loss_list.append(training_loss / len(train_dataloader))

        # Save model
        torch.save(model.state_dict(), f'{args.save_dir}/U_net_model_{epoch}.pth')

        # Validation
        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for data in tqdm(val_dataloader):
                image_file, mask_file = data
                image_file = image_file.float().to(device)
                mask_file = mask_file.float().to(device)

                output = model(image_file)[0]
                loss = custom_loss(output, mask_file.unsqueeze(1))
                validation_loss += loss.item()

        val_loss_list.append(validation_loss / len(val_dataloader))

        print(f'Training loss: {train_loss_list[-1]}')
        print(f'Validation loss: {val_loss_list[-1]}')

        # Update block rate and alpha
        block_rate += args.block_rate_increment
        alpha += args.alpha_decrement

        # Update dataloaders
        train_dataloader, val_dataloader, _ = get_dataloaders(args.datapath, args.batch_size, block_rate)

    end_time = time.time()
    print(f'Training time: {end_time - start_time}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net model")
    parser.add_argument("--datapath", type=str, required=True, help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--block_rate", type=float, default=0.01, help="Initial block rate")
    parser.add_argument("--alpha", type=float, default=0.01, help="Initial alpha value")
    parser.add_argument("--block_rate_increment", type=float, default=0.03, help="Block rate increment per epoch")
    parser.add_argument("--alpha_increment", type=float, default=0.01, help="Alpha increment per epoch for boundary loss")
    parser.add_argument("--save_dir", type=str, default="./models/weights", help="Directory to save models")

    args = parser.parse_args()
    train(args)
