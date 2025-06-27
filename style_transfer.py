import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import random
from tqdm import tqdm
# import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(input_nc, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)

class StyleTransferDataset(Dataset):
    def __init__(self, busi_dir, bus_uclm_dir, transform=None, mode='train'):
        self.busi_dir = busi_dir
        self.bus_uclm_dir = bus_uclm_dir
        self.transform = transform
        self.mode = mode
        
        # Load BUSI data
        self.busi_frame = pd.read_csv(os.path.join(busi_dir, f'{mode}_frame.csv'))
        
        # Load BUS-UCLM data
        self.bus_uclm_frame = pd.read_csv(os.path.join(bus_uclm_dir, f'{mode}_frame.csv'))
        
        print(f"BUSI {mode} samples: {len(self.busi_frame)}")
        print(f"BUS-UCLM {mode} samples: {len(self.bus_uclm_frame)}")

    def __len__(self):
        return min(len(self.busi_frame), len(self.bus_uclm_frame))

    def __getitem__(self, idx):
        # Load BUSI image
        busi_image_path = self.busi_frame.image_path.iloc[idx]
        busi_type = busi_image_path.split()[0]
        busi_img_path = os.path.join(self.busi_dir, busi_type, 'image', busi_image_path)
        
        # Load BUS-UCLM image
        bus_uclm_image_path = self.bus_uclm_frame.image_path.iloc[idx]
        bus_uclm_type, bus_uclm_fn = bus_uclm_image_path.split(maxsplit=1)
        bus_uclm_img_path = os.path.join(self.bus_uclm_dir, bus_uclm_type, 'images', bus_uclm_fn)
        
        busi_img = Image.open(busi_img_path).convert('L')
        bus_uclm_img = Image.open(bus_uclm_img_path).convert('L')
        
        if self.transform:
            busi_img = self.transform(busi_img)
            bus_uclm_img = self.transform(bus_uclm_img)
        
        return busi_img, bus_uclm_img

class CycleGAN:
    def __init__(self, device='cuda'):
        self.device = device
        
        # Initialize networks
        self.G_AB = Generator().to(device)  # BUS-UCLM -> BUSI
        self.G_BA = Generator().to(device)  # BUSI -> BUS-UCLM
        self.D_A = Discriminator().to(device)  # Discriminator for BUSI
        self.D_B = Discriminator().to(device)  # Discriminator for BUS-UCLM
        
        # Loss functions
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        
        # Optimizers
        self.optimizer_G = optim.Adam(
            list(self.G_AB.parameters()) + list(self.G_BA.parameters()),
            lr=0.0002, betas=(0.5, 0.999)
        )
        self.optimizer_D_A = optim.Adam(self.D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D_B = optim.Adam(self.D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def train(self, dataloader, epochs=100, save_interval=10):
        for epoch in range(epochs):
            for i, (real_A, real_B) in enumerate(tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')):
                real_A = real_A.to(self.device)
                real_B = real_B.to(self.device)
                
                # Train generators
                self.optimizer_G.zero_grad()
                
                # Identity loss
                loss_id_A = self.criterion_identity(self.G_BA(real_A), real_A)
                loss_id_B = self.criterion_identity(self.G_AB(real_B), real_B)
                loss_identity = (loss_id_A + loss_id_B) / 2
                
                # GAN loss
                fake_B = self.G_AB(real_A)
                loss_GAN_AB = self.criterion_GAN(self.D_B(fake_B), torch.ones_like(self.D_B(fake_B)))
                
                fake_A = self.G_BA(real_B)
                loss_GAN_BA = self.criterion_GAN(self.D_A(fake_A), torch.ones_like(self.D_A(fake_A)))
                
                # Cycle loss
                recov_A = self.G_BA(fake_B)
                loss_cycle_A = self.criterion_cycle(recov_A, real_A)
                
                recov_B = self.G_AB(fake_A)
                loss_cycle_B = self.criterion_cycle(recov_B, real_B)
                
                # Total generator loss
                loss_G = loss_GAN_AB + loss_GAN_BA + 10 * loss_cycle_A + 10 * loss_cycle_B + 5 * loss_identity
                loss_G.backward()
                self.optimizer_G.step()
                
                # Train discriminators
                self.optimizer_D_A.zero_grad()
                loss_D_A = (self.criterion_GAN(self.D_A(real_A), torch.ones_like(self.D_A(real_A))) +
                           self.criterion_GAN(self.D_A(fake_A.detach()), torch.zeros_like(self.D_A(fake_A.detach())))) / 2
                loss_D_A.backward()
                self.optimizer_D_A.step()
                
                self.optimizer_D_B.zero_grad()
                loss_D_B = (self.criterion_GAN(self.D_B(real_B), torch.ones_like(self.D_B(real_B))) +
                           self.criterion_GAN(self.D_B(fake_B.detach()), torch.zeros_like(self.D_B(fake_B.detach())))) / 2
                loss_D_B.backward()
                self.optimizer_D_B.step()
                
                if i % 100 == 0:
                    print(f'[{epoch}/{epochs}][{i}/{len(dataloader)}] '
                          f'Loss_D: {loss_D_A.item():.4f}/{loss_D_B.item():.4f} '
                          f'Loss_G: {loss_G.item():.4f}')
            
            # Save models
            if (epoch + 1) % save_interval == 0:
                self.save_models(f'style_transfer_epoch_{epoch+1}.pth')
    
    def save_models(self, filename):
        torch.save({
            'G_AB_state_dict': self.G_AB.state_dict(),
            'G_BA_state_dict': self.G_BA.state_dict(),
            'D_A_state_dict': self.D_A.state_dict(),
            'D_B_state_dict': self.D_B.state_dict(),
        }, filename)
    
    def load_models(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.G_AB.load_state_dict(checkpoint['G_AB_state_dict'])
        self.G_BA.load_state_dict(checkpoint['G_BA_state_dict'])
        self.D_A.load_state_dict(checkpoint['D_A_state_dict'])
        self.D_B.load_state_dict(checkpoint['D_B_state_dict'])
    
    def transfer_style(self, bus_uclm_image):
        """Transfer BUS-UCLM image to BUSI style"""
        self.G_AB.eval()
        with torch.no_grad():
            return self.G_AB(bus_uclm_image)

def main():
    # Configuration
    busi_dir = "dataset/BioMedicalDataset/BUSI"
    bus_uclm_dir = "dataset/BioMedicalDataset/BUS-UCLM"
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((352, 352)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Create dataset and dataloader
    dataset = StyleTransferDataset(busi_dir, bus_uclm_dir, transform, mode='train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    
    # Initialize CycleGAN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cyclegan = CycleGAN(device)
    
    # Train the model
    print("Starting style transfer training...")
    cyclegan.train(dataloader, epochs=50, save_interval=10)
    
    print("Style transfer training completed!")

if __name__ == "__main__":
    import pandas as pd
    main() 