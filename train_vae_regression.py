"""
Created by William J. Frieden Templeton & Justin P. Miner
Institution: Carnegie Mellon University
Date: April 24, 2024

Some notes up top: 
- Running "python3 train_vae_regression.py --test" will run the MNIST dataset as it is loaded in the folder.
- This script will create subdirectories wherever it is placed for model saving
- To visually watch progress, go to the subfolder and open "trial.jpg" to see the final image of the training iteration
"""

from dataloader import TRSDataset, transform, transform_test
from model import VAE_Regression, VAE_Classic, loss_regression, loss_classic 
from utils import EarlyStop

import torch
import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt
import warnings, argparse, copy, os, glob, time
from itertools import product
warnings.filterwarnings("ignore", category=UserWarning) 

# Visualize some input images
def visualize_grid(dataloader, data_dir, num_images=9):
    images, _ = next(iter(dataloader))
    images = images.numpy()

    # Set up the grid
    fig, axes = plt.subplots(3, 3, figsize=(9,9))
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            ax.imshow(images[i][0], cmap='gray')  # Assuming the images are grayscale
            ax.axis('off')
    plt.savefig(f"{data_dir}/Train_Input_example.jpg", dpi=100)

def dataloader(files_train, files_test, predmean, predstd, batch_size):
    #Dataloader  objet
    train_data = TRSDataset(files_train, transform, predmean, predstd)
    train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, num_workers = 4, pin_memory=True) #8 for 2 gpus

    val_data = TRSDataset(files_test, transform_test, predmean, predstd)
    val_iter = torch.utils.data.DataLoader(val_data, batch_size, shuffle=False, num_workers = 4, pin_memory=True) 
    return train_data, train_iter, val_data, val_iter

def train_vae(data_dir, net, save_name, optimizer, train_data, train_iter, val_data, val_iter):

    #Max epochs
    max_epochs = 1000
    #Early stopping criteria (patience defines number of epochs without decrease in loss)
    early_stop = EarlyStop(patience = 25, save_name = save_name)


    #Uncomment for multi GPU
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #   # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     net = torch.nn.DataParallel(net)
    # else:
    #     net = net.to(device)


    print("training on ", device)
    best_val_loss = float('inf')  # To keep track of best validation loss
    
    train_hist, val_hist = [], []

    
    for epoch in range(max_epochs):

        train_loss, n, start = 0.0, 0, time.time()

        # Training loop
        net.train()
        for X, r in tqdm.tqdm(train_iter, ncols=50):

            X = X.to(device)
            r = r.to(device)
            X_hat, z_mean, z_logvar, r_mean, r_logvar, z = net(X)

            l, lr, la, lk = loss(X, X_hat, z, z_mean, z_logvar, r, r_mean, r_logvar)
            l = l.to(device)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_loss += l.cpu().item()
            n += X.shape[0]

        train_loss /= n
        train_hist.append(copy.copy(train_loss))
        print('epoch %d, train loss %.4f , reco loss %.4f, accu loss %.4f, kl loss %.4f, time %.1f sec' % (epoch, train_loss, lr, la, lk, time.time() - start))


        # Convert the tensors to numpy arrays for visualization
        X_np = X.cpu().detach().numpy()[0][0] # Taking the first sample and channel for visualization
        X_hat_np = X_hat.cpu().detach().numpy()[0][0]

        # Assuming the images are grayscale and need to be normalized to [0, 1] for visualization
        X_np = 0.5 * (X_np + 1) # Normalize to [0, 1] if required
        X_hat_np = 0.5 * (X_hat_np + 1)

        # Plot original image
        fig = plt.figure(figsize=(10,5))
        plt.subplot(1, 2, 1)
        plt.imshow(X_np, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        # Plot reconstructed image
        plt.subplot(1, 2, 2)
        plt.imshow(X_hat_np, cmap='gray')
        plt.title('Reconstructed Image')
        plt.axis('off')
        plt.tight_layout()

        plt.savefig(f"{data_dir}/Trial.jpg",dpi=40)
        plt.close()

        # Validation loop
        val_loss, n = 0.0, 0
        net.eval()
        with torch.no_grad():
            for X_val, r_val in tqdm.tqdm(val_iter, ncols=50):
                X_val = X_val.to(device)
                r_val = r_val.to(device)

                X_hat_val, z_mean_val, z_logvar_val, r_mean_val, r_logvar_val, z_val = net(X_val)

                l_val, lr_val, la_val, lk_val = loss(X_val, X_hat_val, z_val, z_mean_val, z_logvar_val, r_val, r_mean_val, r_logvar_val)
                l_val = l_val.to(device)

                val_loss += l_val.cpu().item()
                n += X_val.shape[0]

        val_loss /= n
        
        print('------- val loss %.4f , reco loss %.4f, accu loss %.4f, kl loss %.4f, time %.1f sec' % (val_loss, lr_val, la_val, lk_val, time.time() - start))
        val_hist.append(copy.copy(val_loss))
        # Check early stopping condition
        if early_stop(train_loss, net, optimizer):
            print("Early stopping triggered.")
            break

    checkpoint = torch.load(early_stop.save_name)
    net.load_state_dict(checkpoint["net"])
    
    np.savetxt(f'{data_dir}/train_hist.txt', train_hist)
    np.savetxt(f'{data_dir}/val_hist.txt', val_hist)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VAE model with specified parameters and dataset")
    # parser.add_argument("--model", type=str, default="Regression", help="Type of VAE model: 'Regression' or 'Classic'")
    parser.add_argument("--help_option", action="store_true", help="Display help text")
    parser.add_argument("--test",action="store_true", help="Run Model on MNIST Dataset")
    parser.add_argument("--root_dir", type=str, default="your_data", help="Directory containing image files (Default is MNIST)")

    parser.add_argument("--ls", type=int, default=3, help="Size of latent space (Default is 3, recommend 32 for non-MNIST data)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate (Default is 1e-4)")
    parser.add_argument("--bs", type=int, default=128, help="Batch Size (Default is 128)")
    parser.add_argument("--px", type=int, default=128, help="Image Size (Default is 128)")
    
    args = parser.parse_args()
    
    if args.help_option:
        parser.print_help()
        exit()
    
    # Pulling datetime to create serial number
    sn = datetime.now()
    sn = sn.strftime("%Y%m%d%H%M%S")  # Changed %s to %S for correct second formatting
    
    hyperparameters = {
        'px':[args.px],
        'bs':[args.bs],
        'lr':[args.lr],
        'wd':[4e-5],
        'ls':[args.ls]
    }
    
    if args.test:
        print("Running in test mode with MNIST dataset.")
        # Load MNIST or handle it accordingly here
        # Setup MNIST specific parameters or defaults
        files = glob.glob(f'MNIST/*/*.png', recursive=True)  # This would be your dataset loading logic
        root_dir = "MNIST"
    else:
        print(f"Running with custom dataset from {args.root_dir}.")
        files = glob.glob(f'{args.root_dir}/*/*.png', recursive=True) #<------ FILE TYPE OF IMAGE
        root_dir = args.root_dir

    for params in list(product(*hyperparameters.values())):
        px, bs, lr, wd, ls = params
        print(f"image_size={px}\nbatch_size={bs} \nlearning_rate={lr} \nlatent_space={ls}")
    VAE = VAE_Regression
    loss = loss_regression

    data_dir = f"{root_dir}_{sn}_Regression_533Mpa_px{int(px)}_bs{int(bs)}_lr{int(lr*1e6)}x1e6_wd{int(wd*1e4)}x1e4_ls{int(ls)}"
    os.makedirs(data_dir, exist_ok=True)
    
    """ IMPORTANT
    So this nums is rather important, as it should be the property value we are regressing.
    The folder structure should be: DATA PARENT FOLDER / PROPERTY VALUE / IMAGES
    If the subfolders with property values are compltex (i.e., "MYDATA_4123MPA), alter nums HERE to extract just the number, AND nums in dataloader.py
    """ 
    # nums = [int(file.split('/')[1].split('_')[1]) for file in files]
    nums = [int(file.split('/')[1]) for file in files]

    #Determines mean and std (for normalization)
    predmean = np.mean(nums)
    predstd = np.std(nums)

    #Splits files to record training and testing
    files_train, files_test = train_test_split(files, test_size=0.1, random_state=1)
    print(f'Loaded {len(files)} files, split {len(files_train)} for training, {len(files_test)} for testing')

    np.save(f'{data_dir}/files_test.npy', files_test)
    np.save(f'{data_dir}/files_train.npy', files_train)

    train_data, train_iter, val_data, val_iter = dataloader(files_train, files_test, predmean, predstd, bs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = VAE(shape=(1, px, px), nhid=ls, device=device)
    net.to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=wd)
    train_vae(data_dir, net, f"{data_dir}/VAE_regression.pt", optimizer, train_data, train_iter, val_data, val_iter)
