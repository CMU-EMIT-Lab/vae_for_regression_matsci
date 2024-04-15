from dataloader import TRSDataset, transform, transform_test
import torch
import tqdm
from model import VAE_Regression, VAE_Classic, loss_regression, loss_classic 
from utils import EarlyStop
import time
import glob
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
import warnings
import copy
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
    # Pulling datetime to create serial number
    sn = datetime.now()
    sn = sn.strftime("%Y%m%d%H%M%s")

    #Derived from https://github.com/QingyuZhao/VAE-for-Regression
    #https://arxiv.org/abs/1904.05948

    hyperparameters = {
        'px':[128],
        'bs':[128],
        'lr':[1e-4],
        'wd':[4e-5],
        'ls':[32]
        # 'ls': [8]
    }

    root_dir = 'EP_Data_90overlap_removed'
    files = glob.glob(f'{root_dir}/533_*/*.jpg', recursive=True)
#
    # root_dir = 'fatigue_900'
    # files = glob.glob(f'{root_dir}/*/*.png')

    for params in list(product(*hyperparameters.values())):

        model_type = "Regression"
        if model_type == "Regression":
            VAE = VAE_Regression
            loss = loss_regression
        else:
            VAE = VAE_Classic
            loss = loss_classic

        #Defines the data directory
        px, bs, lr, wd, ls = params
        
        data_dir = f"{root_dir}_{sn}_{model_type}_533Mpa_px{int(px)}_bs{int(bs)}_lr{int(lr*1e6)}x1e6_wd{int(wd*1e4)}x1e4_ls{int(ls)}"
        os.makedirs(data_dir)

        #Gets all of the property values
        nums = []
        for file in files:
            num = file.split('/')[1]
            num = int(num.split('_')[1])
            num = int(num)
            nums.append(num)

        #Determines mean and std (for normalization)
        predmean = np.mean(nums)
        predstd = np.std(nums)

        #Splits files to record training and testing
        files_train, files_test = train_test_split(files, test_size = 0.1, random_state=1)

        files_train = np.array(files_train)
        files_test = np.array(files_test)

        print(f'Loaded {len(files)} files')
        print(f'Split {len(files_train)} for training')
        print(f'Split {len(files_test)} for testing')

        np.save(f'{data_dir}/files_test.npy', files_test)
        np.save(f'{data_dir}/files_train.npy', files_train)

        train_data, train_iter, val_data, val_iter = dataloader(files_train, files_test, predmean, predstd, bs)

        # Visualizing the 9 images
        visualize_grid(train_iter, data_dir)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')

        #Initialize VAE Regression Model
        net = VAE(shape=(1, px, px), nhid = ls, device = device)
        net.to(device)
        # print(summary(net, (1,256,256), device = device))
        save_name = f"{data_dir}/VAE_regression.pt"

        #Learning rate
        #Faster adam optimizer
        print(lr)
        print(wd)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=wd)
        
        train_vae(data_dir, net, save_name, optimizer, train_data, train_iter, val_data, val_iter)
