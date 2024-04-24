from torch.utils.data import Dataset
import PIL
from torchvision.transforms import v2
import torch
import cv2 as cv
import numpy as np

class TRSDataset(Dataset):

    def __init__(self, filelist, transform, predmean = 0.0, predstd = 1.0):
        #predmean and pred std are the prediction mean and prediction std used for normalization
        #transform is the list of transformations applied
        #filelist is the list of files to use (need to patch before)
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.transform = transform
        self.file_list = filelist
        self.data = []
        self.predmean = predmean
        self.predstd = predstd
        
        #extracts properties from each file
        #also passes over images not in patches
        count = 0
        for file in self.file_list:
            num = file.split('/')[1]
            # num = int(num.split('_')[1])
            num = int(num)
            num = (num - predmean)/predstd
            if file.startswith('.'):
                continue
            else:
                self.data.append([file, num])

        
        self.im_dim = (128,128)

    def __len__(self):
        return len(self.data)

    #Makes each image grayscale and transforms
    def __getitem__(self, idx):
        img_path, n = self.data[idx]
        image = PIL.Image.open(img_path)

        # Binarize Porosity Images
        image = image.convert('L')
        image = np.array(image) 
        blur = cv.GaussianBlur(image,(3,3),0)
        ret, thresh = cv.threshold(blur, 150, 255, cv.THRESH_BINARY)
        # thresh = cv.adaptiveThreshold(thresh,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 51, 12)
        image = PIL.Image.fromarray(thresh)
        # image = PIL.ImageOps.invert(image)
        image = image.convert('1')
        

        image = PIL.ImageOps.grayscale(image)
        img_tensor = self.transform(image)
        return img_tensor, n

#This defines the transformation functions for each model (VAE regression has no extension)
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale = True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    # v2.Normalize([0.5], [0.5]),
    v2.Resize([128, 128]),
])

transform_test = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale = True),
    # v2.Normalize([0.5], [0.5]),
    v2.Resize([128, 128])
])

transform_cnn_test = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale = True),
    v2.Normalize([0.5], [0.5])
])

transform_cnn = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale = True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    # v2.Normalize([0.5], [0.5]),
    v2.RandomAffine(degrees=45),
    # v2.Resize((256,256))
    #v2.GaussianBlur(kernel_size=(5,5), sigma = (0.001,2)),
    # v2.ColorJitter(brightness = 0.5, contrast = 0.25)
])
