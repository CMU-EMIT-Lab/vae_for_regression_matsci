# Variational Autoencoder with Regression for Materials Science


Pytorch Implementation for Variational Autoencoder with Regression for Materials Science

This work was inspired by https://github.com/QingyuZhao/VAE-for-Regression 



## Installing

Clone the repository on your local machine
```shell
git clone https://github.com/CMU-EMIT-Lab/vae_for_regression_matsci.git
```
Install the required packages. For these packages to work, a cuda version of 12.1+ is required. You can check this by running, 

```shell
nvcc --version
```

 It is recommended to proceed in a new environment first. You can do that using

```shell
conda create --name vae_matsci python=3.10.12
conda activate vae_matsci
pip install -r requirements.txt
```



## Data Labeling/Reading

Inside of the data subdirectory, folders should be named according to the property value. Within the folders, image patches should be placed as a 128x128 image file. To modify the glob string where data is read in, you will need to change the following variables

In dataloader.py, in the self.file_list variable you will have to change how the numbers are read in

In train_vae_regression.py, you will need to change the files variable

Glob strings are very similar to regex. To learn the modifier characters, we recommend you look at this link: https://www.malikbrowne.com/blog/a-beginners-guide-glob-patterns/



## Data Augmentation

Data augmentation strategies can be found in the dataloader.py file. Note that there are different transformations for training and testing.



## Training your own model


You can then train a new model by running

```shell
python train_vae_regression.py
```



## Evaluating your own model

To evaluate the model, you can run the jupyter notebook files eval_regression.ipynb, heatmap.ipynb, and interpolation.ipynb. Note, you will have to change the glob strong as was done with the data reading step and you will have to change the name of the model file.



## Reference

Please use the following reference if you utilize this code.

```
@article{templetonminer2024vae,
  title={Expediting Structure-Property Analyses using Variational Autoencoders with Regression},
  author={Frieden Templeton, William and Miner, Justin P. and Ngo, Austin and Fitzwater, Lauren and Reddy, Tharun and Abranovic, Brandon and Prichard, Paul and Lewandowski, John J. and Narra, Sneha Prabha},
  journal={Computational Materials Science},
  volume={},
  pages={},
  year={2024},
  publisher={Elsevier}
}
