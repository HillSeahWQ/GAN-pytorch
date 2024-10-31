# Deep Convolutional Generative Adversarial Networks (DCGAN)

## Project Overview
Implemenation of DCGAN - a relatively primitive Generative Adversarial Network (GAN).

Trained on the [celebrity dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) for 200 epochs. The checkpoint models for the Generator, Discriminator and their respective optimizers have been saved in the models folder.

### Papers:

[DCGAN](https://arxiv.org/abs/1511.06434)

---

### Setup Prerequisites

1. **Install Python 3.12.3**
2. **Install Poetry**
3. **Install Nvidia CUDA 12.1**
   - Note: The version of PyTorch in this project uses CUDA 12.1 for GPU computing.

---

### Steps to Run
1. **Clone this repository**
2. **Install Dependencies:**

   ```bash
   poetry install
   ```
3. **Enter the virtual environment:**

   ```bash
   poetry shell
   ```
4. **Training**
- Default
   ```bash
    python train.py --device <DEVICE> --dataset_path <DATASET_PATH> --epochs <EPOCHS> --img_channels <IMG_CHANNELS>
    ```
- (Optional) Customising Training Hyperparameters
    ```bash
    python train.py --device <DEVICE> --dataset_path <DATASET_PATH> --run_name <RUN_NAME> --epochs <EPOCHS> --batch_size <BATCH_SIZE> --img_size <IMG_SIZE> --img_channels <IMG_CHANNELS> --z_dim <Z_DIM> --lr <LEARNING_RATE> --b1 <BETA1>
    ```

--- 
### Training Arguments

- --device: Specify the device for training *("cuda" for GPU, "cpu" for CPU).*
- --dataset_path: Path to the dataset for training. *(Default: \<YOUR_OWN_DATASET_PATH>)*
- --run_name: A name for this training run (for logging). *(Default: `"DCGAN")*
- --epochs: Number of training epochs. *(Default: 500)*
- --batch_size: Batch size for training. *(Default: 128)*
- --img_size: Resolution of input images. *(Default: 64)*
- --img_channels: Number of color channels in images. *(Default: 3 for RGB images)*
- --z_dim: Dimension of the noise vector. *(Default: 100)*
- --lr: Learning rate for optimizers. *(Default: 2e-4)*
- --b1: Beta1 hyperparameter for Adam optimizer. *(Default: 0.5)*

## Results
