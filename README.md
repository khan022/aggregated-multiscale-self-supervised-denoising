# aggregated-multiscale-self-supervised-denoising
# Aggregated Multiscale Self-Supervised Denoising

This repository contains the code for a self-supervised image denoising training procedure, testing, and some sample images, developed as part of a master's degree thesis.

## Training

To train and run the code in this repository, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies (provided in the environment.yml).
3. Run the code using the instructions provided below.

## Usage

To use the code in this repository, follow these steps:

1. In order to use this code the location of the regular image has to be provided to noisy_data_creation.npy
2. For target image creation the location of the noisy image has to be provided to target_data_ceration.npy
3. The creating_npy_for_training.py needs both noisy images and target images locations.
4. The training is done using gradient tape of tensorflow with custom optimzer and custom loss functions.
5. Some demo images of testing is provided here.
6. The link for this thesis is http://www.riss.kr/link?id=T16658595.

## Contact

For questions or help with using the code in this repository, please contact shafkat.kh022@gmail.com.
