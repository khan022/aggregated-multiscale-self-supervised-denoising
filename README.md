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
5. Some patches of different methods comparing with our method on three different datasets are provided demo_results folder.

### The abstract of the thesis (Aggregated Multiscale Self-Supervised Denoising)

In typical image denoising approaches, both supervised and unsupervised learning methods does not take account of individual image's particular image prior, the noise statistics, or both. The networks learned from external images inherently suffer from a domain gap problem as the image priors and noise statistics can be significantly different from the training and test images. So, it is difficult if the methods primarily requires clean images to train denoising. Furthermore, some images inherently generate significant noise (satellite images of distant galaxies, medical images like MRI images, CT scans, X-Ray images, etc.), and there are no clean images for training. Here the problems dominantly lie with the data delivery system. Our approach takes the noisy images and creates a new version of them with specific pre-processing; by doing so, we make the target pseudo clear image for the deep neural network. We generate multiple versions of these noisy images using interpolation of arrays and train the network to the extent where the network can learn the information on the images without the noises. In practice, the noisy pictures are blurred on three different scales. These blurred versions and the original noisy images are combined together to create a single set. This set captures all the necessary information from all of the four groups. The network architecture uses the concatenation of the module concept to learn the clear images from a versatile perspective. Then we trained the model using the main noisy set as input and the newly created set as the target to predict a much cleaner image from a regular noisy image. This method creates an output image where the structural integrity image is sustained and the noise component is removed from the image.

### The link for the full thesis is http://www.riss.kr/link?id=T16658595.

## Contact

For questions or help with using the code in this repository, please contact shafkat.kh022@gmail.com.
