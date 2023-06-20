# DeepVision

Implementing Computer Vision Architectures from Scratch using TensorFlow or PyTorch

## Introduction

This repository contains implementations of various Computer Vision Architectures from scratch using TensorFlow or PyTorch. The implementations are done in a modular fashion, so that the individual components of the architectures can be used in other projects.

## Architectures

## Classification Models

###  [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf): This is the first Convolutional Neural Network (CNN) architecture proposed by Yann LeCun in 1998. It was used to classify handwritten digits. The architecture consists of three convolutional layers, two subsampling layers, and three fully connected layers. For more details, refer to the [LeNet](Notes/classification/LeNet.md) notes.

---

###  [AlexNet](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf): This is the first CNN architecture to win the ImageNet competition in 2012. It was proposed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton. The architecture consists of five convolutional layers, three pooling layers, and two fully connected layers. For more details, refer to the [AlexNet](Notes/classification/AlexNet.md) notes.

---

### [VGG](https://arxiv.org/pdf/1409.1556v6.pdf): This is the first CNN architecture to use very deep convolutional layers. It was proposed by Karen Simonyan and Andrew Zisserman. The architecture consists of 16 convolutional layers, 5 pooling layers, and 3 fully connected layers. For more details, refer to the [VGG](Notes/classification/VGG.md) notes.

---

### [ResNet](https://arxiv.org/pdf/1512.03385.pdf): This is the first CNN architecture to use residual connections. It was proposed by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. The architecture consists of 18 convolutional layers, 5 pooling layers, and 3 fully connected layers. For more details, refer to the [ResNet](Notes/classification/ResNet.md) notes.

---

### [Inception](https://arxiv.org/pdf/1409.4842v1.pdf): This is the first CNN architecture to use inception modules. It was proposed by Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, and Jonathon Shlens. The architecture consists of 22 convolutional layers, 5 pooling layers, and 3 fully connected layers. For more details, refer to the [Inception](Notes/classification/Inception.md) notes.

---

### [Densenet](https://arxiv.org/pdf/1608.06993v5.pdf): This is the first CNN architecture to use dense blocks. It was proposed by Gao Huang, Zhuang Liu, Laurens van der Maaten, and Kilian Q. Weinberger. The architecture consists of 121 convolutional layers, 5 pooling layers, and 3 fully connected layers. For more details, refer to the [Densenet](Notes/classification/DenseNet.md) notes.

---

### [Xception](https://arxiv.org/pdf/1610.02357v3.pdf): This is the first CNN architecture to use depthwise separable convolutions. It was proposed by François Chollet. The architecture consists of 36 convolutional layers, 5 pooling layers, and 3 fully connected layers. For more details, refer to the [Xception](Notes/classification/Xception.md) notes.

---

### [MobileNets](https://arxiv.org/pdf/1704.04861v1.pdf): This is the first CNN architecture to use depthwise separable convolutions. It was proposed by Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. The architecture consists of 28 convolutional layers, 5 pooling layers, and 3 fully connected layers. For more details, refer to the [MobileNet](Notes/classification/MobileNets.md) notes.


---

### [EfficientNet](https://arxiv.org/pdf/1905.11946v5.pdf): This is the first CNN architecture to use compound scaling. It was proposed by Mingxing Tan and Quoc V. Le. The architecture consists of 28 convolutional layers, 5 pooling layers, and 3 fully connected layers. For more details, refer to the [EfficientNet](Notes/classification/EfficientNet.md) notes.

---
---

## Generative Models

### [NeuralStyleTransfer](https://arxiv.org/pdf/1508.06576.pdf): This is a technique that is used to transfer the style of one image onto another image. It was proposed by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge. For more details, refer to the [NeuralStyleTransfer](Notes/generative/NeuralStyleTransfer.md) notes.

---

### [AutoEncoder](https://www.science.org/doi/pdf/10.1126/science.1127647): This is a neural network architecture that is used to learn efficient data encodings in an unsupervised manner. It was proposed by Geoffrey Hinton and his students at the University of Toronto. The architecture consists of two parts: an encoder and a decoder. The encoder learns to compress the input data into a lower dimensional representation, and the decoder learns to reconstruct the input data from the lower dimensional representation. For more details, refer to the [AutoEncoder](Notes/generative/Autoencoder.md) notes.

---

### [Variational AutoEncoder](https://arxiv.org/pdf/1312.6114v10.pdf): This is a neural network architecture that is used to learn efficient data encodings in an unsupervised manner. It was proposed by Diederik P Kingma and Max Welling. The architecture consists of two parts: an encoder and a decoder. The encoder learns to compress the input data into a lower dimensional representation, and the decoder learns to reconstruct the input data from the lower dimensional representation. The encoder is trained using a variational inference technique. For more details, refer to the [Variational AutoEncoder](Notes/generative/VariationalAutoencoder.md) notes.

---

### [Generative Adversarial Network](https://arxiv.org/pdf/1406.2661.pdf): This is a neural network architecture that is used to learn efficient data encodings in an unsupervised manner. It was proposed by Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. The architecture consists of two parts: a generator and a discriminator. The generator learns to generate data that is similar to the input data, and the discriminator learns to distinguish between the input data and the generated data. For more details, refer to the [Generative Adversarial Network](Notes/generative/GANs.md) notes.

---
---

## Object Detection Models

### [U-Net](https://arxiv.org/pdf/1505.04597v1.pdf): This is a neural network architecture that is used for image segmentation. It was proposed by Olaf Ronneberger, Philipp Fischer, and Thomas Brox. The architecture consists of an encoder and a decoder. The encoder learns to compress the input data into a lower dimensional representation, and the decoder learns to reconstruct the input data from the lower dimensional representation. For more details, refer to the [U-Net](Notes/detection/U-Net.md) notes.