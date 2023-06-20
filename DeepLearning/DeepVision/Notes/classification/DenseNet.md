# DenseNet

## Densely Connected Convolutional Networks (DenseNet) by Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger

![DenseNet](images/0602.jpeg)

## Motivation

- The main idea of DenseNet is to connect each layer to every other layer in a feed-forward fashion.

- For each layer, the feature-maps of all preceding layers are used as inputs, and its own feature-maps are used as inputs into all subsequent layers. Hence total number of connections in a `L` Network is `L(L+1)/2`.

- DenseNets have several compelling advantages: they alleviate the `vanishing-gradient problem`, `strengthen feature propagation`, `encourage feature reuse`, and `substantially reduce the number of parameters`.

- In contrast to ResNet, DenseNet don't combine the feature-maps through summation, but through `concatenation`.

## History of Increasing Network's Depth

- `Skip-Connections` are used to increase the depth of the network. But the main problem with skip-connections is that there many layers contribute very little and can be safely removed without hurting the performance. This makes The `ResNet` similar to unrolled RNN but the number of parameters are substantially because each layer has its own weights.

- `Stochastic Depth` is used to address this problem. In this method, layers are dropped randomly during training. This shows that not all the layers may be needed and highlights that there is a great amount of redundancy in deep (Residual) networks.

- An orthogonal approach to making networks deeper (with the help of skip-connections) is to increase the network width. The GoogLeNet uses an `Inception module` which concatenates feature-maps produced by filters of different sizes.

- In `DenseNet`, authors `concatenate the feature-maps` of all layers this differentiates between the informaation that is added to the Network and the information that is preserved. Instead of drawing representational power from extremely deep or wide architectures, DenseNets exploit the potential of the network through `feature reuse`.

![DenseNet](images/0601.jpeg)

## Terminology

**Growth Rate**: This determines the number of features maps output into individual layers inside dense block.

**Dense Connectivity**: By Dense connectivity, authors mean that each layer is connected to every other layer in a feed-forward fashion.

**Composite Function**: Each layer in the dense block is a composite function of `Batch Normalization`, `ReLU`, and `Convolution` this will be one convolution layer.

**Transition Layer**: It aggregates the feature maps from a dense blocka and reduce its dimensions. So Max Pooling is enabled in this layer

# Architecture

![Architecture](images/0603.jpeg)

The network consists of 3 parts:

- **Dense Blocks**: Each dense block consists of multiple dense layers. Each dense layer has a composite function of `Batch Normalization`, `ReLU`, and `Convolution`. The input of each dense layer is a concatenation of all the feature maps of the previous layers. The output of each dense layer is fed into the next dense layer. The feature maps of all layers are concatenated again and fed into the next dense layer. This process is repeated until the last dense layer in the block.

- **Transition Layers**: The transition layers are used to reduce the number of feature maps. The transition layers consist of a `Batch Normalization`, `1x1 Convolution`, followed by `2x2 Average Pooling` layer. The number of feature maps in the transition layer is reduced by a factor of `θ` (referred to as the compression factor) compared to the previous dense block. The transition layers are used to control the growth of the number of feature maps and to reduce the model complexity.

- **Classification Layer**: The global average pooling is used to reduce the number of parameters in the model. The output of the global average pooling is fed into a `softmax` layer for classification.

## DenseNetC and DenseNetBC

- If a dense block contains `m feature-maps`, we let the following `transition layer` generate $thetam$ output featuremaps, where `0 <θ ≤1` is referred to as the `compression factor`. When `θ = 1`, the number of feature-maps across transition layers remains unchanged. We refer the DenseNet with `θ <1` as `DenseNet-C`, and we set `θ = 0.5` in our experiment. When both the `bottleneck and transition layers with θ < 1` are used, we refer to our model as `DenseNet-BC`.


## De



# My Observation:

- DenseNet is Exactly same as that of ResNet only difference is that in ResNet we `add` the input of current layer to the output of current layer but in DenseNet we `concatenate` the input of current layer to the output of current layer. 

- Because the input of a particular layer is already concatenated with the output of all the previous layers.