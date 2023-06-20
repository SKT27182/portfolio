import tensorflow.keras.datasets as tfds
import tensorflow.keras.utils as tfku

class Datasets:

    def __init__(self):
        pass

    def load_mnist(self):
        (x_train, y_train), (x_test, y_test) = tfds.mnist.load_data()

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # reshape the data to 4D tensor
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

        y_train = tfku.to_categorical(y_train, 10)
        y_test = tfku.to_categorical(y_test, 10)

        return (x_train, y_train), (x_test, y_test)
    
    def load_cifar10(self):
        (x_train, y_train), (x_test, y_test) = tfds.cifar10.load_data()

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # reshape the data to 4D tensor
        x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
        x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)

        y_train = tfku.to_categorical(y_train, 10)
        y_test = tfku.to_categorical(y_test, 10)

        return (x_train, y_train), (x_test, y_test)
    
    def load_cifar100(self):
        (x_train, y_train), (x_test, y_test) = tfds.cifar100.load_data()

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # reshape the data to 4D tensor
        x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
        x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)

        y_train = tfku.to_categorical(y_train, 100)
        y_test = tfku.to_categorical(y_test, 100)

        return (x_train, y_train), (x_test, y_test)
    
    def load_fashion_mnist(self):
        (x_train, y_train), (x_test, y_test) = tfds.fashion_mnist.load_data()

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # reshape the data to 4D tensor
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

        y_train = tfku.to_categorical(y_train, 10)
        y_test = tfku.to_categorical(y_test, 10)

        return (x_train, y_train), (x_test, y_test)
    
    def load_imdb(self):
        (x_train, y_train), (x_test, y_test) = tfds.imdb.load_data()

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # reshape the data to 4D tensor
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

        y_train = tfku.to_categorical(y_train, 10)
        y_test = tfku.to_categorical(y_test, 10)

        return (x_train, y_train), (x_test, y_test)
    
    def load_dataset(self, requsted_dataset):
        if requsted_dataset == "mnist":
            return self.load_mnist()
        elif requsted_dataset == "cifar10":
            return self.load_cifar10()
        elif requsted_dataset == "cifar100":
            return self.load_cifar100()
        elif requsted_dataset == "fashion_mnist":
            return self.load_fashion_mnist()
        elif requsted_dataset == "imdb":
            return self.load_imdb()
        else:
            raise Exception("Dataset not found")
    
