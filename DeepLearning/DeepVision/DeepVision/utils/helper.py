import tensorflow as tf

def losses(loss):
    if loss == "mse":
        return tf.keras.losses.MeanSquaredError()
    elif loss == "mae":
        return tf.keras.losses.MeanAbsoluteError()
    elif loss == "mape":
        return tf.keras.losses.MeanAbsolutePercentageError()
    elif loss == "msle":
        return tf.keras.losses.MeanSquaredLogarithmicError()
    elif loss == "binary_crossentropy":
        return tf.keras.losses.BinaryCrossentropy()
    elif loss == "categorical_crossentropy":
        # requires one-hot encoded labels
        return tf.keras.losses.CategoricalCrossentropy()
    elif loss == "sparse_categorical_crossentropy":
        # requires integer encoded labels
        return tf.keras.losses.SparseCategoricalCrossentropy()
    else:
        raise ValueError("Invalid loss function")
    
def metrics(metric):
    if metric not in ["accuracy", "precision", "recall", "auc", "f1_score"]:
        raise ValueError("Invalid metric")
    
    return metric
        
    
def optimizers(optimizer, learning_rate):
    if optimizer == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer == "adagrad":
        return tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif optimizer == "adadelta":
        return tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
    elif optimizer == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "adamax":
        return tf.keras.optimizers.Adamax(learning_rate=learning_rate)
    elif optimizer == "nadam":
        return tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    else:
        raise ValueError("Invalid optimizer")
    
