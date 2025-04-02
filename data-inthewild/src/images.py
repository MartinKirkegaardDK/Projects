import os
from collections import Counter
from sklearn.model_selection import train_test_split
import json
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib.style as style
import numpy as np
import random

def create_train_test(config, path, data_key):
    """
    config: takes the config file
    path: Takes the path to the scraped data json file
    data_key: is the key which you want as labels
    """
    with open(path, "r", encoding= "utf8") as f:
        data = json.load(f)

        link = "https://tasty.co/recipe/"
        X_unfiltered = []
        Y_unfiltered = []

        for key in os.listdir("../../data/images"):
            
            d_key = key.replace(".jpg", "")
            d_key = link + d_key 
            if d_key in data:
                Y_unfiltered.append(data[d_key][data_key])
                X_unfiltered.append(key)

        X = []
        Y = []
        for input, label in zip(X_unfiltered, Y_unfiltered):
            if label == []:
                continue
            X.append(input)
            Y.append(label)
        
        X_train, X_val, y_train, y_val = train_test_split(X,Y, test_size=0.2, random_state=44)
        X_train = [os.path.join('../../data/images/', str(f)) for f in X_train]
        X_val = [os.path.join('../../data/images/', str(f)) for f in X_val]
        return X_train, X_val, y_train, y_val
    
def parse_function(filename, label):
    """Function that returns a tuple of normalized image array and labels array.
    Args:
        filename: string representing path to image
        label: 0/1 one-dimensional array of size N_LABELS
    """
    # Read an image from a file
    image_string = tf.io.read_file(filename)
    # Decode it into a dense vector 
    image_decoded = tf.image.decode_jpeg(image_string, channels=3) # Keep RGB color channels to match the input format of the model
    # Resize it to fixed shape
    image_resized = tf.image.resize(image_decoded, [224, 224]) # IMG_SIZE : 224 Specify height and width of image to match the input format of the model

    # Normalize it from [0, 255] to [0.0, 1.0]
    image_normalized = image_resized / 255.0
    return image_normalized, label


def create_dataset(filenames, labels, config,is_training=True):
    """Load and parse dataset.
    Args:
        filenames: list of image paths
        labels: numpy array of shape (BATCH_SIZE, N_LABELS)
        is_training: boolean to indicate training mode
    """
    
    # Create a first dataset of file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    # Parse and preprocess observations in parallel
    #dataset = dataset.map(parse_function, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(parse_function)

    
    if is_training == True:
        # This is a small dataset, only load it once, and keep it in memory.
        #dataset = dataset.cache()
        # Shuffle the data each buffer size
        dataset = dataset.shuffle(buffer_size=config["SHUFFLE_BUFFER_SIZE"])
        
    # Batch the data for multiple steps
    dataset = dataset.batch(config["BATCH_SIZE"], drop_remainder= True)
    # Fetch batches in the background while the model is training.
    #dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset

def to_tensorflow(X_train, X_val, y_train, y_val, config):
    """converts the dataset into tensorflow data types"""
    # The original targets are lists of strings that can be easily understood by humans.  
    # But, if we want to build and train a neural network we need to create binary labels (multi-hot encoding).  
    # This is critical for multi-label classification.  

    # In order to binarize our labels, we will be using scikit-learn's MultiLabelBinarizer.  

    # Fit the multi-label binarizer on the training set
    mlb = MultiLabelBinarizer()
    mlb.fit(y_train)



    N_LABELS = len(mlb.classes_)
    # transform the targets of the training and test sets
    y_train_bin = mlb.transform(y_train)
    y_val_bin = mlb.transform(y_val)

    train_ds = create_dataset(X_train, y_train_bin,config)
    val_ds = create_dataset(X_val, y_val_bin, config)

    return train_ds, val_ds, N_LABELS, y_val_bin, mlb



@tf.function
def macro_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0) #sum
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    cost = 1 - soft_f1 # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = tf.reduce_mean(cost) # average on all labels
    return macro_cost


@tf.function
def macro_f1(y, y_hat, thresh=0.5):
    """Compute the macro F1-score on a batch of observations (average F1 across labels)
    
    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive
        
    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1

def load_functions(config):
     
    if config["LOSS_FUNCTION"] == "macro_soft_f1":
        loss = macro_soft_f1
    elif config["LOSS_FUNCTION"] == "binary_crossentropy":
        loss = tf.keras.metrics.binary_crossentropy
    else:
        raise "Not a valid loss functions"

    if config["METRICS"] == "macro_f1":
        metric = macro_f1
    else:
        raise "Not a valid metric"

    return loss, metric


def create_model(config, input_shape, n_labels):
    base_model = tf.keras.applications.MobileNetV2(input_shape =input_shape, include_top=False, weights='imagenet')

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    inputs = tf.keras.Input(shape=input_shape) #This is the input layer
    x = base_model(inputs, training=False) #This is the base model which we dont want to train

    x = global_average_layer(x)#This pooling layer transforms the shape from 7^2 X 1280 -> 1 X 1280

    x = tf.keras.layers.Dense(config["LAYER1"], activation=config["LAYER1_ACTIVATION"])(x)
    x = tf.keras.layers.Dropout(config["LAYER1_DROPOUT"])(x)#Adding dropout  

    x = tf.keras.layers.Dense(config["LAYER2"], activation=config["LAYER2_ACTIVATION"])(x)
    x = tf.keras.layers.Dropout(config["LAYER2_DROPOUT"])(x)#Adding dropout

    outputs = tf.keras.layers.Dense(n_labels, activation=config["OUTPUTLAYER_ACTIVATION"])(x)
    model = tf.keras.Model(inputs, outputs)

    loss, metric = load_functions(config)
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=float(config["LR"])),
        loss=loss,
        metrics=[metric])
    return model
    




def augment_image(image, label):
    # Create a list to store augmented versions of the image
    augmented_images = []

    random_float = random.uniform(-1, 1)
    # Add Gaussian noise
    std_dev = random_float  # Adjust the standard deviation as needed
    noisy_image = image + tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=std_dev, dtype=tf.float32)
    augmented_images.append(noisy_image)


    # Apply horizontal flip
    augmented_images.append(tf.image.flip_left_right(noisy_image))

    # Rotate the image by 90 degrees 3 times
    for angle in range(3):
        rotated_image = tf.image.rot90(noisy_image, k=angle)
        augmented_images.append(rotated_image)

    return augmented_images, [label] * 5


def augment_image_new(image, label):
    # Create a list to store augmented versions of the image
    augmented_images = []

    # Original image
    augmented_images.append(image)

    # Apply horizontal flip
    augmented_images.append(tf.image.flip_left_right(image))

    # Rotate the image by 40 degrees eight times
    for angle in range(40, 360, 40):
        rotated_image = tf.image.rot90(image, k=angle // 90)
        augmented_images.append(rotated_image)


    # Add random brightness noise
    augmented_images.append(tf.image.random_brightness(image, max_delta=0.2))

    # Add random contrast noise
    augmented_images.append(tf.image.random_contrast(image, lower=0.5, upper=1.5))

    # Add random saturation noise
    augmented_images.append(tf.image.random_saturation(image, lower=0.5, upper=1.5))

    # Add random hue noise
    augmented_images.append(tf.image.random_hue(image, max_delta=0.2))


    return augmented_images, [label] * 14    


def get_predictions(model, mlb, val_ds, boundary = 0.9):

    # Predict on the validation set with both models
    predictions = model.predict(val_ds)
    #y_hat_val_bce = model_bce.predict(val_ds)
    one_hot_encoded = []
    #We want to create a one hot onceded vector where if the probability of a label is over the boundary (0.9)
    #Then it labels index as a prediction
    for prediction in predictions:
        one_hot_encoded.append([1 if x > boundary else 0 for x in prediction])
    
    filtered_predictions = mlb.inverse_transform(np.array(one_hot_encoded))
    return filtered_predictions


def plot(img_path, prediction, truth, model_name):
    style.use('default')
    plt.figure(figsize=(8,4))
    plt.imshow(Image.open(img_path))
    dish_name = img_path.split('../../data/images/')[1].replace(".jpg","")
    plt.title(f"Dish: {dish_name} \n\nTruth: {truth} \n\nPredictions: {prediction}")

    save_dir = f"../../visualizations/images/predicted_images/{model_name}/"
    
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    plt.savefig(f"{save_dir}predicted_{dish_name}.jpg",bbox_inches='tight')


def plot_predictions(predictions, x_val, y_val, model_name, n):

    true_ingredients = y_val
    counter = 0
    for dish, prediction, truth in zip(x_val, predictions, true_ingredients):
        plot(dish, prediction, truth, model_name)

        counter += 1
        if counter == n:
            break



def learning_curves(history):
    """Plot the learning curves of loss and macro f1 score 
    for the training and validation datasets.
    
    Args:
        history: history callback of fitting a tensorflow keras model 
    """
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    macro_f1 = history.history['macro_f1']
    val_macro_f1 = history.history['val_macro_f1']
    
    epochs = len(loss)

    style.use("bmh")
    plt.figure(figsize=(8, 8))

    plt.subplot(2, 1, 1)
    plt.plot(range(1, epochs+1), loss, label='Training Loss')
    plt.plot(range(1, epochs+1), val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')

    plt.subplot(2, 1, 2)
    plt.plot(range(1, epochs+1), macro_f1, label='Training Macro F1-score')
    plt.plot(range(1, epochs+1), val_macro_f1, label='Validation Macro F1-score')
    plt.legend(loc='lower right')
    plt.ylabel('Macro F1-score')
    plt.title('Training and Validation Macro F1-score')
    plt.xlabel('epoch')

    plt.show()
    
    return loss, val_loss, macro_f1, val_macro_f1