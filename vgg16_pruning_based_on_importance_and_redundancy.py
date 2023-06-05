import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D,Dropout, BatchNormalization,  MaxPooling2D, Flatten, Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tensorflow.keras import datasets,models,layers
from scipy.stats import pearsonr
from sklearn.metrics.cluster import adjusted_mutual_info_score
from keras.optimizers import Adam
from numba import cuda
import numpy as np
from keras.regularizers import l2
from keras.optimizers import schedules
import os
import math
import sys

####################################################################
################   HELPER FUNCTIONS.  ##############################
####################################################################

# Function to measure pearson correlation 
def np_pearson_cor(x, y):
    xv = x - x.mean(axis=0)
    yv = y - y.mean(axis=0)
    xvss = (xv * xv).sum(axis=0)
    yvss = (yv * yv).sum(axis=0)
    result = np.matmul(xv.transpose(), yv) / (np.sqrt(np.outer(xvss, yvss)))
    return np.maximum(np.minimum(result, 1.0), -1.0)


# Function to get the average entropy values 
def calculate_entropy_across_axis_1(average_pooled_filter_output):
  # Find the number of classes in this distribution 
  num_channels   = average_pooled_filter_output.shape[1]
  num_examples = average_pooled_filter_output.shape[0]
  
  entropy_channel = np.zeros(num_channels)

  # Measure entropy per channel 
  for channel_idx in range(0, num_channels, 1):    
    hist_f1, bins_f1 = np.histogram(average_pooled_filter_output[ :,channel_idx ], bins=10)
    classes_f1       = np.digitize(average_pooled_filter_output[ :,channel_idx], bins_f1)
    bin_count        = np.bincount(classes_f1) + 1 
    bin_probablities = bin_count/(num_examples)
      
    # Find the log of each probablity 
    bin_probablities = bin_probablities  
    log_bin_probablities = np.log(bin_probablities)

    # Shannon probablity is  Sum (-( p(x)) * log (p(x)))
    bin_probablities_product = -1 * np.multiply(log_bin_probablities, bin_probablities) 
    
    #Sum across all classes 
    entropy_channel[channel_idx] = np.sum(bin_probablities_product)
   
  return (entropy_channel)

####################################################################
##########   BASE MODEL    #########################################
####################################################################
# Load the CIFAR10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the data
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

mean = np.mean(x_train, axis=(0, 1, 2, 3))
std = np.std(x_train, axis=(0, 1, 2, 3))
x_train = (x_train - mean) / (std + 1e-7)
x_test = (x_test - mean) / (std + 1e-7)


# Convert the labels to categorical data
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# Create a new subset of the test dataset with only the first 10000 elements
# This is going to be used for E and S calculation 
x_test_subset = x_train [:10000]
y_test_subset = y_train_cat[:10000]

# Define a VGG model
model = Sequential()

tf.random.set_seed(0)

# LAYER 1
model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3), kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
model.add(MaxPooling2D((2, 2), strides=(2, 2), padding="same"))

# LAYER 2
model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
model.add(Dropout(0.5))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
model.add(MaxPooling2D((2, 2), strides=(2, 2), padding="same"))

# LAYER 3
model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
model.add(Dropout(0.5))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
model.add(Dropout(0.5))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
model.add(MaxPooling2D((2, 2), strides=(2, 2), padding="same"))

# LAYER 4
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu" , kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
model.add(Dropout(0.5))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu",  kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
model.add(Dropout(0.5))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu",  kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

# LAYER 5
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu",  kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
model.add(Dropout(0.5))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu",  kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
model.add(Dropout(0.5))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu",  kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

# FULLY CONNECTED LAYERS 
model.add(Flatten())
model.add(Dense(4096, activation='relu', kernel_regularizer=l2(0.1), bias_regularizer=l2(0.1)))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu', kernel_regularizer=l2(0.1), bias_regularizer=l2(0.1)))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

checkpoint_path = "training_4/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

checkpoint_path2 = "training_5/cp.ckpt"
checkpoint_dir2 = os.path.dirname(checkpoint_path2)


# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path2,
                                                 save_weights_only=True,
                                                 verbose=1)


opt = Adam(learning_rate=0.000075) # lr_decayed_fn)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Test the accuracy of the base model 
ModelLoss, ModelAccuracy = model.evaluate(x_test, y_test_cat)

# Loads the weights
model.load_weights(checkpoint_path)

# Train the model, you can uncomment this if you want to further train this model 
# model.fit(x_train, y_train_cat, epochs=100, batch_size=128, callbacks=[cp_callback])

# Test the accuracy of the base model 
ModelLoss, ModelAccuracy = model.evaluate(x_test, y_test_cat)
print('Model Accuracy BASE is {}'.format(ModelAccuracy))

# Get the threshold values 
similarity_treshold = float(sys.argv[2])
entropy_treshold = float(sys.argv[1])

# Create an extactor model 
extractor = keras.Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])

# Initalize some variables that will be used in the loop
layer_number = -1
num_channels = 3
num_output_vals = 0 
flops_unpruned_total = 1
flops_pruned_total = 1

###############################################################################################
# Main loop of the Pruning alogorithm
# Go over all the layers one after the other and sequentially prune them based on S and E 
###############################################################################################
pruning_percentage = 0.2 
for layer in model.layers:
    layer_number = layer_number + 1 
    # We can skip the last dense layer 
    if (layer_number >= (len(model.layers)-1)) :
        continue 
    print("PROCESSING - ", layer.name)

    # Currently we only apply our flow to convolution and dense layers
    if (isinstance(layer, tf.keras.layers.Conv2D)) :
        # Create a numpy array with dimentions (num_example, num_hidden_layer_channels)
        hidden_layer_average_pool = np.random.rand(len(x_test_subset), layer.filters)
        batch_output_entropy = []
        
        # For the entire training data, we need to find the filter outputs 
        # Create a list per filter (2-D numpy array) of the average values of the filter outputs 
        # For this version, we are only pruning the layer 4 
        for batch_idx in range(0, (len(x_test_subset)), 25):
          # Measure the mutual information between each filter   
          current_layer_output = extractor(x_test_subset[batch_idx:batch_idx+25])
          current_layer_output_average_pool = tf.reduce_mean(current_layer_output[layer_number], axis=[1, 2])
          hidden_layer_average_pool[batch_idx:batch_idx+25] =  current_layer_output_average_pool 
        
        ####################################################################
        ######### PART 1 - MEASURE ENTRORY OF THE FILTER AND PRUNE #########
        ####################################################################
        # Measure the entropy of the channel
        hidden_layer_average_pool_entropy = calculate_entropy_across_axis_1(hidden_layer_average_pool)
        
        filters_pruned_entropy = []
        filters_pruned_similarity = [] 
        filters_already_processed = []

        # Remove all nodes that have low entropy s
        for hidden_node in range(0, hidden_layer_average_pool.shape[1]):
            if(hidden_layer_average_pool_entropy[hidden_node] < entropy_treshold):
              layer = model.layers[layer_number]
              W_matrix = layer.weights[0]
              b_list = layer.weights[1]
        
              # Set the weights of the W matrix as 0 
              mask_b = np.ones(b_list.shape)
              mask_b[hidden_node] = 0
              mask_b_tensor  = tf.convert_to_tensor(mask_b, dtype=float)
              new_b_list     = tf.math.multiply(mask_b_tensor, b_list)
        
              # Set the bias term as 0
              mask = np.ones(W_matrix.shape)
              mask[:,:,:,hidden_node] = 0
              mask_tensor  = tf.convert_to_tensor(mask, dtype=float)
              new_W_matrix = tf.math.multiply(mask_tensor, W_matrix)
        
              # Assign new weights to the model 
              new_weights = [new_W_matrix, new_b_list]
              model.layers[layer_number].set_weights(new_weights)
        
              filters_already_processed.append(hidden_node)
              filters_pruned_entropy.append(hidden_node)
        
        ModelLoss, ModelAccuracy = model.evaluate(x_test, y_test_cat)
        print("NUMBER OF CNN FILTERS ENTROPY PRUNED ", len(filters_pruned_entropy), "OF", hidden_layer_average_pool.shape[1])
        
        print('Model Accuracy AFTER ENTROPY  is {}'.format(ModelAccuracy))
        
        if layer_number > 4:
         
            ####################################################################
            ###### PART 2 - PERFORM SIMILARITY MAPPING AND PRUNE ###############
            ####################################################################
            
            # Measure the Similarity information between all the filters 
            similarity_mapping = np.zeros((hidden_layer_average_pool.shape[1], hidden_layer_average_pool.shape[1]), dtype=float)
             
            for f1_idx in range(0, (hidden_layer_average_pool.shape[1]), 1):
              if(f1_idx not in filters_pruned_entropy): 
                for f2_idx in range(f1_idx, (hidden_layer_average_pool.shape[1]), 1):  
                    if(f2_idx not in filters_pruned_entropy):
                        corr= np_pearson_cor(hidden_layer_average_pool[:,f2_idx],hidden_layer_average_pool[ :,f1_idx ])
                        similarity_mapping[f1_idx, f2_idx] = abs(corr)
            
            # Using the Similarity matrix, generate a list of filters that will be pruned
            pruning_percentage_applied = np.minimum(1, pruning_percentage)
            print("Pruning percentage applied", pruning_percentage_applied)
            max_pruning = math.floor((hidden_layer_average_pool.shape[1] - len(filters_pruned_entropy)) * pruning_percentage_applied)
            pruning_percentage = pruning_percentage_applied + 0.1 
            for pruning_index in range(0, max_pruning, 1):  
              max_similarity = 0 ;
              max_similarity_filters_index  = []
              for f1_idx in range(0, (hidden_layer_average_pool.shape[1]), 1):
                if ((f1_idx not in filters_pruned_entropy)):
                  for f2_idx in range(f1_idx, (hidden_layer_average_pool.shape[1]), 1):
                    if ((f2_idx not in filters_pruned_entropy)):
                      if ((similarity_mapping[f1_idx, f2_idx] > max_similarity) and (similarity_mapping[f1_idx, f2_idx] > similarity_treshold)):
                        if(f1_idx != f2_idx):
                          if ((f1_idx not in filters_already_processed) and (f2_idx not in filters_already_processed)):
                            max_similarity =  similarity_mapping[f1_idx, f2_idx]
                            max_similarity_filters_index.clear()
                            max_similarity_filters_index.append(f1_idx)
                            max_similarity_filters_index.append(f2_idx)
             
              if(max_similarity != 0) : 
                filters_already_processed.append(max_similarity_filters_index[0])
                filters_already_processed.append(max_similarity_filters_index[1])
                filters_pruned_similarity.append(max_similarity_filters_index[0])
            
            # Prune away the one of the filters that are similar 
            for hidden_node in range(0, hidden_layer_average_pool.shape[1]):
              if( hidden_node in  filters_pruned_similarity):
                  #print("Setting weight to zero for filter", hidden_node)
                  layer = model.layers[layer_number]
                  W_matrix = layer.weights[0]
                  b_list = layer.weights[1]
            
                  # Set the weights of the W matrix as 0 
                  mask_b = np.ones(b_list.shape)
                  mask_b[hidden_node] = 0
                  mask_b_tensor  = tf.convert_to_tensor(mask_b, dtype=float)
                  new_b_list     = tf.math.multiply(mask_b_tensor, b_list)
            
                  # Set the bias term as 0
                  mask = np.ones(W_matrix.shape)
                  mask[:,:,:,hidden_node] = 0
                  mask_tensor  = tf.convert_to_tensor(mask, dtype=float)
                  new_W_matrix = tf.math.multiply(mask_tensor, W_matrix)
            
                  # Assign new weights to the model 
                  new_weights = [new_W_matrix, new_b_list]
                  model.layers[layer_number].set_weights(new_weights)
               
        ModelLoss, ModelAccuracy = model.evaluate(x_test, y_test_cat)
        print('Model Accuracy AFTER SIMILARIY BEFORE RETRAIN {}'.format(ModelAccuracy))
        print("NUMBER OF CNN FILTERS SIMILARITY PRUNED ", len(filters_pruned_similarity), "OF", hidden_layer_average_pool.shape[1])

        for var in model.optimizer.variables():
            var.assign(tf.zeros_like(var))
     
        model.fit(x_train, y_train_cat, epochs=3, batch_size=128)
        ModelLoss, ModelAccuracy = model.evaluate(x_test, y_test_cat)
       
        print('Model Accuracy AFTER SIMILARIY AFTER RETRAIN {}'.format(ModelAccuracy))

        # Calculate the MACC per CNN
        # flops_unpruned = K × K × Cin × Hout × Wout × Cout
        flops_unpruned  =  layer.kernel_size[0] * layer.kernel_size[0] * layer.input_shape[-1] * layer.output_shape[1] * layer.output_shape[2] * layer.output_shape[3] 
        flops_pruned    =  layer.kernel_size[0] * layer.kernel_size[0] * num_channels * layer.output_shape[1] * layer.output_shape[2] * (layer.output_shape[3] - len(filters_pruned_entropy) - len(filters_pruned_similarity))
        num_channels = (layer.output_shape[3] - len(filters_pruned_entropy) - len(filters_pruned_similarity))
        num_output_vals = num_channels *  layer.output_shape[1] * layer.output_shape[2]
        flops_unpruned_total  = flops_unpruned_total + flops_unpruned
        flops_pruned_total    = flops_pruned_total   + flops_pruned

    elif(isinstance(layer, tf.keras.layers.Dense) )  :
        # Create a numpy array with dimentions (num_example, num_hidden_layer_channels)
        hidden_layer_activation_value = np.random.rand(len(x_test_subset), layer.weights[1].shape[0])
        batch_output_entropy = []
        
        # For the entire training data, we need to find the filter outputs 
        for batch_idx in range(0, (len(x_test_subset)), 25):
          # Measure the mutual information between each filter   
          current_layer_output = extractor(x_test_subset[batch_idx:batch_idx+25])
          current_layer_output_activation_value = current_layer_output[layer_number]
          hidden_layer_activation_value[batch_idx:batch_idx+25] =  current_layer_output_activation_value 
       
        ####################################################################
        ######### PART 1 - MEASURE ENTRORY OF THE FILTER AND PRUNE #########
        ####################################################################
        # Measure the entropy of the channel
        hidden_layer_activation_value_entropy = calculate_entropy_across_axis_1(hidden_layer_activation_value)

        filters_pruned_entropy = []
        filters_pruned_similarity = [] 
        filters_already_processed = []
        
        # Remove all nodes that have low entropy s
        for hidden_node in range(0, hidden_layer_activation_value.shape[1]):
            if( hidden_layer_activation_value_entropy[hidden_node] < entropy_treshold) : 
              #print("Setting weight to zero for filter", hidden_node)
              
              layer = model.layers[layer_number]
              W_matrix = layer.weights[0]
              b_list = layer.weights[1]
        
              # Set the weights of the W matrix as 0 
              mask_b = np.ones(b_list.shape)
              mask_b[hidden_node] = 0
              mask_b_tensor  = tf.convert_to_tensor(mask_b, dtype=float)
              new_b_list     = tf.math.multiply(mask_b_tensor, b_list)
        
              # Set the bias term as 0
              mask = np.ones(W_matrix.shape)
              mask[:,hidden_node] = 0
              mask_tensor  = tf.convert_to_tensor(mask, dtype=float)
              new_W_matrix = tf.math.multiply(mask_tensor, W_matrix)
        
              # Assign new weights to the model 
              new_weights = [new_W_matrix, new_b_list]
              model.layers[layer_number].set_weights(new_weights)
        
              filters_already_processed.append(hidden_node)
              filters_pruned_entropy.append(hidden_node)
        
        ModelLoss, ModelAccuracy = model.evaluate(x_test, y_test_cat )
        
        print('Model Accuracy AFTER ENTROPY  is {}'.format(ModelAccuracy))
       
        print("NUMBER OF DENSE LAYERS PRUNED ", len(filters_pruned_entropy))

        if layer_number < 4:
            continue
        
        ####################################################################
        ###### PART 2 - PERFORM SIMILARITY MAPPING AND PRUNE ###############
        ####################################################################
        
        
        # Measure the mutual information between all the filters 
        similarity_mapping = np.zeros((hidden_layer_activation_value.shape[1], hidden_layer_activation_value.shape[1]), dtype=float)
        
        for f1_idx in range(0, (hidden_layer_activation_value.shape[1]), 1):
          if ((f1_idx not in filters_pruned_entropy)): 
            #print("PROCESSING ", f1_idx)
            for f2_idx in range(f1_idx, (hidden_layer_activation_value.shape[1]), 1):  
                if ((f2_idx not in filters_pruned_entropy)):
                    corr = np_pearson_cor(hidden_layer_activation_value[0:300,f2_idx],hidden_layer_activation_value[0:300,f1_idx ])
                    similarity_mapping[f1_idx, f2_idx] = abs(corr)
                    if ((similarity_mapping[f1_idx, f2_idx] > similarity_treshold)):
                      if(f1_idx != f2_idx):
                          if ((f1_idx not in filters_already_processed) and (f2_idx not in filters_already_processed)):
                            filters_already_processed.append(f1_idx)
                            filters_already_processed.append(f2_idx) 
                            filters_pruned_similarity.append(f1_idx)
        
        # Prune away the one of the filters that are similar 
        for hidden_node in range(0, hidden_layer_activation_value.shape[1]):
          if( hidden_node in  filters_pruned_similarity):
              #print("Setting weight to zero for filter", hidden_node)
              layer = model.layers[layer_number]
              W_matrix = layer.weights[0]
              b_list = layer.weights[1]
        
              # Set the weights of the W matrix as 0 
              mask_b = np.ones(b_list.shape)
              mask_b[hidden_node] = 0
              mask_b_tensor  = tf.convert_to_tensor(mask_b, dtype=float)
              new_b_list     = tf.math.multiply(mask_b_tensor, b_list)
        
              # Set the bias term as 0
              mask = np.ones(W_matrix.shape)
              mask[:,hidden_node] = 0
              mask_tensor  = tf.convert_to_tensor(mask, dtype=float)
              new_W_matrix = tf.math.multiply(mask_tensor, W_matrix)
        
              # Assign new weights to the model 
              new_weights = [new_W_matrix, new_b_list]
              model.layers[layer_number].set_weights(new_weights)

        
        ModelLoss, ModelAccuracy = model.evaluate(x_test, y_test_cat)
        print('Model Accuracy AFTER SIMILARIY BEFORE RETRAIN {}'.format(ModelAccuracy))
        print("NUMBER OF DENSE LAYERS PRUNED ", len(filters_pruned_similarity))
        for var in model.optimizer.variables():  
            var.assign(tf.zeros_like(var))

        model.fit(x_train, y_train_cat, epochs=3, batch_size=128)
        
        ModelLoss, ModelAccuracy = model.evaluate(x_test, y_test_cat )
       
        # Calculate the MACC per Dense layee 
        # flops_unpruned = Input shape x Output shape
        flops_unpruned  = layer.input_shape[-1] * layer.output_shape[-1]
        flops_pruned    = num_output_vals * (layer.output_shape[-1] - len(filters_pruned_entropy) - len(filters_pruned_similarity))
        num_output_vals = (layer.output_shape[-1] - len(filters_pruned_entropy) - len(filters_pruned_similarity))
        flops_unpruned_total  = flops_unpruned_total + flops_unpruned
        flops_pruned_total    = flops_pruned_total   + flops_pruned

        print('Model Accuracy AFTER SIMILARIY AFTER RETRAIN {}'.format(ModelAccuracy))

####################################################################
################  COLLECT ACCURACY AFTER PRUNING  ################## 
####################################################################
prun_ratio_ach = (flops_unpruned_total/flops_pruned_total)

ModelLoss, ModelAccuracy = model.evaluate(x_test, y_test_cat)


file_name = "data_stats"+ str(similarity_treshold) + str(entropy_treshold)
file = open(file_name, 'w')
file.write("Model Accuracy AFTER REGRESSION  is "+ str(ModelAccuracy) + " prun_ratio_ach " + str(prun_ratio_ach) + " SIMILARITY " + str(similarity_treshold) + " ENTROPY "  +  str(entropy_treshold))
file.close()
