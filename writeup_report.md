# **Behavioral Cloning** 

#### Note: I used tensorflow 2.3.0 with the tensorflow.keras package. I did NOT use the tensorflow 1.xx.xx version that came with the workspace.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

After trying simpler models, I built the following CNN as the final model. 
- 5 convolutions layers, each followed by batchnorm, maxpool and dropout (last layer has ReLU instead of maxpool)
- Flatten the last convolutional layer
- 3 fully connected layers, the first two have dropout

Note: batch normalization made a huge difference in performance. Adding bacthnorm made the biggest positive different of all architecture choices. 

With this model trained on the provided data and a single loop I recorded, I was still having difficulty on turns.  I fixed that in the following way:
- I recorded more data only on turns
- In the training loop, I signficantly oversampled images where the car was turning. This was since much of the track was straight, as well as the fact that the model was failing on certain turns on the track, which where either very tight or had dirt on the outside of the turn.
- In drive.py, I reduce speed when the car is turning. Also, I smooth the turning with a running average. I could avoid modifying drive.py if I attained more data. However, this was sufficient to get the car to go around the track, and should really be done anyways if an RNN is not used.

The data processing and training strategy is further detailed in model.py with comments.

#### 2. Final Model Architecture

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 65, 320, 32)       4736      
_________________________________________________________________
batch_normalization (BatchNo (None, 65, 320, 32)       128       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 32, 160, 32)       0         
_________________________________________________________________
dropout (Dropout)            (None, 32, 160, 32)       0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 32, 160, 64)       51264     
_________________________________________________________________
batch_normalization_1 (Batch (None, 32, 160, 64)       256       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 16, 80, 64)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 16, 80, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 16, 80, 128)       73856     
_________________________________________________________________
batch_normalization_2 (Batch (None, 16, 80, 128)       512       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 8, 40, 128)        0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 8, 40, 128)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 40, 128)        147584    
_________________________________________________________________
batch_normalization_3 (Batch (None, 8, 40, 128)        512       
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 4, 20, 128)        0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 4, 20, 128)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 4, 20, 256)        295168    
_________________________________________________________________
flatten (Flatten)            (None, 20480)             0         
_________________________________________________________________
batch_normalization_4 (Batch (None, 20480)             81920     
_________________________________________________________________
re_lu (ReLU)                 (None, 20480)             0         
_________________________________________________________________
dropout_4 (Dropout)          (None, 20480)             0         
_________________________________________________________________
dense (Dense)                (None, 128)               2621568   
_________________________________________________________________
re_lu_1 (ReLU)               (None, 128)               0         
_________________________________________________________________
dropout_5 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 32)                4128      
_________________________________________________________________
re_lu_2 (ReLU)               (None, 32)                0         
_________________________________________________________________
dropout_6 (Dropout)          (None, 32)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 33        
=================================================================
Total params: 3,281,665
Trainable params: 3,240,001
Non-trainable params: 41,664
_________________________________________________________________
