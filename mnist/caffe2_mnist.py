#!/usr/bin/env python
# -*- coding:utf-8 -*-  

'''
This is the sample code for classification on MNIST dataset.
(url: https://github.com/BIGBALLON/Caffe2_Demo)
some of the code modified from https://caffe2.ai/docs/tutorial-MNIST.html

    change the argument [USE_GPU] to use caffe2 with CPU(or GPU)
    modify the function [AddModel] to create your own architecture.
    change the variable [image_name] to test your own picture.
    
'''

import os
import numpy as np

from caffe2.python.predictor import mobile_exporter          # save net
from caffe2.proto import caffe2_pb2                          # set device option & load net
from caffe2.python import (
    workspace,
    brew,
    core,
    model_helper,
    optimizer,
    )

IMAGE_CHANNELS      = 1                                      # input image channels
NUM_CLASSES         = 10                                     # number of image classes
TRAINING_ITERS      = 10000                                  # total training iterations
TRAINING_BATCH_SIZE = 128                                    # batch size for training

INIT_NET            = './model/init_net.pb'
PREDICT_NET         = './model/predict_net.pb' 

CURRENT_FOLDER      = os.path.join('./')
DATA_FOLDER         = os.path.join(CURRENT_FOLDER, 'data')
ROOT_FOLDER         = os.path.join(CURRENT_FOLDER, 'model')
USE_GPU             = True 
GPU_ID              = 0                                      # GPU numbers

def DownloadData():
    db_missing  = False
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)   
        print("Your data folder was not found!! This was generated: {}".format(DATA_FOLDER))
    if os.path.exists(os.path.join(DATA_FOLDER,"mnist-train-nchw-lmdb")):
        print("lmdb train db found!")
    else:
        db_missing = True
    if os.path.exists(os.path.join(DATA_FOLDER,"mnist-test-nchw-lmdb")):
        print("lmdb test db found!")
    else:
        db_missing = True
    # Attempt the download of the db if either was missing
    if db_missing:
        print("one or both of the MNIST lmbd dbs not found!!")
        db_url = "http://download.caffe2.ai/databases/mnist-lmdb.zip"
        try:
            '''Downloads resources from s3 by url and unzips them to the provided path'''
            import requests, zipfile
            from io import BytesIO
            print("Downloading... {} to {}".format(db_url, DATA_FOLDER))
            r = requests.get(db_url, stream=True)
            z = zipfile.ZipFile(BytesIO(r.content))
            z.extractall(DATA_FOLDER)
            print("Completed download and extraction.")
        except Exception as ex:
            print("Failed to download dataset. Please download it manually from {}".format(db_url))
            print("Unzip it and place the two database folders here: {}".format(DATA_FOLDER))
            raise ex

    print("training data folder:" + DATA_FOLDER)
    print("workspace root folder:" + ROOT_FOLDER)

def AddInput(model, batch_size, db, db_type, device_opts):
    with core.DeviceScope(device_opts):
        # load the data
        data_uint8, label = brew.db_input(
            model,
            blobs_out=["data_uint8", "label"],
            batch_size=batch_size,
            db=db,
            db_type=db_type,
        )
        # cast the data to float
        data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)

        # scale data from [0,255] down to [0,1]
        data = model.Scale(data, data, scale=float(1./256))

        # don't need the gradient for the backward pass
        data = model.StopGradient(data, data)
        return data, label

def AddModel(model, data, device_opts,is_test=False):
    with core.DeviceScope(device_opts):
        conv1 = brew.conv(
            model, 
            data, 
            'conv1', 
            dim_in=IMAGE_CHANNELS, 
            dim_out=6, 
            weight_init=('MSRAFill', {}),
            kernel=5, 
            stride=1, 
            pad=0)

        relu1 = brew.relu(model, conv1, 'relu1')
        pool1 = brew.max_pool(model, relu1, 'pool1', kernel=2, stride=2)
        
        conv2 = brew.conv(
            model, 
            pool1, 
            'conv2', 
            dim_in=6, 
            dim_out=16, 
            weight_init=('MSRAFill', {}),
            kernel=5, 
            stride=1, 
            pad=0)

        relu2 = brew.relu(model, conv2, 'relu2')
        pool2 = brew.max_pool(model, relu2, 'pool2', kernel=2, stride=2)
        
        # Fully connected layers
        fc1 = brew.fc(model, pool2, 'fc1', dim_in=16*4*4, dim_out=120)
        relu3 = brew.relu(model, fc1, 'relu3')
        dropout1 = brew.dropout(model, relu3, 'dropout1', ratio=0.5, is_test=is_test)
        
        fc2 = brew.fc(model, dropout1, 'fc2', dim_in=120, dim_out=84)
        relu4 = brew.relu(model, fc2, 'relu4')
        dropout2 = brew.dropout(model, relu4, 'dropout2', ratio=0.5, is_test=is_test)

        fc3 = brew.fc(model, dropout2, 'fc3', dim_in=84, dim_out=NUM_CLASSES)
        # Softmax layer
        softmax = brew.softmax(model, fc3, 'softmax')
        
        return softmax

def AddTrainingOperators(model, softmax, label, device_opts):
    with core.DeviceScope(device_opts):
        xent = model.LabelCrossEntropy([softmax, label], 'xent')
        # Compute the expected loss
        loss = model.AveragedLoss(xent, "loss")
        brew.accuracy(model, [softmax, label], "accuracy")
        # Use the average loss we just computed to add gradient operators to the model
        model.AddGradientOperators([loss])
        # Use SGD optimizer
        optimizer.build_sgd(
            model,
            base_learning_rate=0.1,
            weight_decay=1e-5,
            gamma=0.999, 
            policy='step', 
            stepsize=50,
            nesterov=1,
        )
        # feel free to use other optimizers [e.g. adam ,adagrad]
        # e.g.
        # optimizer.build_adam(
        #     model, 
        #     base_learning_rate=1e-4,
        #     weight_decay=1e-4,
        #     )

def AddAccuracy(model, softmax, label, device_opts):
    with core.DeviceScope(device_opts):
        accuracy = brew.accuracy(model, [softmax, label], "accuracy")
        return accuracy

def SaveNet(INIT_NET, PREDICT_NET, workspace, model):
    init_net, predict_net = mobile_exporter.Export(
        workspace, model.net, model.params
    )
    
    with open(PREDICT_NET, 'wb') as f:
        f.write(model.net._net.SerializeToString())
    with open(INIT_NET, 'wb') as f:
        f.write(init_net.SerializeToString())

def LoadNet(INIT_NET, PREDICT_NET, device_opts):
    init_def = caffe2_pb2.NetDef()
    with open(INIT_NET, 'rb') as f:
        init_def.ParseFromString(f.read())
        init_def.device_option.CopyFrom(device_opts)
        workspace.RunNetOnce(init_def.SerializeToString())
    
    net_def = caffe2_pb2.NetDef()
    with open(PREDICT_NET, 'rb') as f:
        net_def.ParseFromString(f.read())
        net_def.device_option.CopyFrom(device_opts)
        workspace.CreateNet(net_def.SerializeToString(), overwrite=True)

def DoTrain():
    # Check workspace's root folder & reset
    if os.path.exists(ROOT_FOLDER):
        import shutil
        print("Looks like you ran this before, so we need to cleanup those old files...")
        shutil.rmtree(ROOT_FOLDER)
    os.makedirs(ROOT_FOLDER)
            
    workspace.ResetWorkspace(ROOT_FOLDER)

    arg_scope = {"order": "NCHW"}
    
    # Initialize with ModelHelper class
    train_model = model_helper.ModelHelper(name="train_net", arg_scope=arg_scope)

    # Add data layer from training_lmdb
    data, label = AddInput(
        train_model, 
        batch_size=TRAINING_BATCH_SIZE,
        db=os.path.join(DATA_FOLDER, 'mnist-train-nchw-lmdb'),
        db_type='lmdb',
        device_opts=DEVICE_OPTS
        )
    # Add model definition, save return value to 'softmax' variable
    softmax = AddModel(train_model, data, DEVICE_OPTS)
    # Add training operators using the softmax output from the model
    AddTrainingOperators(train_model, softmax, label, DEVICE_OPTS)

    test_model = model_helper.ModelHelper(
        name="test_net", 
        arg_scope=arg_scope, 
        init_params=False
        )

    data, label = AddInput(
        test_model, 
        batch_size=100,
        db=os.path.join(DATA_FOLDER, 'mnist-test-nchw-lmdb'),
        db_type='lmdb',
        device_opts=DEVICE_OPTS
        )

    softmax = AddModel(test_model, data, DEVICE_OPTS)
    AddAccuracy(test_model, softmax, label, DEVICE_OPTS)

    # Deployment model. 
    # We simply need the main AddModel part.
    deploy_model = model_helper.ModelHelper(
        name="deploy_net", 
        arg_scope=arg_scope, 
        init_params=False
        )
    AddModel(deploy_model, "data", DEVICE_OPTS, True)

    # Initialize and create the training network
    workspace.RunNetOnce(train_model.param_init_net)
    workspace.CreateNet(train_model.net, overwrite=True)

    # Set the iterations number and track the accuracy & loss
    accuracy = np.zeros(TRAINING_ITERS)
    loss = np.zeros(TRAINING_ITERS)

    # MAIN TRAINING LOOP!
    print("Start training...")
    for i in range(TRAINING_ITERS):
        workspace.RunNet(train_model.net)
        accuracy[i] = workspace.blobs['accuracy']
        loss[i] = workspace.blobs['loss']
        # Check the accuracy and loss every so often
        if i % 50 == 0:
            print("Iter: {:5d}, Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss[i], accuracy[i]))
    print("Done.")

    # param_init_net here will only create a data reader
    # Other parameters won't be re-created because we selected [init_params=False] before
    workspace.RunNetOnce(test_model.param_init_net)
    workspace.CreateNet(test_model.net, overwrite=True)

    # Testing Loop 
    # batch size:        100 
    # iteration:         100
    # total test images: 10000
    test_accuracy = np.zeros(100)
    for i in range(100):
        # Run a forward pass of the net on the current batch
        workspace.RunNet(test_model.net)
        # Collect the batch accuracy from the workspace
        test_accuracy[i] = workspace.FetchBlob('accuracy')
        
    print('test_accuracy: {:.4f}'.format(test_accuracy.mean()))
    
    # Save INIT_NET & PREDICT_NET
    SaveNet(INIT_NET, PREDICT_NET, workspace, deploy_model)


def DoImageTest():
    
    # Reset workspace
    workspace.ResetWorkspace(ROOT_FOLDER)
    
    # Load model
    LoadNet(INIT_NET, PREDICT_NET, DEVICE_OPTS)
    
    # We use opencv to read images, 
    # Please make sure you have installed it.
    # pip install opencv-python(check it)
    import cv2
    image_name = './test_img/9.jpg'
    img = cv2.imread( image_name )                              # Load test image
    img = cv2.resize(img, (28,28))                              # Resize to 28x28
    img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )               # Covert to grayscale
    img = img.reshape((1,1,28,28)).astype('float32')            # Reshape to (1,1,28,28)
    workspace.FeedBlob("data", img, device_option=DEVICE_OPTS)  # FeedBlob
    workspace.RunNet('deploy_net', num_iter=1)                  # Forward

    print("\nInput: {}".format(img.shape))
    pred = workspace.FetchBlob("softmax")
    print("Output: {}".format(pred))
    print("Output class: {}".format(np.argmax(pred)))


if __name__ == '__main__':

    # 1. Set global init level & Device Option: CUDA or CPU
    
    # We suggest you to use GPU(if you can) since the speed of CPU is very slow

    core.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    if USE_GPU:
        DEVICE_OPTS = core.DeviceOption(caffe2_pb2.CUDA, GPU_ID)  
    else:
        DEVICE_OPTS = core.DeviceOption(caffe2_pb2.CPU, 0)
    
    # 2. Download training data if need
    DownloadData()

    # 3. Start training & save pb files.
    DoTrain()

    # 4. Do a real image test if you need
    DoImageTest()
    # Then you can see 
    # Input: ones
    # Output: [[0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
    # Output class: 9
    