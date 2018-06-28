#!/usr/bin/env python
# -*- coding:utf-8 -*-  

from caffe2.proto import caffe2_pb2
from caffe2.python.predictor import mobile_exporter
from caffe2.python import (
    workspace,
    core,
    model_helper,
    brew,
    optimizer,
    utils,
    )
from data_utility import (
    data_augmentation,
    prepare_data,
    color_preprocessing,
    next_batch,
    next_batch_random,
    dummy_input,
    )
from models import create_lenet, create_resnet_32x32
import numpy as np

USE_GPU = True
GPU_ID = 0
BATCH_SIZE = 128
EPOCHS = 150
TRAIN_IMAGES = 50000
TEST_IMAGES = 10000
INIT_NET = './init_net.pb'
PREDICT_NET = './predict_net.pb'


def add_accuracy(model, softmax, label, device_opts):
    with core.DeviceScope(device_opts):
        accuracy = brew.accuracy(model, [softmax, label], "accuracy")
        return accuracy

def add_training_operators(model, softmax, device_opts) :

    with core.DeviceScope(device_opts):
        xent = model.LabelCrossEntropy([softmax, "label"], 'xent')
        loss = model.AveragedLoss(xent, "loss")
        brew.accuracy(model, [softmax, "label"], "accuracy")

        model.AddGradientOperators([loss])
        opt = optimizer.build_sgd(
            model, 
            base_learning_rate=0.1, 
            policy="step", 
            stepsize=50000 * 50 // BATCH_SIZE, 
            weight_decay=3e-4,
            momentum=0.9, 
            gamma=0.1,
            nesterov=1,
            ) 

def save_net(init_net_pb, predict_net_pb, model):
    extra_params = []
    extra_blobs = []
    for blob in workspace.Blobs():
        name = str(blob)
        if name.endswith("_rm") or name.endswith("_riv"):
            extra_params.append(name)
            extra_blobs.append(workspace.FetchBlob(name))
    for name, blob in zip(extra_params, extra_blobs):
        workspace.FeedBlob(name, blob)
        model.params.append(name)

    init_net, predict_net = mobile_exporter.Export(
        workspace, 
        model.net, 
        model.params
        )
    
    with open(predict_net_pb, 'wb') as f:
        f.write(model.net._net.SerializeToString())
    with open(init_net_pb, 'wb') as f:
        f.write(init_net.SerializeToString())

def load_net(init_net_pb, predict_net_pb, device_opts):
    init_def = caffe2_pb2.NetDef()
    with open(init_net_pb, 'rb') as f:
        init_def.ParseFromString(f.read())
        init_def.device_option.CopyFrom(device_opts)
        workspace.RunNetOnce(init_def.SerializeToString())
    
    net_def = caffe2_pb2.NetDef()
    with open(predict_net_pb, 'rb') as f:
        net_def.ParseFromString(f.read())
        net_def.device_option.CopyFrom(device_opts)
        workspace.CreateNet(net_def.SerializeToString(), overwrite=True)

def do_train(init_net_pb, predict_net_pb, epochs, batch_size, device_opts) :

    workspace.ResetWorkspace()

    data, label = dummy_input()
    workspace.FeedBlob("data", data, device_option=device_opts)
    workspace.FeedBlob("label", label, device_option=device_opts)

    train_arg_scope = {
        'order': 'NCHW',
        'use_cudnn': True,
        'cudnn_exhaustive_search': True,
    }

    train_model= model_helper.ModelHelper(name="train_net", arg_scope=train_arg_scope)
    # softmax = create_lenet(train_model, device_opts=device_opts)
    
    # print(train_model.net.Proto().name)

    softmax = create_resnet_32x32(
        model=train_model,
        data='data', 
        num_input_channels=3,
        num_groups=5,
        num_labels=10, 
        device_opts=device_opts,
        is_test=False)

    add_training_operators(train_model, softmax, device_opts=device_opts)
    workspace.RunNetOnce(train_model.param_init_net)
    workspace.CreateNet(train_model.net)

    test_model= model_helper.ModelHelper(name="test_net", init_params=False)
    # softmax = create_lenet(test_model, device_opts=device_opts, is_test=True)
    softmax = create_resnet_32x32(
        model=test_model,
        data='data', 
        num_input_channels=3,
        num_groups=5,
        num_labels=10, 
        device_opts=device_opts,
        is_test=True)
    add_accuracy(test_model, softmax, "label", device_opts)

    workspace.RunNetOnce(test_model.param_init_net)
    workspace.CreateNet(test_model.net, overwrite=True)
    
    # print(workspace.Blobs())

    print('\ntraining for', epochs, 'epochs')

    for e in range(0, epochs):
        loss_train = []
        acc_train  = []
        loss_test  = []
        acc_test   = []
        for i in range(0, TRAIN_IMAGES//batch_size + 1):
            data, label = next_batch(i, batch_size, train_x, train_y, TRAIN_IMAGES)
            data = data_augmentation(data)

            workspace.FeedBlob("data", data, device_option=device_opts)
            workspace.FeedBlob("label", label, device_option=device_opts)

            workspace.RunNet(train_model.net, 1) 
            loss_train.append(workspace.FetchBlob("loss"))
            acc_train.append(workspace.FetchBlob("accuracy"))

        for i in range(0, TEST_IMAGES//1000):
            data, label = next_batch(i, 1000, test_x, test_y, TEST_IMAGES)
            workspace.FeedBlob("data", data, device_option=device_opts)
            workspace.FeedBlob("label", label, device_option=device_opts)

            workspace.RunNet(test_model.net, 1) 
            loss_test.append(workspace.FetchBlob("loss"))
            acc_test.append(workspace.FetchBlob("accuracy"))
        lr = workspace.FetchBlob("SgdOptimizer_0_lr_gpu0")
        print("epochs: {:3}, lr: {:.4f}, train_loss: {:.4f}, train_acc: {:.4f}, test_loss: {:.4f}, test_acc: {:.4f}".format(
            e, lr, np.mean(loss_train),np.mean(acc_train), np.mean(loss_test),np.mean(acc_test)))

    print('training done')

    # save net to forward !!
    deploy_model= model_helper.ModelHelper(name="deploy_net", init_params=False)
    # create_lenet(deploy_model, device_opts=device_opts, is_test=True)
    create_resnet_32x32(
        model=deploy_model,
        data='data', 
        num_input_channels=3,
        num_groups=5,
        num_labels=10, 
        device_opts=device_opts,
        is_test=True)
    workspace.RunNetOnce(deploy_model.param_init_net)
    workspace.CreateNet(deploy_model.net, overwrite=True)

    save_net(init_net_pb, predict_net_pb, deploy_model)

def do_test():
    print ('\n== loading deploy model to test ==')
    workspace.ResetWorkspace()
    load_net(INIT_NET, PREDICT_NET, device_opts=device_opts)
     
    test_batch = 10
    data, label = next_batch_random(test_batch, test_x, test_y)
    workspace.FeedBlob("data", data, device_option=device_opts)
    workspace.RunNet('deploy_net', 1)

    print ('== done. ==')

    print ("\nInput: ones")
    print ("Output:", workspace.FetchBlob("softmax"))
    print ("Output class: ", np.argmax(workspace.FetchBlob("softmax"),axis=1))
    print ("Real class  : ", label)

if __name__ == '__main__':
    
    # 1. Set global init level & Device Option: CUDA or CPU
    core.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    if USE_GPU:
        device_opts = core.DeviceOption(caffe2_pb2.CUDA, GPU_ID)  
    else:
        device_opts = core.DeviceOption(caffe2_pb2.CPU, 0)

    # 2. Prepare data
    # try to download & extract
    # then do shuffle & -std/mean normalization
    train_x, train_y, test_x, test_y = prepare_data()
    train_x, test_x = color_preprocessing(train_x, test_x)

    # 3. Start training & save pb files.
    do_train(
        INIT_NET,
        PREDICT_NET,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        device_opts=device_opts,
        )

    # 4. Do a test if you need
    do_test()

