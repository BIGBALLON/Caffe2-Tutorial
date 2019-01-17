#!/usr/bin/env python
# -*- coding:utf-8 -*-

from caffe2.proto import caffe2_pb2
from caffe2.python.predictor import mobile_exporter
from caffe2.python import (
    core,
    brew,
    cnn,
    utils,
    workspace,
    model_helper,
    optimizer,
    net_drawer,
    data_parallel_model
)
from data_utility import (
    prepare_data,
    next_batch,
    dummy_input,
    normalization,
    next_batch_random,
    data_augmentation,
)
from models import create_resnet
import time
import tabulate
import argparse
import numpy as np
import config

parser = argparse.ArgumentParser(description='Caffe2 CIFAR-10 DEMO CODE')
parser.add_argument('--gpus', type=int, default=1, metavar='NUMBER',
                    help='gpu number(default: 1)')
parser.add_argument('--groups', type=int, default=3, metavar='NUMBER',
                    help='the depth of network(default: 3) [total layers = groups * 6 + 2]')
parser.add_argument('--learning_rate', type=float, default=0.1, metavar='NUMBER',
                    help='learning rate(default: 0.1)')
parser.add_argument('--batch_size', type=int, default=128, metavar='NUMBER',
                    help='batch size(default: 128)')
parser.add_argument('--epochs', type=int, default=180, metavar='NUMBER',
                    help='epochs(default: 180)')
parser.add_argument('--eval_freq', type=int, default=4, metavar='NUMBER',
                    help='the number of evaluate interval(default: 4)')
parser.add_argument('--use_gpu', type=bool, default=True, metavar='BOOL',
                    help='use gpu or not (default: True)')
parser.add_argument('--use_augmentation', type=bool, default=True, metavar='BOOL',
                    help='use augmentation or not(default: True)')
parser.add_argument('--init_net', type=str,
                    default='./init_net.pb', metavar='STRING')
parser.add_argument('--predict_net', type=str,
                    default='./predict_net.pb', metavar='STRING')

args = parser.parse_args()
print("\n=============== Argument ===============\n")
print(args)
print("\n=============== Argument ===============")


def input_builder_fun(model):
    return None


def add_accuracy(model, softmax):
    return brew.accuracy(model, [softmax, "label"], "accuracy")


def add_softmax_with_loss(model, last_out):
    softmax, loss = model.net.SoftmaxWithLoss(
        [last_out, "label"], ["softmax", "loss"])
    return softmax, loss


def model_build_fun_train(model, loss_scale):
    last_out = create_resnet(
        model=model,
        data="data",
        num_input_channels=config.IMG_CHANNELS,
        num_groups=args.groups,
        num_labels=config.CLASS_NUM,
        is_test=False
    )
    softmax, loss = add_softmax_with_loss(model, last_out)
    loss = model.Scale(loss, scale=loss_scale)
    add_accuracy(model, softmax)
    return [loss]


def model_build_fun_test(model, loss_scale):
    last_out = create_resnet(
        model=model,
        data="data",
        num_input_channels=config.IMG_CHANNELS,
        num_groups=args.groups,
        num_labels=config.CLASS_NUM,
        is_test=True
    )
    softmax, loss = add_softmax_with_loss(model, last_out)
    loss = model.Scale(loss, scale=loss_scale)
    add_accuracy(model, softmax)
    return [loss]


def model_build_fun_deploy(model, loss_scale):
    last_out = create_resnet(
        model=model,
        data='data',
        num_input_channels=config.IMG_CHANNELS,
        num_groups=args.groups,
        num_labels=config.CLASS_NUM,
        is_test=True
    )
    brew.softmax(model, last_out, "softmax")
    return []


def add_optimizer(model):
    stepsz = int(60 * config.TRAIN_IMAGES / args.batch_size / args.gpus)
    return optimizer.build_sgd(
        model,
        base_learning_rate=args.learning_rate,
        policy="step",
        stepsize=stepsz,
        gamma=0.1,
        weight_decay=1e-4,
        momentum=0.9,
        nesterov=1,
    )


if __name__ == '__main__':

    # 1. Set global init level & Device Option: CUDA or CPU
    core.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    workspace.ResetWorkspace()

    train_x, train_y, test_x, test_y = prepare_data()
    train_x, test_x = normalization(train_x, test_x)

    devices = list(range(0, args.gpus))
    print(devices)
    arg_scope = {'order': 'NCHW', 'use_cudnn': True, }

    train_model = model_helper.ModelHelper(
        name="train_net", arg_scope=arg_scope)

    data_parallel_model.Parallelize_GPU(
        train_model,
        input_builder_fun=input_builder_fun,
        forward_pass_builder_fun=model_build_fun_train,
        optimizer_builder_fun=add_optimizer,
        devices=devices,
    )

    test_model = model_helper.ModelHelper(
        name="test_net", arg_scope=arg_scope, init_params=False)

    data_parallel_model.Parallelize_GPU(
        test_model,
        input_builder_fun=input_builder_fun,
        forward_pass_builder_fun=model_build_fun_test,
        optimizer_builder_fun=None,
        devices=devices,
    )

    deploy_model = model_helper.ModelHelper(
        name="deploy_net", arg_scope=arg_scope, init_params=False)

    data_parallel_model.Parallelize_GPU(
        deploy_model,
        input_builder_fun=input_builder_fun,
        forward_pass_builder_fun=model_build_fun_deploy,
        optimizer_builder_fun=None,
        devices=[0],
    )

    # Each run has same input, independent of number of gpus
    columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'time']
    batch_per_device = args.batch_size // len(devices)
    for e in range(0, args.epochs):
        time_ep = time.time()
        train_res = {}
        loss_sum = 0.0
        correct = 0.0

        test_res = {'loss': None, 'accuracy': None}
        batch_num = config.TRAIN_IMAGES // args.batch_size + 1
        for i in range(0, batch_num):
            data, labels = next_batch_random(
                args.batch_size, train_x, train_y)
            for (j, g) in enumerate(devices):
                st = j * batch_per_device
                en = st + batch_per_device
                data_device = data[st:en]
                labels_device = labels[st:en]
                with core.DeviceScope(core.DeviceOption(train_model._device_type, g)):
                    workspace.FeedBlob(
                        "{}_{}/data".format(train_model._device_prefix,
                                            g), data_device
                    )
                    workspace.FeedBlob(
                        "{}_{}/label".format(train_model._device_prefix,
                                             g), labels_device
                    )
            if i == 0 and e == 0:
                workspace.RunNetOnce(train_model.param_init_net)
                workspace.CreateNet(train_model.net)
                workspace.RunNetOnce(test_model.param_init_net)
                workspace.CreateNet(test_model.net, overwrite=True)
                workspace.RunNetOnce(deploy_model.param_init_net)
                workspace.CreateNet(deploy_model.net, overwrite=True)

            workspace.RunNet(train_model.net.Proto().name)
            loss_sum += workspace.FetchBlob("gpu_0/loss")
            correct += workspace.FetchBlob("gpu_0/accuracy")

        time_ep = time.time() - time_ep
        lr = workspace.FetchBlob(
            data_parallel_model.GetLearningRateBlobNames(train_model)[0])

        values = [
            e + 1,
            lr,
            loss_sum / batch_num,
            correct / batch_num,
            test_res['loss'],
            test_res['accuracy'],
            time_ep,
        ]
        table = tabulate.tabulate(
            [values], columns, tablefmt='simple', floatfmt='8.4f')
        if e % 25 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)

    checkpoint_params = data_parallel_model.GetCheckpointParams(train_model)

    init_net, _ = mobile_exporter.Export(
        workspace,
        deploy_model.net,
        checkpoint_params
    )
    with open("predict_net.pb", 'wb') as f:
        f.write(deploy_model.net._net.SerializeToString())
    with open("init_net.pb", 'wb') as f:
        f.write(init_net.SerializeToString())
