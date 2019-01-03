#!/usr/bin/env python
# -*- coding:utf-8 -*-

from caffe2.proto import caffe2_pb2
from caffe2.python.predictor import mobile_exporter
from caffe2.python import (
    core,
    brew,
    utils,
    workspace,
    model_helper,
    optimizer,
    net_drawer,
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
parser.add_argument('--groups', type=int, default=3, metavar='NUMBER',
                    help='the depth of network(default: 3) [total layers = groups * 6 + 2]')
parser.add_argument('--learning_rate', type=float, default=0.1, metavar='NUMBER',
                    help='learning rate(default: 0.1)')
parser.add_argument('--batch_size', type=int, default=128, metavar='NUMBER',
                    help='batch size(default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='NUMBER',
                    help='epochs(default: 200)')
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


def add_sortmax(model, last_out, device_opts):
    with core.DeviceScope(device_opts):
        softmax = brew.softmax(model, last_out, 'softmax')
        return softmax


def add_softmax_with_loss(model, last_out, device_opts):
    with core.DeviceScope(device_opts):
        softmax, loss = model.net.SoftmaxWithLoss(
            [last_out, "label"], ["softmax", "loss"])
        return softmax, loss


def add_accuracy(model, softmax, device_opts):
    with core.DeviceScope(device_opts):
        return brew.accuracy(model, [softmax, "label"], "accuracy")


def get_lr_blob_name(opt, use_gpu):
    if use_gpu:
        lr_blob_name = opt.get_gpu_blob_name('lr', 0, '')
    else:
        lr_blob_name = opt.get_gpu_blob_name('lr')
    return workspace.FetchBlob(lr_blob_name)


def add_training_operators(model, last_out, device_opts):

    with core.DeviceScope(device_opts):

        softmax, loss = add_softmax_with_loss(model, last_out, device_opts)
        _ = add_accuracy(model, softmax, device_opts)

        model.AddGradientOperators([loss])

        stepsz = int(60 * config.TRAIN_IMAGES / args.batch_size)
        opt = optimizer.build_sgd(
            model,
            base_learning_rate=args.learning_rate,
            policy="step",
            stepsize=stepsz,
            gamma=0.1,
            weight_decay=1e-4,
            momentum=0.9,
            nesterov=1,
        )

        # [Optional] feel free to use adam or other optimizers
        # opt = optimizer.build_adam(
        #     model,
        #     base_learning_rate=1e-3,
        #     weight_decay=1e-4,
        #     )
        return opt


def save_net(init_net_pb, predict_net_pb, model):
    extra_params = [b for b in workspace.blobs if b.endswith(
        "_rm") or b.endswith("_riv")]
    for name in extra_params:
        model.params.append(name)

    init_net, _ = mobile_exporter.Export(
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
    # return network's name ('deploy')
    return net_def.name


def train_one_epoch(model, train_x, train_y):
    loss_sum = 0.0
    correct = 0.0
    batch_num = config.TRAIN_IMAGES // args.batch_size + 1
    for _ in range(0, batch_num):
        data, label = next_batch_random(args.batch_size, train_x, train_y)
        if args.use_augmentation:
            data = data_augmentation(data)

        workspace.FeedBlob("data", data, device_option=device_opts)
        workspace.FeedBlob("label", label, device_option=device_opts)
        workspace.RunNet(model.net)

        loss_sum += workspace.FetchBlob("loss")
        correct += workspace.FetchBlob("accuracy")

    return {
        'loss': loss_sum / batch_num,
        'accuracy': correct / batch_num * 100.0,
    }


def do_evaluate(model, test_x, test_y, device_opts):
    loss_sum = 0.0
    correct = 0.0
    batch_num = config.TEST_IMAGES // 200

    for i in range(0, batch_num):
        data, label = next_batch(
            i, 200, test_x, test_y, config.TEST_IMAGES)
        workspace.FeedBlob("data", data, device_option=device_opts)
        workspace.FeedBlob("label", label, device_option=device_opts)

        workspace.RunNet(model.net)

        loss_sum += workspace.FetchBlob("loss")
        correct += workspace.FetchBlob("accuracy")

    return {
        'loss': loss_sum / batch_num,
        'accuracy': correct / batch_num * 100.0,
    }


def do_train(
        train_x,
        train_y,
        test_x,
        test_y,
        device_opts
):

    data, label = dummy_input()
    workspace.FeedBlob("data", data, device_option=device_opts)
    workspace.FeedBlob("label", label, device_option=device_opts)

    train_arg_scope = {'order': 'NCHW', 'use_cudnn': True, }

    train_model = model_helper.ModelHelper(
        name="train_net", arg_scope=train_arg_scope)
    last_out = create_resnet(
        model=train_model,
        data='data',
        num_input_channels=config.IMG_CHANNELS,
        num_groups=args.groups,
        num_labels=config.CLASS_NUM,
        device_opts=device_opts,
        is_test=False
    )
    opt = add_training_operators(
        train_model, last_out, device_opts=device_opts)

    test_model = model_helper.ModelHelper(name="test_net", init_params=False)
    last_out = create_resnet(
        model=test_model,
        data='data',
        num_input_channels=config.IMG_CHANNELS,
        num_groups=args.groups,
        num_labels=config.CLASS_NUM,
        device_opts=device_opts,
        is_test=True
    )
    softmax, _ = add_softmax_with_loss(test_model, last_out, device_opts)
    add_accuracy(test_model, softmax, device_opts)

    workspace.RunNetOnce(train_model.param_init_net)
    workspace.CreateNet(train_model.net)

    workspace.RunNetOnce(test_model.param_init_net)
    workspace.CreateNet(test_model.net, overwrite=True)

    # print(workspace.Blobs())

    print('\n== Training for', args.epochs, 'epochs ==\n')
    columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'time']

    for e in range(0, args.epochs):

        time_ep = time.time()

        train_res = train_one_epoch(train_model, train_x, train_y)

        if e == 0 or e % args.eval_freq == 0 or e == args.epochs - 1:
            test_res = do_evaluate(test_model, test_x, test_y, device_opts)
        else:
            test_res = {'loss': None, 'accuracy': None}

        time_ep = time.time() - time_ep
        lr = get_lr_blob_name(opt, args.use_gpu)
        values = [
            e + 1,
            lr,
            train_res['loss'],
            train_res['accuracy'],
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
    print('== Training done. ==')

    print('== Save deploy model ==')
    save_deploy_model(device_opts)
    print('== Done. ==')


def save_deploy_model(device_opts):
    # save net forward only !!
    deploy_model = model_helper.ModelHelper(
        name="deploy_net", init_params=False)
    last_out = create_resnet(
        model=deploy_model,
        data='data',
        num_input_channels=config.IMG_CHANNELS,
        num_groups=args.groups,
        num_labels=config.CLASS_NUM,
        device_opts=device_opts,
        is_test=True)
    add_sortmax(deploy_model, last_out, device_opts)

    workspace.RunNetOnce(deploy_model.param_init_net)
    workspace.CreateNet(deploy_model.net, overwrite=True)

    #
    # save network images
    # try to install the following packages if you need
    # sudo apt install python-pydot python-pydot-ng graphviz -y
    # sudo pip3 install pydot
    #

    # graph = net_drawer.GetPydotGraphMinimal(deploy_model)
    # graph.write_svg('net.svg')

    save_net(args.init_net, args.predict_net, deploy_model)


def do_test(test_x, test_y, device_opts):
    print('\n== loading deploy model to test ==')
    workspace.ResetWorkspace()
    net_name = load_net(args.init_net, args.predict_net,
                        device_opts=device_opts)

    test_batch = 10
    data, label = next_batch_random(test_batch, test_x, test_y)
    workspace.FeedBlob("data", data, device_option=device_opts)
    workspace.RunNet(net_name)

    print('== done. ==')
    print("input shape :", data.shape)
    # feel free to see the result
    # print ("Output last_out:\n", workspace.FetchBlob("last_out"))
    # print ("Output softmax:\n", workspace.FetchBlob("softmax"))
    print("Output class: ", np.argmax(workspace.FetchBlob("softmax"), axis=1))
    print("Real class  : ", label)


if __name__ == '__main__':

    # 1. Set global init level & Device Option: CUDA or CPU
    core.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    workspace.ResetWorkspace()
    if args.use_gpu:
        device_opts = core.DeviceOption(caffe2_pb2.CUDA, 0)
    else:
        device_opts = core.DeviceOption(caffe2_pb2.CPU, 0)

    # 2. Prepare data
    # try to download & extract
    # then do shuffle & -std/mean normalization
    train_x, train_y, test_x, test_y = prepare_data()
    train_x, test_x = normalization(train_x, test_x)

    # 3. Start training & save pb files.
    do_train(
        train_x,
        train_y,
        test_x,
        test_y,
        device_opts,
    )

    # 4. Do a test if you need
    do_test(test_x, test_y, device_opts)
