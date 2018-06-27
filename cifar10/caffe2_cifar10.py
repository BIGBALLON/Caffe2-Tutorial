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
    )
import numpy as np

BATCH_SIZE = 128
EPOCHS = 180

class ResNetBuilder():

    def __init__(self, model, prev_blob, no_bias, is_test, spatial_bn_mom=0.9):
        self.model = model
        self.comp_count = 0
        self.comp_idx = 0
        self.prev_blob = prev_blob
        self.is_test = is_test
        self.spatial_bn_mom = spatial_bn_mom
        self.no_bias = 1 if no_bias else 0

    def add_conv(self, in_filters, out_filters, kernel, stride=1, pad=0):
        self.comp_idx += 1
        self.prev_blob = brew.conv(
            self.model,
            self.prev_blob,
            'comp_%d_conv_%d' % (self.comp_count, self.comp_idx),
            in_filters,
            out_filters,
            weight_init=("MSRAFill", {}),
            kernel=kernel,
            stride=stride,
            pad=pad,
            no_bias=self.no_bias,
        )
        return self.prev_blob

    def add_relu(self):
        self.prev_blob = brew.relu(
            self.model,
            self.prev_blob,
            self.prev_blob,  # in-place
        )
        return self.prev_blob

    def add_spatial_bn(self, num_filters):
        self.prev_blob = brew.spatial_bn(
            self.model,
            self.prev_blob,
            'comp_%d_spatbn_%d' % (self.comp_count, self.comp_idx),
            num_filters,
            epsilon=1e-3,
            momentum=self.spatial_bn_mom,
            is_test=self.is_test,
        )
        return self.prev_blob

    '''
    Add a "bottleneck" component as decribed in He et. al. Figure 3 (right)
    '''

    def add_bottleneck(
        self,
        input_filters,   # num of feature maps from preceding layer
        base_filters,    # num of filters internally in the component
        output_filters,  # num of feature maps to output
        down_sampling=False,
        spatial_batch_norm=True,
    ):
        self.comp_idx = 0
        shortcut_blob = self.prev_blob

        # 1x1
        self.add_conv(
            input_filters,
            base_filters,
            kernel=1,
            stride=1
        )

        if spatial_batch_norm:
            self.add_spatial_bn(base_filters)

        self.add_relu()

        # 3x3 (note the pad, required for keeping dimensions)
        self.add_conv(
            base_filters,
            base_filters,
            kernel=3,
            stride=(1 if down_sampling is False else 2),
            pad=1
        )

        if spatial_batch_norm:
            self.add_spatial_bn(base_filters)
        self.add_relu()

        # 1x1
        last_conv = self.add_conv(base_filters, output_filters, kernel=1)
        if spatial_batch_norm:
            last_conv = self.add_spatial_bn(output_filters)

        # Summation with input signal (shortcut)
        # If we need to increase dimensions (feature maps), need to
        # do a projection for the short cut
        if (output_filters > input_filters):
            shortcut_blob = brew.conv(
                self.model,
                shortcut_blob,
                'shortcut_projection_%d' % self.comp_count,
                input_filters,
                output_filters,
                weight_init=("MSRAFill", {}),
                kernel=1,
                stride=(1 if down_sampling is False else 2),
                no_bias=self.no_bias,
            )
            if spatial_batch_norm:
                shortcut_blob = brew.spatial_bn(
                    self.model,
                    shortcut_blob,
                    'shortcut_projection_%d_spatbn' % self.comp_count,
                    output_filters,
                    epsilon=1e-3,
                    momentum=self.spatial_bn_mom,
                    is_test=self.is_test,
                )

        self.prev_blob = brew.sum(
            self.model, [shortcut_blob, last_conv],
            'comp_%d_sum_%d' % (self.comp_count, self.comp_idx)
        )
        self.comp_idx += 1
        self.add_relu()

        # Keep track of number of high level components if this ResNetBuilder
        self.comp_count += 1

    def add_simple_block(
        self,
        input_filters,
        num_filters,
        down_sampling=False,
        spatial_batch_norm=True
    ):
        self.comp_idx = 0
        shortcut_blob = self.prev_blob

        # 3x3
        self.add_conv(
            input_filters,
            num_filters,
            kernel=3,
            stride=(1 if down_sampling is False else 2),
            pad=1
        )

        if spatial_batch_norm:
            self.add_spatial_bn(num_filters)
        self.add_relu()

        last_conv = self.add_conv(num_filters, num_filters, kernel=3, pad=1)
        if spatial_batch_norm:
            last_conv = self.add_spatial_bn(num_filters)

        # Increase of dimensions, need a projection for the shortcut
        if (num_filters != input_filters):
            shortcut_blob = brew.conv(
                self.model,
                shortcut_blob,
                'shortcut_projection_%d' % self.comp_count,
                input_filters,
                num_filters,
                weight_init=("MSRAFill", {}),
                kernel=1,
                stride=(1 if down_sampling is False else 2),
                no_bias=self.no_bias,
            )
            if spatial_batch_norm:
                shortcut_blob = brew.spatial_bn(
                    self.model,
                    shortcut_blob,
                    'shortcut_projection_%d_spatbn' % self.comp_count,
                    num_filters,
                    epsilon=1e-3,
                    is_test=self.is_test,
                )

        self.prev_blob = brew.sum(
            self.model, [shortcut_blob, last_conv],
            'comp_%d_sum_%d' % (self.comp_count, self.comp_idx)
        )
        self.comp_idx += 1
        self.add_relu()

        # Keep track of number of high level components if this ResNetBuilder
        self.comp_count += 1


# The conv1 and final_avg kernel/stride args provide a basic mechanism for
# adapting resnet50 for different sizes of input images.
def create_resnet50(
    model,
    data,
    num_input_channels,
    num_labels,
    label=None,
    is_test=False,
    no_loss=False,
    no_bias=0,
    conv1_kernel=7,
    conv1_stride=2,
    final_avg_kernel=7,
):
    # conv1 + maxpool
    brew.conv(
        model,
        data,
        'conv1',
        num_input_channels,
        64,
        weight_init=("MSRAFill", {}),
        kernel=conv1_kernel,
        stride=conv1_stride,
        pad=3,
        no_bias=no_bias
    )

    brew.spatial_bn(
        model,
        'conv1',
        'conv1_spatbn_relu',
        64,
        epsilon=1e-3,
        momentum=0.1,
        is_test=is_test
    )
    brew.relu(model, 'conv1_spatbn_relu', 'conv1_spatbn_relu')
    brew.max_pool(model, 'conv1_spatbn_relu', 'pool1', kernel=3, stride=2)

    # Residual blocks...
    builder = ResNetBuilder(model, 'pool1', no_bias=no_bias,
                            is_test=is_test, spatial_bn_mom=0.1)

    # conv2_x (ref Table 1 in He et al. (2015))
    builder.add_bottleneck(64, 64, 256)
    builder.add_bottleneck(256, 64, 256)
    builder.add_bottleneck(256, 64, 256)

    # conv3_x
    builder.add_bottleneck(256, 128, 512, down_sampling=True)
    for _ in range(1, 4):
        builder.add_bottleneck(512, 128, 512)

    # conv4_x
    builder.add_bottleneck(512, 256, 1024, down_sampling=True)
    for _ in range(1, 6):
        builder.add_bottleneck(1024, 256, 1024)

    # conv5_x
    builder.add_bottleneck(1024, 512, 2048, down_sampling=True)
    builder.add_bottleneck(2048, 512, 2048)
    builder.add_bottleneck(2048, 512, 2048)

    # Final layers
    final_avg = brew.average_pool(
        model,
        builder.prev_blob,
        'final_avg',
        kernel=final_avg_kernel,
        stride=1,
        global_pooling=True,
    )

    # Final dimension of the "image" is reduced to 7x7
    last_out = brew.fc(
        model, final_avg, 'last_out_L{}'.format(num_labels), 2048, num_labels
    )

    if no_loss:
        return last_out

    # If we create model for training, use softmax-with-loss
    if (label is not None):
        (softmax, loss) = model.SoftmaxWithLoss(
            [last_out, label],
            ["softmax", "loss"],
        )

        return (softmax, loss)
    else:
        # For inference, we just return softmax
        return brew.softmax(model, last_out, "softmax")


def create_resnet_32x32(
    model, data, num_input_channels, num_groups, num_labels, device_opts, is_test=False,
):
    with core.DeviceScope(device_opts):
        '''
        Create residual net for smaller images (sec 4.2 of He et. al (2015))
        num_groups = 'n' in the paper
        '''
        # conv1 + maxpool
        brew.conv(
            model, data, 'conv1', num_input_channels, 16, kernel=3, stride=1
        )
        brew.spatial_bn(
            model, 'conv1', 'conv1_spatbn', 16, epsilon=1e-3, is_test=is_test
        )
        brew.relu(model, 'conv1_spatbn', 'relu1')

        # Number of blocks as described in sec 4.2
        filters = [16, 32, 64]

        builder = ResNetBuilder(model, 'relu1', no_bias=0, is_test=is_test)
        prev_filters = 16
        for groupidx in range(0, 3):
            for blockidx in range(0, 2 * num_groups):
                builder.add_simple_block(
                    prev_filters if blockidx == 0 else filters[groupidx],
                    filters[groupidx],
                    down_sampling=(True if blockidx == 0 and
                                   groupidx > 0 else False))
                    
            prev_filters = filters[groupidx]

        # Final layers
        brew.average_pool(
            model, builder.prev_blob, 'final_avg', kernel=8, stride=1
        )
        brew.fc(model, 'final_avg', 'last_out', 64, num_labels)
        softmax = brew.softmax(model, 'last_out', 'softmax')
        return softmax

def create_model(model, device_opts, is_test=False) :
    with core.DeviceScope(device_opts):
        conv1 = brew.conv(
            model, 
            'data', 
            'conv1', 
            dim_in=3, 
            dim_out=32, 
            weight_init=('MSRAFill', {}),
            kernel=5, 
            stride=1, 
            pad=0)
        brew.spatial_bn(
            model, 'conv1', 'conv1_spatbn', 32, epsilon=1e-3, is_test=is_test
        )
        relu1 = brew.relu(model, 'conv1_spatbn', 'relu1')
        # relu1 = brew.relu(model, conv1, 'relu1')
        pool1 = brew.max_pool(model, relu1, 'pool1', kernel=2, stride=2)
        
        conv2 = brew.conv(
            model, 
            pool1, 
            'conv2', 
            dim_in=32, 
            dim_out=64, 
            weight_init=('MSRAFill', {}),
            kernel=5, 
            stride=1, 
            pad=0)
        brew.spatial_bn(
            model, 'conv2', 'conv2_spatbn', 64, epsilon=1e-3, is_test=is_test
        )
        relu2 = brew.relu(model, 'conv2_spatbn', 'relu2')
        # relu2 = brew.relu(model, conv2, 'relu2')
        pool2 = brew.max_pool(model, relu2, 'pool2', kernel=2, stride=2)
        
        # Fully connected layers
        fc1 = brew.fc(model, pool2, 'fc1', dim_in=64*5*5, dim_out=256)
        relu3 = brew.relu(model, fc1, 'relu3')
        
        fc2 = brew.fc(model, relu3, 'fc2', dim_in=256, dim_out=256)
        relu4 = brew.relu(model, fc2, 'relu4')

        fc3 = brew.fc(model, relu4, 'fc3', dim_in=256, dim_out=10)
        # Softmax layer
        softmax = brew.softmax(model, fc3, 'softmax')
        # model.net.AddExternalOutput(softmax)
        return softmax

def add_accuracy(model, softmax, label, device_opts):
    with core.DeviceScope(device_opts):
        accuracy = brew.accuracy(model, [softmax, label], "accuracy")
        return accuracy

# add loss and optimizer
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
            stepsize=50000 * 100 // BATCH_SIZE, 
            weight_decay=1e-4,
            momentum=0.9, 
            gamma=0.1,
            ) 

def train(INIT_NET, PREDICT_NET, epochs, batch_size, device_opts) :

    data, label = next_batch(batch_size,train_x, train_y)
    workspace.FeedBlob("data", data, device_option=device_opts)
    workspace.FeedBlob("label", label, device_option=device_opts)

    train_model= model_helper.ModelHelper(name="train_net",arg_scope={"order": "NCHW"})
    softmax = create_model(train_model, device_opts=device_opts)
    # softmax = create_resnet_32x32(
    #     model=train_model,
    #     data='data', 
    #     num_input_channels=3,
    #     num_groups=3,
    #     num_labels=10, 
    #     device_opts=device_opts,
    #     is_test=False)

    add_training_operators(train_model, softmax, device_opts=device_opts)


    test_model= model_helper.ModelHelper(name="test_net", init_params=False)
    softmax = create_model(test_model, device_opts=device_opts, is_test=True)
    add_accuracy(test_model, softmax, "label", device_opts)
    # create_resnet_32x32(
    #     model=test_model,
    #     data='data', 
    #     num_input_channels=3,
    #     num_groups=3,
    #     num_labels=10, 
    #     device_opts=device_opts,
    #     is_test=True)

    workspace.RunNetOnce(train_model.param_init_net)
    workspace.CreateNet(train_model.net)

    workspace.RunNetOnce(test_model.param_init_net)
    workspace.CreateNet(test_model.net, overwrite=True)
    
    print(workspace.Blobs())

    print('\ntraining for', epochs, 'epochs')

    for e in range(0, epochs):
        loss_train = []
        acc_train  = []
        loss_test  = []
        acc_test   = []
        
        for i in range(0, 50000//batch_size + 1):
            data, label = next_batch(batch_size, train_x, train_y)
            data_augmentation(data)

            workspace.FeedBlob("data", data, device_option=device_opts)
            workspace.FeedBlob("label", label, device_option=device_opts)

            workspace.RunNet(train_model.net, 1) 
            loss_train.append(workspace.FetchBlob("loss"))
            acc_train.append(workspace.FetchBlob("accuracy"))

        for i in range(0, 10000//1000):
            data, label = next_batch(1000, test_x, test_y)
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
    create_model(deploy_model, device_opts=device_opts, is_test=True)

    workspace.RunNetOnce(test_model.param_init_net)
    workspace.CreateNet(test_model.net, overwrite=True)

    save_net(INIT_NET, PREDICT_NET, deploy_model)

def save_net(INIT_NET, PREDICT_NET, model) :

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
        workspace, model.net, model.params
    )
    
    with open(PREDICT_NET, 'wb') as f:
        f.write(model.net._net.SerializeToString())
    with open(INIT_NET, 'wb') as f:
        f.write(init_net.SerializeToString())

def load_net(INIT_NET, PREDICT_NET, device_opts):
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

INIT_NET = './init_net.pb'
PREDICT_NET = './predict_net.pb'


core.GlobalInit(['caffe2', '--caffe2_log_level=0'])
workspace.ResetWorkspace()

device_opts = core.DeviceOption(caffe2_pb2.CUDA, 0) 

train_x, train_y, test_x, test_y = prepare_data()
train_x, test_x = color_preprocessing(train_x, test_x)


train(INIT_NET, PREDICT_NET, epochs=EPOCHS, batch_size=BATCH_SIZE, device_opts=device_opts)

print ('\n********************************************')
print ('loading test model')


load_net(INIT_NET, PREDICT_NET, device_opts=device_opts)


data, label = next_batch(10,test_x, test_y)
workspace.FeedBlob("data", data, device_option=device_opts)

workspace.RunNet('deploy_net', 1)

print ('done')

print ("\nInput: ones")
print ("Output:", workspace.FetchBlob("softmax"))
print ("Output class: ", np.argmax(workspace.FetchBlob("softmax"),axis=1))
print ("Real class  : ", label)