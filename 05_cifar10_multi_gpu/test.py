from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from future.utils import viewkeys
from multiprocessing import Process, Queue
import numpy as np
import os
import shutil
import tempfile
import unittest
import time
from mock import Mock
from hypothesis import assume, given
import hypothesis.strategies as st

from caffe2.proto import caffe2_pb2
from caffe2.python import brew, core, cnn, data_parallel_model, dyndep, \
    model_helper, optimizer, rnn_cell, workspace
from caffe2.python.test_util import TestCase


def run_model(devices):
    '''
    Helper function for test_equiv
    '''
    def input_builder_fun(model):
        return None

    def model_build_fun(model, loss_scale):
        fc = model.FC("data", "fc", 16, 1,
                      ("ConstantFill", {}), ("ConstantFill", {}))
        fc_fl = model.FlattenToVec(fc, "fc_fl")
        sigm = model.Sigmoid(fc_fl, "sigm")
        sq = model.SquaredL2Distance([sigm, "label"], "sq")
        loss = model.AveragedLoss(sq, "loss")
        loss = model.Scale(loss, scale=loss_scale)
        return [loss]

    def add_optimizer(model):
        return optimizer.build_sgd(
            model,
            0.1,
            policy="fixed",
            max_gradient_norm=5.0,
            allow_lr_injection=True,
        )

    workspace.ResetWorkspace()
    model = cnn.CNNModelHelper(
        order="NHWC",
        name="test{}".format(devices),
    )
    data_parallel_model.Parallelize_GPU(
        model,
        input_builder_fun=input_builder_fun,
        forward_pass_builder_fun=model_build_fun,
        optimizer_builder_fun=add_optimizer,
        devices=devices,
    )

    # Each run has same input, independent of number of gpus
    batch_size = 64
    for i in range(0, 10):
        full_data = np.random.rand(batch_size, 16)
        full_labels = np.round(full_data[:, 0])
        batch_per_device = batch_size // len(devices)

        for (j, g) in enumerate(devices):
            st = j * batch_per_device
            en = st + batch_per_device
            data = full_data[st:en, :].astype(np.float32)
            labels = full_labels[st:en].astype(np.float32)
            with core.DeviceScope(core.DeviceOption(model._device_type, g)):
                workspace.FeedBlob(
                    "{}_{}/data".format(model._device_prefix, g), data
                )
                workspace.FeedBlob(
                    "{}_{}/label".format(model._device_prefix, g), labels
                )

        if i == 0:
            workspace.RunNetOnce(model.param_init_net)
            workspace.CreateNet(model.net)

        workspace.RunNet(model.net.Proto().name)

    return workspace.FetchBlob("{}_0/fc_w".format(model._device_prefix))


run_model([0, 1, 2, 3])
# run_model([0], False)
