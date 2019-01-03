#!/usr/bin/env python
# -*- coding:utf-8 -*-
from caffe2.python import core, brew


class ResNetBuilder():

    def __init__(self, model, prev_blob, no_bias, is_test):
        self.model = model
        self.comp_count = 0
        self.comp_idx = 0
        self.prev_blob = prev_blob
        self.is_test = is_test
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
            is_test=self.is_test,
        )
        return self.prev_blob

    def add_simple_block(
        self,
        input_filters,
        num_filters,
        down_sampling=False,
        spatial_batch_norm=True,
    ):
        self.comp_idx = 0
        shortcut_blob = self.prev_blob

        if spatial_batch_norm:
            self.add_spatial_bn(input_filters)
        pre_relu = self.add_relu()

        self.add_conv(
            input_filters,
            num_filters,
            kernel=3,
            stride=(1 if down_sampling is False else 2),
            pad=1,
        )

        if spatial_batch_norm:
            self.add_spatial_bn(num_filters)
        self.add_relu()

        last_conv = self.add_conv(num_filters, num_filters, kernel=3, pad=1)

        # Increase of dimensions, need a projection for the shortcut
        if (num_filters != input_filters):
            shortcut_blob = brew.conv(
                self.model,
                pre_relu,
                'shortcut_projection_%d' % self.comp_count,
                input_filters,
                num_filters,
                weight_init=("MSRAFill", {}),
                kernel=1,
                stride=(1 if down_sampling is False else 2),
                no_bias=self.no_bias,
            )

        self.prev_blob = brew.sum(
            self.model,
            [shortcut_blob, last_conv],
            'comp_%d_sum_%d' % (self.comp_count, self.comp_idx),
        )
        self.comp_idx += 1
        self.comp_count += 1


def create_resnet(
    model, data,
    num_input_channels,
    num_groups,
    num_labels,
    device_opts,
    is_test=False,
):
    with core.DeviceScope(device_opts):
        filters = [16, 32, 64]
        brew.conv(model, data, 'conv1', num_input_channels,
                  filters[0], no_bias=1, kernel=3, stride=1, pad=1)

        builder = ResNetBuilder(model, 'conv1', no_bias=1, is_test=is_test)

        # input: 32x32x16 output: 32x32x16
        for _ in range(num_groups):
            builder.add_simple_block(
                filters[0], filters[0], down_sampling=False)

        # input: 32x32x16 output: 16x16x32
        builder.add_simple_block(filters[0], filters[1], down_sampling=True)
        for _ in range(1, num_groups):
            builder.add_simple_block(
                filters[1], filters[1], down_sampling=False)

        # input: 16x16x32 output: 8x8x64
        builder.add_simple_block(filters[1], filters[2], down_sampling=True)
        for _ in range(1, num_groups):
            builder.add_simple_block(
                filters[2], filters[2], down_sampling=False)

        brew.spatial_bn(model, builder.prev_blob, 'last_spatbn',
                        filters[2], epsilon=1e-3, is_test=is_test)
        brew.relu(model, 'last_spatbn', 'last_relu')
        # Final layers
        brew.average_pool(model, 'last_relu', 'final_avg', kernel=8, stride=1)
        last_out = brew.fc(model, 'final_avg', 'last_out', 64, num_labels)

        return last_out
