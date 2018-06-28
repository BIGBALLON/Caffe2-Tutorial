from caffe2.python import core, brew

def create_lenet(model, device_opts, is_test=False) :
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
        brew.spatial_bn(model, 'conv2', 'conv2_spatbn', 64, epsilon=1e-3, is_test=is_test)
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
        return softmax

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

        # 3x3
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
        # Keep track of number of high level components if this ResNetBuilder
        self.comp_count += 1


def create_resnet_32x32(
    model, data, num_input_channels, num_groups, num_labels, device_opts, is_test=False,
    ):
    with core.DeviceScope(device_opts):

        brew.conv(model, data, 'conv1', num_input_channels, 16, kernel=3, stride=1)

        filters = [16, 32, 64]

        builder = ResNetBuilder(model, 'conv1', no_bias=0, is_test=is_test)
        prev_filters = 16
        for groupidx in range(0, 3):
            for blockidx in range(0, num_groups):
                builder.add_simple_block(
                    prev_filters if blockidx == 0 else filters[groupidx],
                    filters[groupidx],
                    down_sampling=(True if blockidx == 0 and
                                   groupidx > 0 else False))
                    
            prev_filters = filters[groupidx]

        brew.spatial_bn(model, builder.prev_blob, 'last_spatbn', 64, epsilon=1e-3, is_test=is_test)
        brew.relu(model, 'last_spatbn', 'last_relu')
        # Final layers
        brew.average_pool(
            model, 'last_relu', 'final_avg', kernel=8, stride=1
        )
        brew.fc(model, 'final_avg', 'last_out', 64, num_labels)
        softmax = brew.softmax(model, 'last_out', 'softmax')
        return softmax

