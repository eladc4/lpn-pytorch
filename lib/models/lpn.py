import os
import logging
import math
import torch
import torch.nn as nn
from .lightweight_modules import LW_Bottleneck

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class LPN(nn.Module):

    def __init__(self, block, layers, cfg, **kwargs):
        super(LPN, self).__init__()
        extra = cfg.MODEL.EXTRA

        self.inplanes = 64
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.add_hm_channels = cfg.MODEL.USE_PREV_HM_INPUT
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        self.attention = extra.get('ATTENTION')
        self.fine_tune = cfg.MODEL.FINE_TUNE
        self.num_input_images = cfg.MODEL.NUM_INPUT_IMAGES
        self.new_multi_input_mode = cfg.MODEL.NEW_MULTI_INPUT_MODE
        self.output_activation = cfg.MODEL.OUTPUT_ACTIVATION

        num_input_images = 1 if self.new_multi_input_mode else self.num_input_images
        self.conv1 = nn.Conv2d(num_input_images, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], add_hm_channels=self.add_hm_channels)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=int(extra.FINAL_CONV_KERNEL/2)
        )

        if self.new_multi_input_mode and self.num_input_images > 1:
            self.merge_heatmaps = nn.Sequential(nn.Conv2d(in_channels=cfg.MODEL.NUM_JOINTS * self.num_input_images,
                                                          out_channels=cfg.MODEL.NUM_JOINTS * 3,
                                                          kernel_size=5, stride=1, padding=2),
                                                nn.BatchNorm2d(cfg.MODEL.NUM_JOINTS * 3),
                                                nn.SiLU(),
                                                nn.Conv2d(in_channels=cfg.MODEL.NUM_JOINTS * 3,
                                                          out_channels=cfg.MODEL.NUM_JOINTS,
                                                          kernel_size=3, stride=1, padding=1),
                                                )
        else:
            self.merge_heatmaps = None

    def _make_layer(self, block, planes, blocks, stride=1, add_hm_channels=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        if add_hm_channels:
            layers = [nn.Conv2d(in_channels=self.inplanes+self.num_joints, out_channels=self.inplanes, kernel_size=1)]
        else:
            layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.attention))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, attention=self.attention))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.extend([
                nn.ConvTranspose2d(in_channels=self.inplanes, out_channels=planes, kernel_size=kernel,
                                   stride=2, padding=padding, output_padding=output_padding,
                                   groups=math.gcd(self.inplanes, planes), bias=self.deconv_with_bias),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(planes, planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            ])
            self.inplanes = planes

        return nn.Sequential(*layers)

    def backbone_forward(self, x, hm=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if hm is not None:
            x = torch.cat([x, hm], dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        features = self.deconv_layers(x)
        x = self.final_layer(features)
        if self.output_activation is not None and self.output_activation.lower() == 'sigmoid':
            x = torch.sigmoid(x)
        return x

    def forward(self, x, hm=None, return_inter_hm=False):
        if self.new_multi_input_mode and self.num_input_images > 1:
            x = torch.reshape(x, [-1, 1] + self.input_size)
        if self.fine_tune:
            with torch.no_grad():
                x = self.backbone_forward(x, hm=hm)
        else:
            x = self.backbone_forward(x, hm=hm)

        if self.new_multi_input_mode and self.num_input_images > 1:
            intermediate_heatmaps = torch.reshape(x, [-1, self.num_joints*self.num_input_images] + list(x.shape[2:]))
            x = self.merge_heatmaps(intermediate_heatmaps)
            if self.output_activation is not None and self.output_activation.lower() == 'sigmoid':
                x = torch.sigmoid(x)
        else:
            intermediate_heatmaps = None

        if return_inter_hm:
            return x, intermediate_heatmaps
        else:
            return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            if False:#torch.cuda.is_available():
                pretrained_state_dict = torch.load(pretrained)
            else:
                pretrained_state_dict = torch.load(pretrained, map_location = torch.device('cpu'))

            logger.info('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                    # nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)


resnet_spec = {
    50: (LW_Bottleneck, [3, 4, 6, 3]),
    101: (LW_Bottleneck, [3, 4, 23, 3]),
    152: (LW_Bottleneck, [3, 8, 36, 3])
}


def get_pose_net(cfg, is_train, **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS

    block_class, layers = resnet_spec[num_layers]

    model = LPN(block_class, layers, cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model
