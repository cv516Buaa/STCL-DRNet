import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
import torchvision.transforms.functional as trf


class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier, bn_freeze):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.bn_freeze = bn_freeze

    def forward(self, x, labels=None, features_out=False):
        input_shape = x.shape[-2:]
        features = self.backbone(x, labels)
        if features_out is True:
            out1, out2, out2_pos, out2_neg, aspp_feature = self.classifier(features, features_out)
            out1 = F.interpolate(out1, size=input_shape, mode='bilinear', align_corners=False)
            out2 = F.interpolate(out2, size=input_shape, mode='bilinear', align_corners=False)
            return out1, out2, out2_pos, out2_neg, aspp_feature
        else:
            out1, out2, out2_pos, out2_neg = self.classifier(features, features_out)
            out1 = F.interpolate(out1, size=input_shape, mode='bilinear', align_corners=False)
            out2 = F.interpolate(out2, size=input_shape, mode='bilinear', align_corners=False)
            return out1, out2, out2_pos, out2_neg


    def train(self, mode=True):
        super(_SimpleSegmentationModel, self).train(mode=mode)

        # if 'meta_layers' in self.backbone._modules.keys():
        #     for m in self.backbone.meta_layers.modules():
        #         if isinstance(m, nn.BatchNorm2d):
        #             m.eval()
        #             m.weight.requires_grad = False
        #             m.bias.requires_grad = False

        if self.bn_freeze:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """

    def __init__(self, model, return_layers, num_classes, with_meta=False):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

        self.num_classes = num_classes
        self.tot_classes = sum(num_classes)
        self.with_meta = with_meta
        if with_meta:
            print('using meta block')
            self.meta_layer = nn.ModuleList([nn.Sequential(nn.Conv2d(2048, 2048, 3, stride=1, padding=1, bias=False),
                                                           nn.BatchNorm2d(2048),
                                                           nn.ReLU()) for _ in range(self.tot_classes - 2)])
            self.meta_bn = nn.ModuleList([nn.BatchNorm2d(512) for _ in range(self.tot_classes - 2)])

    def forward(self, x, labels):
        out = OrderedDict()
        for name, module in self.named_children():
            if name == 'meta_layer' or name == 'meta_bn':
                continue
            x = module(x)
            if name == 'layer4' and self.with_meta:
                x = self._meta_forward(x, labels)

            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out

    def _meta_forward(self, outs, labels):
        attention = 0

        if labels is not None:
            labels = labels.detach().clone()
            labels = trf.resize(labels, outs.shape[-2:], 0)
            for i, mod in enumerate(self.meta_layer):
                # 含有 0 unknown 1 bg
                labels_mask = (labels == i + 2).type(torch.float)
                labels_mask = labels_mask.unsqueeze(dim=1)
                masked_feature = torch.mul(outs, labels_mask)
                # norm_feature = self.meta_bn[i](masked_feature)
                meta_feature = torch.sigmoid(mod(masked_feature))
                attention = attention + meta_feature

        else:
            for i, mod in enumerate(self.meta_layer):
                # norm_feature = self.meta_bn[i](outs)
                meta_feature = torch.sigmoid(mod(outs))
                # meta_feature = (meta_feature - meta_feature.mean(dim=[-1, -2], keepdim=True)) / \
                #                (meta_feature.var(dim=[-1, -2], keepdim=True) + 1e-5) ** 0.5
                attention = attention + meta_feature

        attention = attention/len(self.meta_layer)
        outs_meta = attention * outs + outs
        return outs_meta
