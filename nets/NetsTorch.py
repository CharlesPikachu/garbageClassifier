'''
Function:
	nets defined in the torchvision module, support alexnet, resnet and vgg.
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import torch
import torchvision
import torch.nn as nn


'''nets defined in the torchvision module.'''
class NetsTorch(nn.Module):
	def __init__(self, net_name, pretrained, num_classes, **kwargs):
		super(NetsTorch, self).__init__()
		net_name = net_name.lower()
		if net_name == 'alexnet':
			self.net_used = torchvision.models.alexnet(pretrained=pretrained)
		elif net_name == 'vgg11':
			self.net_used = torchvision.models.vgg11(pretrained=pretrained)
		elif net_name == 'vgg11_bn':
			self.net_used = torchvision.models.vgg11_bn(pretrained=pretrained)
		elif net_name == 'vgg13':
			self.net_used = torchvision.models.vgg13(pretrained=pretrained)
		elif net_name == 'vgg13_bn':
			self.net_used = torchvision.models.vgg13_bn(pretrained=pretrained)
		elif net_name == 'vgg16':
			self.net_used = torchvision.models.vgg16(pretrained=pretrained)
		elif net_name == 'vgg16_bn':
			self.net_used = torchvision.models.vgg16_bn(pretrained=pretrained)
		elif net_name == 'vgg19':
			self.net_used = torchvision.models.vgg19(pretrained=pretrained)
		elif net_name == 'vgg19_bn':
			self.net_used = torchvision.models.vgg19_bn(pretrained=pretrained)
		elif net_name == 'resnet18':
			self.net_used = torchvision.models.resnet18(pretrained=pretrained)
		elif net_name == 'resnet34':
			self.net_used = torchvision.models.resnet34(pretrained=pretrained)
		elif net_name == 'resnet50':
			self.net_used = torchvision.models.resnet50(pretrained=pretrained)
		elif net_name == 'resnet101':
			self.net_used = torchvision.models.resnet101(pretrained=pretrained)
		elif net_name == 'resnet152':
			self.net_used = torchvision.models.resnet152(pretrained=pretrained)
		elif net_name == 'inception_v3':
			self.net_used = torchvision.models.inception_v3(pretrained=pretrained)
		else:
			raise ValueError('Unsupport NetsTorch.net_name <%s>...' % net_name)
		if net_name in ['alexnet', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']:
			self.net_used.classifier[-1] = nn.Linear(in_features=4096, out_features=num_classes)
		elif net_name in ['inception_v3', 'resnet50', 'resnet101', 'resnet152']:
			self.net_used.fc = nn.Linear(in_features=2048, out_features=num_classes)
		elif net_name in ['resnet18', 'resnet34']:
			self.net_used.fc = nn.Linear(in_features=512, out_features=num_classes)
	def forward(self, x):
		x = self.net_used(x)
		return x