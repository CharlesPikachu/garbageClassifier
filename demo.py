'''
Function:
	demo.
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import torch
import config
import argparse
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from utils.utils import *
from nets.NetsTorch import NetsTorch


'''classifier'''
def classifier(config, image_path):
	# prepare
	use_cuda = torch.cuda.is_available()
	FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
	classes = loadClasses(config.clsnamespath)
	# model
	model = NetsTorch(net_name=config.net_name, pretrained=False, num_classes=config.num_classes)
	model.load_state_dict(torch.load(config.weightspath))
	if use_cuda:
		model = model.cuda()
	model.eval()
	# transform
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
	# run
	img = Image.open(image_path)
	img_input = transform(img)
	img_input = img_input.type(FloatTensor).unsqueeze(0)
	with torch.no_grad():
		preds = model(img_input)
	preds = nn.Softmax(-1)(preds).cpu()
	max_prob, max_prob_id = preds.view(-1).max(0)
	max_prob = max_prob.item()
	max_prob_id = max_prob_id.item()
	clsname = classes[max_prob_id]
	if max_prob > config.conf_thresh:
		print('[Garbage]: %s, [Conf]: %s.' % (clsname, max_prob))
	else:
		print('No Garbage!!!')


'''run'''
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Garbage classifier.")
	parser.add_argument('-i', dest='image', help='Image to be classified.')
	args = parser.parse_args()
	if args.image:
		classifier(config, args.image)