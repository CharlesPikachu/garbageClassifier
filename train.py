'''
Function:
	train the model.
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import os
import torch
import config
import torch.nn as nn
from utils.utils import *
from nets.NetsTorch import NetsTorch
from utils.datasets import ImageFolder


'''train'''
def train(config):
	# prepare
	if not os.path.exists(config.save_dir):
		os.mkdir(config.save_dir)
	use_cuda = torch.cuda.is_available()
	# define the model
	model = NetsTorch(net_name=config.net_name, pretrained=config.load_pretrained, num_classes=config.num_classes)
	if use_cuda:
		os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
		if config.ngpus > 1:
			model = nn.DataParallel(model).cuda()
		else:
			model = model.cuda()
	model.train()
	# dataset
	dataset_train = ImageFolder(data_dir=config.traindata_dir, image_size=config.image_size, is_train=True)
	dataset_test = ImageFolder(data_dir=config.testdata_dir, image_size=config.image_size, is_train=False)
	dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
	dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
	Logging('Train dataset size: %d...' % len(dataset_train), config.logfile)
	Logging('Test dataset size: %d...' % len(dataset_test), config.logfile)
	# optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
	criterion = nn.CrossEntropyLoss()
	# train
	FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
	for epoch in range(1, config.num_epochs+1):
		Logging('[INFO]: epoch now is %d...' % epoch, config.logfile)
		for batch_i, (imgs, labels) in enumerate(dataloader_train):
			imgs = imgs.type(FloatTensor)
			labels = labels.type(FloatTensor)
			optimizer.zero_grad()
			preds = model(imgs)
			loss = criterion(preds, labels.long())
			if config.ngpus > 1:
				loss = loss.mean()
			Logging('[INFO]: batch%d of epoch%d, loss is %.2f...' % (batch_i, epoch, loss.item()), config.logfile)
			loss.backward()
			optimizer.step()
		if ((epoch % config.save_interval == 0) and (epoch > 0)) or (epoch == config.num_epochs):
			pklpath = os.path.join(config.save_dir, 'epoch_%s.pkl' % str(epoch))
			if config.ngpus > 1:
				cur_model = model.module
			else:
				cur_model = model
			torch.save(cur_model.state_dict(), pklpath)
			acc = test(model, dataloader_test)
			Logging('[INFO]: Accuracy of epoch %d is %.2f...' % (epoch, acc), config.logfile)


'''test'''
def test(model, dataloader):
	model.eval()
	use_cuda = torch.cuda.is_available()
	FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
	n_correct = 0
	n_total = 0
	for batch_i, (imgs, labels) in enumerate(dataloader):
		imgs = imgs.type(FloatTensor)
		labels = labels.type(FloatTensor)
		preds = model(imgs)
		n_correct += (preds.max(-1)[1].long() == labels.long()).sum().item()
		n_total += imgs.size(0)
	acc = n_correct / n_total
	model.train()
	return acc


'''run'''
if __name__ == '__main__':
	train(config)