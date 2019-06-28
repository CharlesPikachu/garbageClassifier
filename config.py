'''
Function:
	config file.
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import os


'''train and test'''
net_name = 'resnet50' # support 'alexnet', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'inception_v3'
num_classes = 6 # num of classes
traindata_dir = os.path.join(os.getcwd(), 'GarbageData/train') # the images dir for train
testdata_dir = os.path.join(os.getcwd(), 'GarbageData/test') # the images dir for test
learning_rate = 2e-4 # learning rate for adam
load_pretrained = True # whether load the pretrained weights from https://download.pytorch.org/models/
num_epochs = 200 # number of epochs while training
batch_size = 32 # batch_size
image_size = (224, 224) # image size for feeding network.
save_interval = 10 # execute the operator of saving model weights every save_interval epochs
save_dir = 'model_saved' # dir for save model
logfile = 'train.log' # file to record train info
gpus = '0,1' # gpu ids used
ngpus = 2 # number of gpu used
num_workers = 4 # number of worker used
clsnamespath = 'classes.data' # save the class names


'''for demo'''
camera_id = 0 # camera id used
weightspath = 'model_saved/epoch_200.pkl' # model weight used
conf_thresh = 0.5 # conf thresh