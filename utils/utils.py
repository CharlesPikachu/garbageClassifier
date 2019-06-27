'''
Function:
	some utils
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import time


'''print function'''
def Logging(message, savefile=None):
	content = '%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message)
	if savefile:
		f = open(savefile, 'a')
		f.write(content + '\n')
		f.close()
	print(content)