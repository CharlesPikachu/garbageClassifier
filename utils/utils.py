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


'''save classes'''
def saveClasses(classes, filename='classes.data'):
	with open(filename, 'w') as f:
		for c in classes:
			f.write(c + '\n')
	return True


'''load classes'''
def loadClasses(filename='classes.data'):
	classes = []
	with open(filename, 'r') as f:
		for line in f.readlines():
			if line.strip('\n'):
				classes.append(line.strip('\n'))
	return classes