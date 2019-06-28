'''
Function:
	Visualization the test results while training
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import re
import matplotlib.pyplot as plt


f = open('train.log', 'r')
contents = f.read()
results = re.findall(r'loss is (.*?)\.\.\.', contents)
losses = []
for result in results:
	losses.append(float(result))
plt.title('The curve of loss')
plt.xlabel('iter')
plt.ylabel('loss')
plt.plot(list(range(len(losses))), losses, 'b')
plt.savefig('losscurve.jpg')
plt.show()