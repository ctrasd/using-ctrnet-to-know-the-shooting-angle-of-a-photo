#ctrnet_rjs角度.py
#按角度对数据集进行分类训练并预测照片入镜角度
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from torch import optim
import torch.nn as nn
import torch as t
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from tensorboardX import SummaryWriter
import cv2



def cv_imread(file_path):
	cv_img=cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
	return cv_img


class Net(nn.Module):
	"""docstring for Net"""
	def __init__(self):
		super(Net, self).__init__()
		self.conv1=nn.Conv2d(3,32,2, padding=1)
		self.conv2=nn.Conv2d(32,32,3,padding=1)
		self.conv3=nn.Conv2d(32,64,3,padding=1)
		self.conv4=nn.Conv2d(64,64,3,padding=1)
		# 全连接层 
		# 1280*980
		self.fc1=nn.Linear(64*60*80,512)
		self.fc2=nn.Linear(512,19)
		self.dropout = nn.Dropout(p=0.5)

	def forward(self,x):
		x=F.relu(self.conv1(x))
		
		x=F.max_pool2d(self.conv2(x),(2,2))
		#print('size:')
		#print(x.size())
		x=self.dropout(x)
		x=self.conv3(x)
		x=F.max_pool2d(self.conv4(x),(2,2))
		#print('size:')
		#print(x.size())
		x=self.dropout(x)
		x=x.view(x.size()[0],-1)
		x=F.relu(self.fc1(x))
		x=self.fc2(x)
		return x

if __name__ == '__main__':
	net=Net()
	net.load_state_dict(t.load('model_dict.pkl'))
	print(type(net))
	print(net)
	writer=SummaryWriter(comment='ctrnet_net')
# 定义对数据的预处理
	transform = transforms.Compose([
			transforms.ToTensor(), # 转为Tensor
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 归一化
                             ])

# 训练集
	trainset = tv.datasets.ImageFolder(
						root='D:/机器学习/角度识别/train_resize_cut', 
						transform=transform)

	trainloader = t.utils.data.DataLoader(
						trainset, 
						batch_size=19,
						shuffle=True, 
						num_workers=0)
# 测试集
	testset = tv.datasets.ImageFolder(
						root='D:/机器学习/角度识别/test_resize_cut',
						transform=transform)

	testloader = t.utils.data.DataLoader(
						testset,
						batch_size=6, 
						shuffle=False,
						num_workers=0)

	def imshow(img):
		img = img / 2 + 0.5
		npimg = img.numpy()
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
	
	(data, label) = trainset[4]
	print(data.size())
	print(label)
	print(type(torchvision.utils.make_grid(data)))
	imshow(torchvision.utils.make_grid(data))
	plt.show()
	'''
	dataiter = iter(trainloader)
	images, labels = dataiter.next()
	print(images.size())
	print(' '.join('%11s'%labels[j] for j in range(5)))
	imshow(torchvision.utils.make_grid(images))  # 拼接成一张
	plt.show()#关掉图片才能往后继续算
	'''

	criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
	optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.02)
	idx=0
	t.set_num_threads(8)
	#input_data = Variable(t.rand(20, 3, 320, 240))
	#writer.add_graph(net,input_data,verbose=False)
	print('done')
	for epoch in range(11):  
		correct = 0 # 预测正确的图片数
		total = 0 # 总共的图片数
		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
        # 输入数据
			inputs, labels = data
			print(inputs.size())
        # 梯度清零
			optimizer.zero_grad()
        
        # forward + backward 
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			running_loss += loss.item()
			loss.backward()   
			_, predicted = t.max(outputs, 1)
			correct += (predicted == labels).sum()
			total+=labels.size(0)
        # 更新参数 
			optimizer.step()
        
			# 打印log信息	
        # loss 是一个scalar,需要使用loss.item()来获取数值，不能使用loss[0]
			
			if i % 2 == 1: # 每2个batch打印一下训练状态
				print('[%d, %5d] loss: %.3f' \
					 % (epoch+1, i+1, running_loss ))
				idx=idx+1
				writer.add_scalar('scalar/running_loss',running_loss, idx)
				running_loss = 0.0
				#t.save(net.state_dict(), 'model_dict.pkl')
				#t.save(net,'model.pkl')
				print('准确率为： %.2f %%'%(100*correct/total))
				writer.add_scalar('scalar/running_correct',(100.0*correct/total), idx)
				correct = 0 
				total = 0 

		if epoch%3==0:
			t.save(net.state_dict(), 'model_dict.pkl')
			print(epoch,' saved')

	print('Finished Training')
	dataiter = iter(testloader)
	images, labels = dataiter.next() # 一个batch返回40张图片
	print(type(images))
	print(images.size())
	print('实际的label: ', ' '.join(\
			'%08s'%labels[j] for j in range(5)))
	imshow(tv.utils.make_grid(images))
	plt.show()

	# 计算图片在每个类别上的分数
	outputs = net(images)
	# 得分最高的那个类
	_, predicted = t.max(outputs.data, 1)

	print('预测结果: ', ' '.join('%5s'\
		% predicted[j] for j in range(5)))

	correct = 0 # 预测正确的图片数
	total = 0 # 总共的图片数


# 由于测试的时候不需要求导，可以暂时关闭autograd，提高速度，节约内存
	with t.no_grad():
		for data in testloader:
			images, labels = data
			outputs = net(images)
			_, predicted = t.max(outputs, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum()

	print('30张测试集中的准确率为: %.2f %%' % (100 * correct / total))
	writer.close()