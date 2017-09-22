from numpy import *
import re

class Bayes:
	def loaddata(self):
		postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
				['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
				['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
				['stop', 'posting', 'stupid', 'worthless', 'garbage'],
				['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
				['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
		classVec = [0,1,0,1,0,1]
		return postingList,classVec

	def createlablelist(self,dataset):
		labelset = set([])
		for item in dataset:
			labelset = labelset | set(item)
		return list(labelset)


	def word2vecset(self,labelset,inputset):   # 词汇表变成向量,词集模型,每个词只有一次,基于伯努利方法
		m = len(labelset)
		retvec = [0]*m
		for word in inputset:
			if word in labelset:
				retvec[labelset.index(word)] = 1
			else:
				print(word)
		return retvec

	def word2vecbag(self,labelset,inputset):  # 词汇表变成向量,词袋模型,每个词可以多次,基于多项式实现
		m = len(labelset)
		retvec = [0]*myword
		for word in inputset:
			if word in labelset:
				retvec[labelset.index(word)] += 1
			else:
				print(word)
		return retvec

	def nb0(self,trainmatrix,trainlabel):   #naive bayes 二分类
		m = len(trainmatrix)
		n = len(trainmatrix[0])
		p1 = sum(trainlabel)/float(m)   #  计算为1的概率P(c=1)
		p0nums = ones(n)   # 拉普拉斯平滑
		p1nums = ones(n)
		p0sum = 2.0
		p1sum = 2.0
		for i in range(m):
			if trainlabel[i] == 1:
				p1nums += trainmatrix[i]
				p1sum += sum(trainmatrix[i])
			else:
				p0nums += trainmatrix[i]
				p0sum += sum(trainmatrix[i])
		p1vec = log(p1nums/p1sum)    # 取对数防止下溢,即多个乘积相乘四舍五入为零的情况
		p0vec = log(p0nums/p0sum)
		return p0vec,p1vec,p1

	def classfynb(self,vec2class,p0vec,p1vec,pc1):
		p1 = sum(vec2class*p1vec) + log(pc1)
		p0 = sum(vec2class*p0vec) + log(1.0-pc1)
		if p1 > p0:
			return 1
		else:
			return 0

	def textparse(self,bigstring):   # 把一长串string变成wordlist
		listoftokens = re.split(r'\W*',bigstring)  # 除了word之外字符,标点之类都删除
		return [tok.lower() for tok in listoftokens if len(tok)>2]   # 返回长度大于2的字符,变成小写


	def testtext(self):   # 邮件分类验证
		doclist = []
		classlist = []
		fulltext = []
		for i in range(1,24):
			wordlist = Bayes().textparse(open('email/spam/%d.txt' % i).read())   # 有数字!!
			doclist.append(wordlist)
			fulltext.extend(wordlist)
			classlist.append(1)
			wordlist = Bayes().textparse(open('email/ham/%d.txt' % i).read())
			doclist.append(wordlist)
			fulltext.extend(wordlist)
			classlist.append(0)
		vocalist = self.createlablelist(doclist)
		trainingset = list(range(46))
		testset = []
		for i in range(30):
			randomindex = int(random.uniform(0,len(trainingset)))
			testset.append(trainingset[randomindex])
			del(trainingset[randomindex])
		trainmat = []
		trainclass = []
		for i in trainingset:
			trainmat.append(self.word2vecset(vocalist,doclist[i]))
			trainclass.append(classlist[i])
		p0v,p1v,pc1 = self.nb0(array(trainmat),array(trainclass))
		error = 0
		for i in testset:
			wordvec = self.word2vecset(vocalist,doclist[i])
			p = self.classfynb(wordvec,p0v,p1v,pc1)
			if p != classlist[i]:
				error += 1
		errorrate = float(error)/len(testset)
		print(errorrate)

			



# listpost,listclass = Bayes().loaddata()
# myword = Bayes().createlablelist(listpost)
# trainmat = []
# for item in listpost:
# 	trainmat.append(Bayes().word2vecset(myword,item))
# p0v,p1v,pc1 = Bayes().nb0(trainmat,listclass)
# test1 = ['love','stupid','garbage']
# doc = array(Bayes().word2vecset(myword,test1))
# a = Bayes().classfynb(doc,p0v,p1v,pc1)


Bayes().testtext()
