from load_data import *
from collections import defaultdict
import json
import numpy 
import matplotlib.pyplot as plt
import os
from math import exp
from math import log

def f(theta, X, y, lam):
  theta = numpy.matrix(theta).T
  X = numpy.matrix(X)
  y = numpy.matrix(y).T
  diff = X*theta - y
  diffSq = diff.T*diff
  diffSqReg = diffSq / len(X) + lam*(theta.T*theta)
  #print "offset =", diffSqReg.flatten().tolist()
  return diffSqReg.flatten().tolist()[0]

def fprime(theta, X, y, lam):
  theta = numpy.matrix(theta).T
  X = numpy.matrix(X)
  y = numpy.matrix(y).T
  diff = X*theta - y
  res = 2*X.T*diff / len(X) + 2*lam*theta
  #print "gradient =", numpy.array(res.flatten().tolist()[0])
  return numpy.array(res.flatten().tolist()[0])

def inner(X,theta):
	res=0.0
	for i in range(len(X)):
		res+=X[i]*theta[i]
	return res

def ComputeMSE(X,y,theta):
	N=len(X)
	MSE=0.0
	for i in range(N):
		MSE = MSE+(y[i]-inner(X[i],theta))*(y[i]-inner(X[i],theta))
	return MSE/float(N)

data=load_data()
unixtime=[]
upvotes=[]
downvotes=[]
comments=[]
score=[]
features=[]
votes=[]
user=[]
image=[]
isfunny=[]
isgif=[]
localtime=[]
title=[]
length=len(data)
time1=[]
v1=[]
time10005=[]
v2=[]
time1001=[]
v3=[]
subreddit=defaultdict(int)
subcount=defaultdict(int)
community=[]

userdic=defaultdict(int)
usercoutn=defaultdict(int)
for record in data:
	unixtime.append(float(record.get('unixtime')))
	upvotes.append(float(record.get('number_of_upvotes')))
	downvotes.append(float(record.get('number_of_downvotes')))
	comments.append(float(record.get('number_of_comments')))
	score.append(float(record.get('score')))
	votes.append(float(record.get('total_votes')))
	user.append(record.get('username'))
	image.append(record.get('image_id'))
	community.append(record.get('subreddit'))
	userdic[record.get('username')]+=float(record.get('number_of_comments'))
	usercoutn[record.get('username')]+=1

	if record.get('subreddit')=='funny':
		isfunny.append(1)
	else:
		isfunny.append(0)
	if record.get('subreddit')=='gif':
		isgif.append(1)
	else:
		isgif.append(0)
	t=record.get('rawtime')
	te=t.split('-')
	localtime.append(int(te[1]))
	subreddit[record.get('subreddit')]+=float(record.get('number_of_comments'))
	subcount[record.get('subreddit')]+=1
#print max(comments)
#print max(comments)/len(comments)
Mintime=min(unixtime)

usertemp=[]
for use in userdic:
	usertemp.append((usercoutn[use],userdic[use]/usercoutn[use],use))
usertemp.sort()
usertemp.reverse()
#print usertemp[:10]
temp=[]
for sub in subreddit:
	temp.append((subreddit[sub]/subcount[sub],sub))

temp.sort()
temp.reverse()
phase=[]
t1=[]
t2=[]
t3=[]
t4=[]
t5=[]
t6=[]
t7=[]
t8=[]
t9=[]
t10=[]
t11=[]
for (ave,sub) in temp:
	if ave>2000:
		t1.append(sub)
	elif ave>900:
		t2.append(sub)
	elif ave>700:
		t3.append(sub)
	elif ave>400:
		t4.append(sub)
	elif ave>300:
		t5.append(sub)
	elif ave>180:
		t6.append(sub)
	elif ave>140:
		t7.append(sub)
	elif ave>100:
		t8.append(sub)
	elif ave>50:
		t9.append(sub)
	elif ave>20:
		t10.append(sub)
	else:
		t11.append(sub)

phase.append(t1)
phase.append(t2)
phase.append(t3)
phase.append(t4)
phase.append(t5)
phase.append(t6)
phase.append(t7)
phase.append(t8)
phase.append(t9)
phase.append(t10)
phase.append(t11)

features=[]
for i in range(length):
	temp=[]
	for j in range(11):
		temp.append(0)
	for j in range(11):
		if community[i] in phase[j]:
			temp[j]=1
	features.append(temp)

phase=[]
t1=[]
t2=[]
t3=[]
t4=[]
t5=[]
t6=[]
t7=[]
t8=[]
t9=[]
t10=[]
t11=[]

for (count,ave,use) in usertemp:
	if ave>1000:
		t1.append(use)
	elif ave>500:
		t2.append(use)
	elif ave>300:
		t3.append(use)
	elif ave>200:
		t4.append(use)
	elif ave>100:
		t5.append(use)
	elif ave>70:
		t6.append(use)
	elif ave>35:
		t7.append(use)
	elif ave>20:
		t8.append(use)
	elif ave>10:
		t9.append(use)
	elif ave>5:
		t10.append(use)
	else:
		t11.append(use)

phase.append(t1)
phase.append(t2)
phase.append(t3)
phase.append(t4)
phase.append(t5)
phase.append(t6)
phase.append(t7)
phase.append(t8)
phase.append(t9)
phase.append(t10)
phase.append(t11)

for i in range(length):
	temp=[]
	for j in range(11):
		temp.append(0)
	for j in range(11):
		if user[i] in phase[j]:
			temp[j]=1
	features[i]+=temp

#plt.plot(comments,score,color='r',label='score')
#plt.plot(comments,votes,color='b',label='votes')

#plt.show()
#time1.sort()
#time10005.sort()
#print time1
#print time10005

#f=open('title/kmean_cluster_10_feature_vector.txt')
pre=localtime[0]
index=1

#f=open('title/svd_tfidf_100.txt')

index=1
'''
for line in f.readlines():
	topic=line.split()
	temp=[1]
	for t in topic:
		temp.append(float(t))
	features.append(temp)
'''
'''
with open('title/F_20_1.json') as f:
	topic=json.load(f)
for t in topic:
	temp=[1]
	#for o in t:
	#	temp.append(float(o))
	features.append(temp)
with open('title/F_20_2.json') as f:
	topic=json.load(f)
for t in topic:
	temp=[1]
	#for o in t:
	#	temp.append(float(o))
	features.append(temp)
with open('title/F_20_3.json') as f:
	topic=json.load(f)
for t in topic:
	temp=[1]
	#for o in t:
	#	temp.append(float(o))
	features.append(temp)
with open('title/F_20_4.json') as f:
	topic=json.load(f)
for t in topic:
	temp=[1]
	#for o in t:
	#	temp.append(float(o))
	features.append(temp)
with open('title/F_20_5.json') as f:
	topic=json.load(f)
for t in topic:
	temp=[1]
	#for o in t:
	#	temp.append(float(o))
	features.append(temp)
'''
with open ('user_feature/user_vec_number_of_comments.json') as f:
	user_comments_dic=json.load(f)

with open ('user_feature/user_vec_total_votes.json') as f:
	user_votes_dic=json.load(f)

with open ('image_feature/image_vec_number_of_comments.json') as f:
	image_comments_dic=json.load(f)

with open ('image_feature/image_vec_total_votes.json') as f:
	image_votes_dic=json.load(f)

with open ('title/F_new.json') as f:
	new=json.load(f)
pretime=unixtime[0]
preid=image[0]

for i in range(length):
	u=user[i]
	im=image[i]
	e=[0,0]
	if image[i]==preid and comments[i]<20:
		features[i].append(-log((float(pretime))/unixtime[i]))
	else:
		preid=image[i]
		pretime=unixtime[i]
		features[i].append(-log((float(pretime)/unixtime[i])))
	features[i]+=new[i]
	features[i].append(1)


	#if user_comments_dic.has_key(u):
	#	temp=user_comments_dic[u]
	#	features[i].append(float(temp[0]))
	#	features[i].append(float(temp[1]))
	#else:
	#	features[i].append(0)
	#	features[i].append(0)
	#if image_votes_dic.has_key(im):
	#	temp=image_votes_dic[im]
	#	features[i].append(float(temp[0]))
	#	features[i].append(float(temp[1]))
	#else:
	#	features[i].append(0)
	
	#features[i].append(isfunny[i])
	#features[i].append(isgif[i])
	#if localtime[i]==pre:
	#	features[i].append(1.0/(index**2))
	#	index+=1
	#else:
	#	pre=localtime[i]
	#	index=1
	#	features[i].append(1.0/(index**2))
	#if localtime[i]==pre and index<7:
	#	features[i].append((7-index))
	#	index+=1
	#elif localtime[i]==pre and index>=7:
	#	features[i].append(0)
	#	index+=1
	#else:
	#	pre=localtime[i]
	#	index=0
	#	features[i].append((7-index))

	#features[i].append(unixtime[i]-Mintime)
	#features[i].append(float(comments[i]))




#theta,residuals,rank,s = numpy.linalg.lstsq(features[:length/3], comments[:length/3])
#y=[]
#for i in range(length/3:2*length/3):
#	y.append(inner(features[i]))
#Mse=ComputeMSE(features[:length/3],comments[:length/3],theta)
#print Mse
#Mse=ComputeMSE(features[length/3:2*length/3],comments[length/3:2*length/3],theta)
#print Mse
#Mse=ComputeMSE(features[2*length/3:length],comments[2*length/3:length],theta)
#print Mse
#print features[:10]
#print votes[:10]

#print inner(features[0],theta)

#y=[inner(features[i],theta) for i in range(length/3)]
#print y[:10]
#print votes[:10]

from sklearn.neural_network import MLPRegressor
MLP_model = MLPRegressor(hidden_layer_sizes=(60,40,20),activation='tanh')
trained_MLP = MLP_model.fit(features[:length/3], comments[:length/3])

# test set evaluation
predicted = trained_MLP.predict(features[length/3:2*length/3])
print predicted[:10]

MSE = 0.0
for i in range(len(predicted)):
    MSE += (predicted[i] - comments[i+length/3])**2

MSE = MSE/len(predicted)

print(MSE)


# test set evaluation
predicted = trained_MLP.predict(features[:length/3])

MSE = 0.0
for i in range(len(predicted)):
    MSE += (predicted[i] - comments[i])**2

MSE = MSE/len(predicted)

print(MSE)


predicted = trained_MLP.predict(features[2*length/3:length])


MSE = 0.0
for i in range(len(predicted)):
    MSE += (predicted[i] - comments[i+2*length/3])**2

MSE = MSE/len(predicted)

print(MSE)

#Mse=ComputeMSE(features[length/3:2*length/3],comments[length/3:2*length/3],theta)



'''
user_comments_dic=open('user_feature/user_vec_number_of_comments.json')
user_votes_dic=json.load('user_feature/user_vec_total_votes.json')

image_comments_dic=json.load('image_feature/image_vec_number_of_comments.json')
image_votes_dic=json.load('image_feature/image_vec_total_votes.json')


'''
