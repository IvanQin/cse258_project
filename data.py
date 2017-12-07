from load_data import *
from collections import defaultdict
import json
import numpy 
import matplotlib.pyplot as plt
import os
from math import log

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
length=len(data)
title=[]
time1=[]
v1=[]
time10005=[]
v2=[]
time1001=[]
v3=[]
community=defaultdict(list)
index=0
for record in data:

	unixtime.append(float(record.get('unixtime')))
	upvotes.append(float(record.get('number_of_upvotes')))
	downvotes.append(float(record.get('number_of_downvotes')))
	comments.append(float(record.get('number_of_comments')))
	score.append(float(record.get('score')))
	votes.append(float(record.get('total_votes')))
	user.append(record.get('username'))
	image.append(record.get('image_id'))
	title.append(record.get('title'))

	community[record.get('subreddit')].append(index)

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
	temp=[]
	temp.append(int(te[0]))
	temp.append(int(te[1]))
	day=te[2].split('T')

	localtime.append([int(te[0]),int(te[1]),int(day[0])])

year=2012

day=3 #5,6,13

test=[]
userdetect=[]
timedetect=[]
noone=[]
imagedetect=defaultdict(int)
length=len(data)
timedic=defaultdict(int)
yeartime1=defaultdict(int)
yeartime2=defaultdict(int)
yeartime3=defaultdict(int)

for i in range(length):
	test.append((upvotes[i],title[i],user[i],localtime[i],isfunny[i],isgif[i]))
	userdetect.append((upvotes[i],user[i]))
	if localtime[i][0]==2010:
		yeartime1[localtime[i][1]]+=upvotes[i]
	if localtime[i][0]==2011:
		yeartime2[localtime[i][1]]+=upvotes[i]
	if localtime[i][0]==2012:
		yeartime3[localtime[i][1]]+=upvotes[i]
	if user[i]=='':
		noone.append(upvotes[i])
	timedic[(localtime[i][0],localtime[i][1],localtime[i][2])]+=1
	imagedetect[image[i]]+=1
	timedetect.append((unixtime[i],upvotes[i]))
timedetect.sort()
x=[]
y=[]
for (ti,up) in timedetect:
	x.append(ti)
	y.append(up)

plt.plot(x,y)
p1.set_xlabel('time')
p1.set_ylabel('upvotes')
p1.show()

'''
bar_width = 0.5
temp=[]
for t in yeartime1:
	temp.append((t,yeartime1[t]))
temp.sort()
index = numpy.arange(12)
up=[]
for (_,u) in temp:
	up.append(u)
p1=plt.subplot(311)
p1.bar(index,up,bar_width,color='b',label="2010")

p1.set_ylabel('upvotes')
plt.xticks(index + bar_width / 2, ('1', '2', '3', '4', '5','6','7','8','9','10','11','12'))
temp=[]
for t in yeartime2:
	temp.append((t,yeartime2[t]))
temp.sort()
index = numpy.arange(12)
up=[]
for (_,u) in temp:
	up.append(u)
p1=plt.subplot(312)
p1.bar(index,up,bar_width,color='b',label="2011")


plt.xticks(index + bar_width / 2, ('1', '2', '3', '4', '5','6','7','8','9','10','11','12'))
temp=[]
for t in yeartime3:
	temp.append((t,yeartime3[t]))
temp.sort()
index = numpy.arange(12)
up=[]
for (_,u) in temp:
	up.append(u)
p1=plt.subplot(313)
p1.bar(index,up,bar_width,color='b',label="2012")
p1.set_xlabel('month')

plt.xticks(index + bar_width / 2, ('1', '2', '3', '4', '5','6','7','8','9','10','11','12'))


plt.show()
'''
'''
temp=[]
for t in timedic:
	temp.append((timedic[t],t))
temp.sort()
temp.reverse()

print temp[:5]
timedetect.sort()
userdetect.sort()
userdetect.reverse()

time1.sort()
time2.sort()
time3.sort()
time4.sort()
time5.sort()

x=[]
y=[]
for (un,up) in time1:
	x.append(un)
	y.append(up)

usercount=defaultdict(list)
for (up,us) in userdetect:
	if up>=1000:
		usercount[us].append(up)
#print usercount

#print len(timedetect)
x=[]
y=[]
for (t,u) in time1:
	x.append(t)
	y.append(u)
p1=plt.subplot(511)
p1.plot(x,y,color='b')
p1.set_xlabel('unixtime')
p1.set_ylabel('upvotes')

x=[]
y=[]
for (t,u) in time2:
	x.append(t)
	y.append(u)
p1=plt.subplot(512)
p1.plot(x,y,color='b')
p1.set_xlabel('unixtime')
p1.set_ylabel('upvotes')

x=[]
y=[]
for (t,u) in time3:
	x.append(t)
	y.append(u)
p1=plt.subplot(513)
p1.plot(x,y,color='b')
p1.set_xlabel('unixtime')
p1.set_ylabel('upvotes')

x=[]
y=[]
for (t,u) in time4:
	x.append(t)
	y.append(u)
p1=plt.subplot(514)
p1.plot(x,y,color='b')
p1.set_xlabel('unixtime')
p1.set_ylabel('upvotes')


x=[]
y=[]
for (t,u) in time5:
	x.append(t)
	y.append(u)
p1=plt.subplot(515)
p1.plot(x,y,color='b')
p1.set_xlabel('unixtime')
p1.set_ylabel('upvotes')

plt.show()
'''
'''
iimage=[]
for i in imagedetect:
	iimage.append((imagedetect[i],i))
iimage.sort()
iimage.reverse()
'''
'''
#print iimage[:5]
im=['6037','5919','174','6219','996']
tempx=[]
tempy=[]
tempz=[]
temp=[]
for j in range(length):
	if image[j]=='6037':
		temp.append((unixtime[j],comments[j],upvotes[j]))
temp.sort()
for (time,co,up) in temp:
	tempx.append(time)
	tempy.append(co)
	tempz.append(up)
p1=plt.subplot(511)
#p1.plot(tempx,tempy,label='comments',color='r')
p1.plot(tempx,tempz,label='upvotes',color='b')
p1.set_xlabel('unixtime')
p1.set_ylabel('upvotes')

tempx=[]
tempy=[]
tempz=[]
temp=[]
for j in range(length):
	if image[j]=='5919':
		temp.append((unixtime[j],comments[j],upvotes[j]))
temp.sort()
for (time,co,up) in temp:
	tempx.append(time)
	tempy.append(co)
	tempz.append(up)
p1=plt.subplot(512)
#p1.plot(tempx,tempy,label='comments',color='r')
p1.plot(tempx,tempz,label='upvotes',color='b')
p1.set_xlabel('unixtime')
p1.set_ylabel('upvotes')

tempx=[]
tempy=[]
tempz=[]
temp=[]
for j in range(length):
	if image[j]=='174':
		temp.append((unixtime[j],comments[j],upvotes[j]))
temp.sort()
for (time,co,up) in temp:
	tempx.append(time)
	tempy.append(co)
	tempz.append(up)
p1=plt.subplot(513)
#p1.plot(tempx,tempy,label='comments',color='r')
p1.plot(tempx,tempz,label='upvotes',color='b')
p1.set_xlabel('unixtime')
p1.set_ylabel('upvotes')

tempx=[]
tempy=[]
tempz=[]
temp=[]
for j in range(length):
	if image[j]=='6219':
		temp.append((unixtime[j],comments[j],upvotes[j]))
temp.sort()
for (time,co,up) in temp:
	tempx.append(time)
	tempy.append(co)
	tempz.append(up)
p1=plt.subplot(514)
#p1.plot(tempx,tempy,label='comments',color='r')
p1.plot(tempx,tempz,label='upvotes',color='b')
p1.set_xlabel('unixtime')
p1.set_ylabel('upvotes')

tempx=[]
tempy=[]
tempz=[]
temp=[]
for j in range(length):
	if image[j]=='996':
		temp.append((unixtime[j],comments[j],upvotes[j]))
temp.sort()
for (time,co,up) in temp:
	tempx.append(time)
	tempy.append(co)
	tempz.append(up)
p1=plt.subplot(515)
#p1.plot(tempx,tempy,label='comments',color='r')
p1.plot(tempx,tempz,label='upvotes',color='b')
p1.set_xlabel('unixtime')
p1.set_ylabel('upvotes')

plt.show()
'''
#print len(noone)
noone.sort()
noone.reverse()
#print noone



#test.sort()
#test.reverse()
#print test[:20]

#plt.show()
#f=plt.figure()
#plt.plot(x,y)
#plt.show()