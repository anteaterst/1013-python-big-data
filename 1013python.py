#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[3]:


names = np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])

data = np. random.randn(7,4)

names


# In[4]:


data


# In[5]:


names == 'Bob'


# In[6]:


data[names == 'Bob']


# In[7]:


data[names == 'Bob', 2:]


# In[8]:


data[names == 'Bob', 3]


# In[9]:


names != 'Bob'


# In[12]:


data[~(names == 'Bob')]


# In[13]:


data[names != 'Joe'] = 7 


# In[14]:


data


# In[15]:


arr = np.empty((8,4))


# In[16]:


for i in range(8):
    arr[i]=i


# In[17]:


arr


# In[18]:


arr[[4,3,0,6]]


# In[20]:


arr[[-3,-5,-7]]


# In[21]:


##배열 전치와 축바꾸기
#배열 전치는 데이터를 복사하지 않고 데이터의 모양이 바뀐 뷰를 반환하는 특별한 기능이다.
arr = np.arange(15).reshape((3,5))


# In[22]:


arr


# In[23]:


arr.T##행렬 계산을 할때 자주 사용하게 될텐데 예를들어 행렬의 내적은 np.dot을 이용해서 구할 수 있다.


# In[24]:


arr = np.random.randn(6,3)#다차원의 배열의 경우는 transpose 메서드는 튜플로 축번을 받아서 치환한다.


# In[25]:


arr


# In[26]:


np.dot(arr.T,arr)


# In[37]:


arr.T 


# In[33]:


arr = np.arange(16).reshape((2,2,4))


# In[34]:


arr


# In[35]:


arr.transpose((1,0,2))


# In[38]:


arr


# In[39]:


arr.swapaxes(1,2)


# In[43]:


np.dot(arr.T,arr)


# In[44]:


arr = np.arange(16).reshape((2,2,4))


# In[45]:


arr


# In[46]:


arr.transpose((1,0,2))


# In[47]:


arr.swapaxes(1, 2)


# In[51]:


#유니버셜 함수 : 배열의 각 원소를 빠르게 처리하는 함수

#ufunc 라고 불리기도 한다 .ndarray안에 있는 데이터 원소별로 연산을 수행 

arr = np.arange(10)


# In[52]:


arr


# In[54]:


np.sqrt(arr)


# In[55]:


np.exp(arr)


# In[56]:


##배열을 이용한 배열 지향 프로그래밍 

#numpy 배열을 사용하면 반복문을 작성하지 않고 간결한 연산 사용

points = np.arange(-5,5,0.01) 


# In[57]:


xs, ys = np.meshgrid(points, points)


# In[58]:


ys   ##numpy 배열을 사용하면 반복문을 자성하지 않고 간결한 연산 사용


# In[59]:


#예로 값들이 놓여있는 그리드 sqart(???)를 계산한다고 하자
#np.meshgird함수는 두개의 1차원 배열을 받아서 가능한 모든 짝을 만들 수  있는 2차원 배열 두개를 반환한다.


# In[64]:


z = np.sqrt(xs ** + ys ** 2)


# In[63]:


z


# In[65]:


##배열 연산으로 조건절 표현하기

# ccnd 값이 Ture 일 경우 xarr 의 값을 취하거나 아니면 yarr의 값을 취한다


# In[66]:


xarr = np.array([1.1,1.2,1.3,1.4,1.5])


# In[67]:


yarr = np.array([2.1,2.2,2.3,2.4,2.5])


# In[68]:


cond = np.array([True, False,True,True,False])


# In[69]:


result = [(x if c else y)
         for x, y, c in zip(xarr, yarr, cond)]


# In[70]:


result


# In[71]:


result = np.where(cond,xarr,yarr)


# In[72]:


result


# In[74]:


arr = np.random.randn(4,4)


# In[75]:


arr


# In[76]:


arr > 0


# In[77]:


np.where(arr > 0,2,-2)


# In[78]:


np.where(arr > 0,2, arr)


# In[79]:


arr = np.random.randn(5,4)


# In[80]:


arr


# In[81]:


arr.mean()


# In[82]:


np.mean(arr)


# In[83]:


arr.sum()


# In[84]:


arr.mean(axis=1)


# In[85]:


arr.sum(axis=0)


# In[86]:


arr.mean(axis=1)


# In[87]:


arr.sum(axis=0)


# In[88]:


arr = np.array([0,1,2,3,4,5,6,7,])


# In[89]:


arr.cumsum()


# In[90]:


arr = np.array([[0,1,2,],[3,4,5],[6,7,8]])


# In[91]:


arr


# In[92]:


arr.cumsum(axis=0)


# In[93]:


arr.cumprod(axis=1)


# In[94]:


arr = np.random.randn(100)


# In[95]:


(arr>0).sum()


# In[96]:


bools = np.array([False, False, True, False])


# In[97]:


bools.any()


# In[98]:


bools.all()


# In[99]:


arr = np.random.randn(6)


# In[100]:


arr


# In[101]:


arr.sort()


# In[102]:


arr


# In[103]:


arr = np.random.randn(5,3)


# In[104]:


arr


# In[105]:


arr.sort(1)


# In[106]:


arr


# In[107]:


large_arr = np.random.randn(1000)


# In[108]:


large_arr.sort()


# In[109]:


large_arr[int(0.05 * len(large_arr))] #5% quantile


# In[110]:


names = np.array(['Bob','Joe','Will','Bob','Will','Joe'])
np.unique(names)


# In[111]:


ints = np.array([3,3,3,2,2,1,1,4,4])


# In[112]:


np.unique(ints)


# In[115]:


sorted(set(names))


# In[116]:


x = np.array([[1.,2.,3.],[4.,5.,6.]])
y = np.array([[6.,23.],[-1,7],[8,9]])
x


# In[117]:


y


# In[118]:


x.dot(y)


# In[119]:


np.dot(x,y)


# In[120]:


#난수생성


# In[121]:


samples = np.random.normal(size=(4,4))


# In[122]:


samples


# In[123]:


from random import normalvariate


# In[125]:


n= 1000000


# In[130]:


from random import normalvariate


# In[133]:


get_ipython().run_line_magic('timeit', 'samples = [normalvariate(0, 1)for _ in range(N)]')
1.77 s +- 126 ms per loop (mean +- std. dev. ofr 7 runs, 1 loop each)


# In[135]:


get_ipython().run_line_magic('timeit', 'np.random.normal(size=N)')
61.7 ms +- 1.32 ms per loop (mean +- std. dev. of 7 runs, 10 loops each)


# In[137]:


np.random.seed(1234)


# In[139]:


import random
position = 0
walk = [position]
steps = 1000
for i in range(steps):
    step = 1 if random.randint(0,1) else -1 
    position += step
    walk.append(position)


# In[141]:


import matplotlib.pyplot as plt

plt.plot(walk[:100])


# In[143]:


nsteps = 1000
draws = np.random.randint(0, 2, size=nsteps)
steps = np.where(draws > 0, 1, -1)
walk = steps.cumsum()


# In[144]:


walk.min()


# In[146]:


walk.max()


# In[147]:


(np.abs(walk) >= 10).argmax()


# In[150]:


nwalks = 5000
nsteps = 1000
draws = np.random.randint(0, 2, size=(nwalks, nsteps))
steps = np.where(draws > 0, 1, -1)
walk = steps.cumsum(1)
walk


# In[151]:


import pandas as pd


# In[152]:


from pandas import Series, DataFrame


# In[153]:


obj = pd.Series([4,7,-5,3])
obj


# In[154]:


obj.values


# In[155]:


obj.index


# In[156]:


obj2 = pd.Series([4,7,-5,3], index=['d','b','a','c'])
obj


# In[157]:


obj2.index


# In[158]:


obj2['a']


# In[160]:


obj2['d'] = 6


# In[161]:


obj2[['c','a','d']]


# In[162]:


obj2[obj2 > 0]


# In[163]:


obj2 * 2


# In[165]:


np.exp(obj2)


# In[166]:


'e' in obj2


# In[167]:


sdata = {'Ohio' : 35000, 'Texas' : 71000, 'Oregon':16000,'Utah':5000}


# In[170]:


obj3 = pd.Series(sdata)


# In[171]:


obj3


# In[172]:


states = ['California','Ohio','Oregon','Texas']


# In[173]:


obj4 = pd.Series(sdata, index=states)


# In[174]:


obj


# In[175]:


pd.isnull(obj4)


# In[176]:


pd.notnull(obj4)


# In[177]:


obj4.isnull()


# In[178]:


obj3


# In[179]:


obj4


# In[180]:


obj4.name = 'population'
obj4.index.name = 'state'
obj4


# In[181]:


obj


# In[ ]:




