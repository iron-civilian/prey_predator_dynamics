#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# In[142]:


N=100 #number of prey initially


# In[21]:


beta=1
alpha=1
gamma=1
delta=1
mu=1
mu_pd=1
r_int=0.3


# In[143]:


"""
prey_pos_arr=np.array([[5,0],[-5,0],[0,5],[0,-5]]) #array of arrays of [xi,yi]
predator_pos_arr=np.array([0,0])

pos_arr_initial=list(prey_pos_arr)
pos_arr_initial.append(predator_pos_arr)
pos_arr_initial=np.array(pos_arr_initial)
#pos_arr_initial

"""

prey_pos_arr=np.random.randint(-10,10,(N,2)) #array of arrays of [xi,yi]
predator_pos_arr=np.array([0,0])
pos_arr_initial=list(prey_pos_arr)
pos_arr_initial.append(predator_pos_arr)
pos_arr_initial=np.array(pos_arr_initial)
#pos_arr_initial



# In[67]:


def distance(pos1,pos2):
    x1,y1=pos1[0],pos1[1]
    x2,y2=pos2[0],pos2[1]
    return ((x1-x2)**2+(y1-y2)**2)**0.5


# In[12]:


"""def N_int_i(pos_arr):
    pos_i=pos_arr[i]
    N=0
    for pos in pos_arr:
        if distance(pos,pos_i)!=0 and distance(pos,pos_i)<=r_int:
            N+=1
    return N"""


# In[102]:


def F_i_prey_prey(i,pos_arr):
    pos_i=pos_arr[i]
    N_int_i=0
    f_i_prey_prey=np.array([0,0])
    for pos_j in pos_arr:
        if distance(pos_j,pos_i)!=0 and distance(pos_j,pos_i)<=r_int:
            N_int_i+=1
            f_i_prey_prey=f_i_prey_prey+beta*(pos_j-pos_i)-alpha*(pos_j-pos_i)/distance(pos_i,pos_j)**2
    
    if N_int_i!=0:
        return f_i_prey_prey/N_int_i
    else:
        return np.array([0,0])


# In[92]:


def F_i_prey_predator(i,pos_arr,pos_predator):
    pos_i=pos_arr[i]
    return -gamma*(pos_predator-pos_i)/distance(pos_predator,pos_i)**2


# In[93]:


def F_predator_prey(pos_arr,pos_predator):
    f_predator_prey=np.array([0,0])
    
    for pos_i in pos_arr:
        f_predator_prey=f_predator_prey+(pos_i-pos_predator)/distance(pos_i,pos_predator)**3
    return (delta/N)*f_predator_prey


# In[94]:


def der(t,pos_arr):
    pos_arr=pos_arr.reshape((-1,2))
    pos_arr_prey=pos_arr[:-1]
    pos_predator=pos_arr[-1]
    dposdt=[]
    
    for i in range(len(pos_arr_prey)):
        dposdt.append((1/mu)*(F_i_prey_prey(i,pos_arr_prey)+F_i_prey_predator(i,pos_arr_prey,pos_predator)))
    dposdt.append((1/mu_pd)*F_predator_prey(pos_arr_prey,pos_predator))
    
    return (np.array(dposdt).flatten())
    


# In[113]:


t0,tf=0,400


# In[137]:


t_points=np.linspace(t0,tf,10000)


# In[144]:


sol=solve_ivp(fun=der,t_span=(t0,tf),y0=pos_arr_initial.flatten(),method='RK45',t_eval=t_points)


# In[145]:


pos_arr_final=(sol.y)[:,-1].reshape((-1,2))


# In[146]:


plt.scatter(prey_pos_arr[:,0],prey_pos_arr[:,1],color='blue')
plt.scatter(predator_pos_arr[0],predator_pos_arr[1],color='red')


# In[147]:


plt.scatter(pos_arr_final[:-1,0],pos_arr_final[:-1,1],color='blue')
plt.scatter(pos_arr_final[-1,0],pos_arr_final[-1,1],color='red')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




