#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import pickle
import scipy.special
import matplotlib.pyplot as plt
from scipy.linalg import expm, sinm, cosm
import tensorflow_scientific as tfs
import tensorflow_probability as tfp


# In[2]:


def collectData(n_instances,n_test):
    with open('forcedosc7.pickle', 'rb') as f:
        
        # import dataset
        
        [X_train,y_train,X_test,y_test] = pickle.load(f)
        X_train = X_train[0:n_instances, :, :]
        y_train = y_train[0:n_instances, :]
        X_test = X_test[0:n_test, :, :]
        y_test = y_test[0:n_test, :]

        
        # label is written as one-hot variables.
        y_train_onehot = np.zeros((n_instances, 1+int(np.max(y_train))))
        y_train_onehot[np.arange(n_instances),np.squeeze(y_train)] = 1
        y_test_onehot = np.zeros((n_test, 1+int(np.max(y_test))))
        y_test_onehot[np.arange(n_test),np.squeeze(y_test)] = 1

        # n_classes : number of labels
        # n_steps : length of s
        n_classes = 1 + int(np.max(y_train))
        n_steps  = X_train.shape[1]
        n = X_train.shape[2]


        return X_train,y_train_onehot,n_classes,n_steps,n,y_train,X_test,y_test_onehot,y_test

# trim dataset
n_instances =8000
n_test = 2000


X_train,y_train_onehot,n_classes,n_steps,n,y_train,X_test,y_test_onehot,y_test = collectData(n_instances,n_test)
X_train = X_train.astype('float32')


# In[5]:


# Construct the dictionary consisting of polynomial functions with maximum degree.
# m : dimension of hidden variables
# mul : 2*pi/period 
# d : dimension of dictionary


mul = 2*np.pi/10 
m = 2
d = 9


# In[6]:


def bigxi2(ht):
    h1 = mul*ht[:,0]
    h2 = mul*ht[:,1]
    return tf.transpose(tf.constant([np.ones(ht.shape[0]), np.cos(h1), np.sin(h1), np.cos(h2), np.cos(h2)*np.cos(h1),
                                  np.cos(h2)*np.sin(h1), np.sin(h2), np.sin(h2)*np.cos(h1), np.sin(h2)*np.sin(h1)]
                                  ,dtype = 'float32'))
    


# In[7]:




beta = tf.Variable(tf.random.uniform(minval=-1, maxval=1,shape = [d, m]), dtype='float32')
B = tf.Variable(tf.random.uniform(minval=-1, maxval=1,shape = [n, m]), dtype='float32')
A = tf.Variable(tf.random.uniform(minval=-1, maxval=1,shape = [m, n_classes]), dtype='float32')
b = tf.Variable(tf.random.uniform([1, n_classes]), dtype='float32')



# In[8]:


def bigxiprime(ht):
    with tf.GradientTape() as g:
        g.watch(ht)
        
        bx = bigxi2(ht)

   
   
    bigxip = g.batch_jacobian(bx, ht)
    
    out = tf.where(tf.math.logical_not(tf.math.is_nan(bigxip)),bigxip,tf.constant(0.0,dtype='float32'))
    return out


# In[10]:


# time domian  = [0,fT]
fT = 10.0

# need for solving ODE
numsteps = n_steps

# dt = fT/numsteps

tgrid = np.linspace(0.,fT,numsteps,dtype= 'float32')
dt = tgrid[1]-tgrid[0]

# need for interpolation when we observe the input X

xtgrid= np.linspace(0.,fT,n_steps,dtype= 'float32')
xdt = xtgrid[1]-xtgrid[0]


# In[21]:


# RHS of ode for h 
def forwardode(ht, t):
    
    part1= tf.matmul( bigxi2(ht),beta)
    # location of xstep right next to the given t
    ii = tf.where((xtgrid >=t))[0][0] 
    
    # linear interporation of x
    xinterp = ((xtgrid[ii] - t)*X_train[:,ii-1,:] + (t - xtgrid[ii-1])*X_train[:,ii,:])/xdt
    
    part2 = tf.matmul(xinterp, B)
    

    return part1+part2


# In[22]:


def forwardode_batch(ht, t):
    part1= tf.matmul( bigxi2(ht),beta)
    
    
    ii = tf.where((xtgrid >=t))[0][0] 
    xinterp = ((xtgrid[ii] - t)*X_train_batch[:,ii-1,:] + (t - xtgrid[ii-1])*X_train_batch[:,ii,:])/xdt
    
    part2 = tf.matmul(xinterp, B)
    

    return part1+part2


# In[24]:


def forwardode_test(ht, t):
    part1= tf.matmul( bigxi2(ht),beta)
    
    
    ii = tf.where((xtgrid >=t))[0][0] 
    xinterp = ((xtgrid[ii] - t)*X_test[:,ii-1,:] + (t - xtgrid[ii-1])*X_test[:,ii,:])/xdt
    
    part2 = tf.matmul(xinterp, B)
    

    return part1+part2


# In[25]:


def forwardsolver(forwardode,n_instances):
    forwardsol = tfs.integrate.odeint_fixed(func=forwardode, 
                           y0=tf.zeros([n_instances,m], dtype='float32'),
                           t=tf.constant(tgrid, dtype='float32'),
                           dt=tf.constant(dt, dtype='float32'))
    solvedh = tf.transpose(forwardsol,(1,0,2)) 
    return solvedh


# In[26]:


# compute the gradient
def get_gradient_A(solvedh,A,b,y_train_onehot,n_classes):

    ypred = tf.nn.softmax(tf.matmul(solvedh[:,-1,:],A)+b)

    diff =tf.cast(y_train_onehot,dtype='float32')- ypred

    grad_h = (-1.0/n_instances)*tf.matmul(diff,tf.transpose(A))
    
    grad_A = (-1.0/n_instances)*tf.matmul(tf.transpose(solvedh[:,-1,:]),diff)

    grad_b = tf.reshape((-1.0/n_instances)*tf.reduce_sum(diff,0),(1,n_classes))
    
    return grad_h,grad_A,grad_b,ypred


# In[16]:


def backwardode(mus, s):
    t = tf.constant(fT, dtype='float64') - tf.cast(s, dtype='float64')
    t2 = tf.constant(fT, dtype='float32') - s
    ii = tf.where((tgrid >= t))[0][0]
    
    # linear interportation of h
    hinterp = ((tgrid[ii] - t2)*solvedh[:,ii-1,:] + (t2 - tgrid[ii-1])*solvedh[:,ii,:])/dt
    h1 = mul*hinterp[:,0].numpy()
    h2 = mul*hinterp[:,1].numpy()

    
    # compute the jacobian of dictionary
    
    # In general case, 
    # bigxip = bigxiprime(hinterp)
    
    
    dh1 = tf.zeros((n_instances,2))
    dh2 = tf.stack([-tf.math.sin(h1),tf.zeros((n_instances,))],axis =1)
    dh3 = tf.stack([tf.math.cos(h1),tf.zeros((n_instances,))],axis =1)
    
    dh4 = tf.stack([tf.zeros((n_instances,)) , -tf.math.sin(h2)],axis =1)
    dh5 = tf.stack([-tf.math.sin(h1)*tf.math.cos(h2) , -tf.math.cos(h1)*tf.math.sin(h2)],axis =1)
    dh6 = tf.stack([tf.math.cos(h1)*tf.math.cos(h2) , -tf.math.sin(h1)*tf.math.sin(h2)],axis =1)
  
    dh7 = tf.stack([tf.zeros((n_instances,)) , tf.math.cos(h2)],axis =1)
    dh8 = tf.stack([-tf.math.sin(h1)*tf.math.sin(h2) , tf.math.cos(h1)*tf.math.cos(h2)],axis =1)
    dh9 = tf.stack([tf.math.cos(h1)*tf.math.sin(h2) , tf.math.sin(h1)*tf.math.cos(h2)],axis =1)
    
    bigxip = mul*tf.stack([dh1,dh2,dh3,dh4,dh5,dh6,dh7,dh8,dh9],axis = 1)
    
    rhsmat = tf.matmul(tf.transpose(bigxip,perm = [0,2,1]),beta)
    
    # because of change of variable, ouput should be multiplied by -1
    output = tf.einsum('ijk,ik->ij',rhsmat,mus)
    return output


# In[17]:


def backwardsolver(backwardode,grad_h):
    backwardsol = tfs.integrate.odeint_fixed(func=backwardode, 
                                              y0=-grad_h, 
                                              t=tf.constant(tgrid, dtype='float32'),
                                              dt=tf.constant(dt, dtype='float32'))

    solvedlam = tf.reverse(tf.transpose(backwardsol, perm=[1,0,2]),[1])
    
    return solvedlam




# In[19]:


def get_gradient_beta(solvedh,solvedlam,X_train,dt):
    
    bigxiatsolvedh = tf.transpose(tf.map_fn(fn = bigxi2, elems = solvedh),perm =[0,2,1])
    grad_beta =-tf.reduce_sum(tf.matmul(bigxiatsolvedh,solvedlam),axis = 0)*dt
    grad_B = -tf.reduce_sum(tf.matmul(tf.transpose(X_train,perm = [0,2,1]),solvedlam),axis = 0)*dt
    return grad_beta,grad_B


# In[27]:


lr = 0.01
bce = tf.keras.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(learning_rate=lr)


# In[ ]:


n_batch = 8000
n_epoch = 300

# Loss and Error are saved
Loss = [] 
Loss_test =[]
Error = [] 
Error_test = []

for i in range(n_epoch):
    
    
    

    # start learning

    np.random.shuffle([X_train,y_train])
    for batch in range(tf.dtypes.cast(n_instances/n_batch, tf.int32) ):
        
        X_train_batch = X_train[batch*n_batch:(batch+1)*n_batch]
        y_train_onehot_batch = y_train_onehot[batch*n_batch:(batch+1)*n_batch]



        solvedh_batch = forwardsolver(forwardode_batch,n_batch)
        grad_h,grad_A,grad_b,ypred = get_gradient_A(solvedh_batch,A,b,y_train_onehot_batch,n_classes)
    
        solvedlam = backwardsolver(backwardode,grad_h)
        grad_beta,grad_B = get_gradient_beta(solvedh_batch,solvedlam,X_train_batch,dt)
        
        
        grad_var_list = [(grad_beta,beta),(grad_B,B),(grad_A,A),(grad_b,b)]
        opt.apply_gradients(grad_var_list)
        
        # change cutoff value(lambda) for sparsity
        lam = tf.constant(0.0,dtype = 'float32')
        beta.assign(tf.where(tf.greater_equal(tf.abs(beta), lam), beta, tf.constant(0.0,dtype='float32')))
    
     # train error          
    solvedh = forwardsolver(forwardode,n_instances)
    grad_h,grad_A,grad_b,ypred = get_gradient_A(solvedh,A,b,y_train_onehot,n_classes)
        
    ypred = tf.nn.softmax(tf.matmul(solvedh[:,-1,:],A)+b)

    loss = bce(ypred,y_train_onehot)
    PPred = np.argmax(ypred,1)

    mis=0
    for j in range(n_instances):
        if (PPred[j]!=y_train[j])==True:
            mis = mis + 1 
    Loss.append(loss.numpy())
    Error.append(mis/n_instances)       

    

    print(i,"------")
    print("loss", loss.numpy())
    print("train error",mis/n_instances)
    
        
# test error
solvedh_test = forwardsolver(forwardode_test,n_test)
ypred_test = tf.nn.softmax(tf.matmul(solvedh_test[:,-1,:],A)+b)

loss_test = bce(ypred_test,y_test_onehot)
PPred_test = np.argmax(ypred_test,1)

mis_test=0
for j in range(n_test):
    if (PPred_test[j]!=y_test[j])==True:
        mis_test = mis_test + 1 
        
Loss_test.append(loss_test.numpy())
Error_test.append(mis_test/n_test)    
print("test error",mis_test/n_test)
      
       
    


# In[27]:


fig, ((ax1, ax2)) = plt.subplots(1, 2, constrained_layout=True,figsize= (15,10) )
ax1.plot(Loss,'b')
ax1.plot(Loss_test,'r')
ax1.set_title('Loss')
ax2.plot(Error)
ax2.plot(Error_test,'r')
ax2.set_title('Error')

plt.show()

