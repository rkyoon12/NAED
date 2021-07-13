

<!-- PROJECT LOGO -->
<br />
<p align="center">
 
   

  <h3 align="center">A non-autonomous equation discovery method for time signal classification</h3>

  <p align="center">
    Document is available below
    <br />
    <a href="https://arxiv.org/pdf/2011.11096.pdf"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    
  </p>
</p>




<!-- ABOUT THE PROJECT -->
## About The Project

This paper proposes a non-autonomous dynamical system framwork for time signal classification. We view time signal data x(t) as continuous forcing term in a non-autonomous dynamical system of hidden variable h,

<p align= "center">
 <img src = https://user-images.githubusercontent.com/35155480/125505530-5e31710b-433b-469c-8a99-17168c18f624.gif>
</p>

Then the solution at the final time T is used to make a class prediction via 

<p align= "center">
 <img src = https://user-images.githubusercontent.com/35155480/125506036-17dadbda-bd74-4060-a501-2b1d5706ebd4.gif>
</p>


* Based on the Equation discovery method, the vector field ![ode](https://user-images.githubusercontent.com/35155480/125507030-04282eab-5445-4e98-a800-07c227d711cf.gif) is defined by Dictionary elements. 
* Using the adjoint method, compute the gradients of objective by solving alternative adjoint equation. 
* Applying the stability theory, we analyze the stability of classifier under the noise in the given data. 
* Benchmark SINDy algorithm, we thresholding the coefficients by cutoff value to prevent overfitting.


## Requirements
* Tensorflow is required to run the code. 
* Install the tensorflow-scientific for solving ODEs. 
  ( this package installation is working under the python version 3.5,3.6 and 3.7. If you are using upper version, please downgrade python first.)

 ```
 pip install tensorflow-scientific
 ```
 
<!-- Usage Example -->
## Usage Example
Explore the example which shows how this code run for time-signal classification. 

### 1. Dataset

Here, we use the synthetic data generated by the forced harmonic oscillator. For given forcing x(t)= \Sigma A_k sin(alpha_k t), solve

<p align= "center">
 <img src = https://user-images.githubusercontent.com/35155480/125506758-3842b968-acbd-4856-a418-1dd661ceafb3.gif>
</p>


Then the class will be assigned depending on the position of solution (either positive or negative) at final time T. The time series x and label y is pickled in  [osc.pickle](https://github.com/rkyoon12/NAED/blob/master/GenerateData/osc.pickle). Modifying the dynamical system in [generate_osc.py](https://github.com/rkyoon12/NAED/blob/master/GenerateData/generate_osc.py), you can generate the other synthetic dataset. 

### 2. Dictionary choice. 

In the NAED model, the right-hand side of dynamical system is given by ![ode](https://user-images.githubusercontent.com/35155480/125507030-04282eab-5445-4e98-a800-07c227d711cf.gif). Here we can pre-specify the entries of dictionary. We tested NAED model using two different dictionaries;  polynomials and Fouriers. The examples of dictionary is described in the document [paper](https://arxiv.org/pdf/2011.11096.pdf).


### 

This shows how to customize the dictionary in the main code. 
* Polynomial dictionary

 Polynomial dictionary consists of all possible polynomials of h with dimension m upto k-th order. For instance, m = 2, k= 1 then, then dictionary consists of  [1,h_1,h_2]. In the code [poly21](https://github.com/rkyoon12/NAED/blob/master/Main/poly21.py), you can change both dimension of $h$ and maximum degree of polynomials.

  ```sh
  # m : dimension of hidden variables
  # maximum degree = maxdeg - 1
  m = 2
  maxdeg = 2

  ```
* Fourier dictionary

 Fourier dictionary consists of outer product of fourier elements with respective frequency.  For instance, m = 2, K= 1 then, then dictionary consists of 

<p align= "center">
 <img src = https://user-images.githubusercontent.com/35155480/125507381-4d6d9233-54ac-4b3b-b446-78450692824d.gif>
</p>

In the codes [Fourier21](https://github.com/rkyoon12/NAED/blob/master/Main/Fourier21.py) and [Fourier22](https://github.com/rkyoon12/NAED/blob/master/Main/Fourier22.py), dictionary elements are manually written. 
 ```sh
# m : dimension of hidden variables
# mul : 2*pi/period 

mul = 2*np.pi/10 
m = 2

def bigxi2(ht):
    h1 = mul*ht[:,0]
    h2 = mul*ht[:,1]
    return tf.transpose(tf.constant([np.ones(ht.shape[0]), np.cos(h1), np.sin(h1), np.cos(h2), np.cos(h2)*np.cos(h1),
                                  np.cos(h2)*np.sin(h1), np.sin(h2), np.sin(h2)*np.cos(h1), np.sin(h2)*np.sin(h1)]
                                  ,dtype = 'float32'))
  ```

### 3. Optimization

The goal of NAED method is training all parameters in model, i.e. beta,B,A and b. We recommend to initialize all parameters by satisfying the theorem 3.1 in paper, which guarantee the existence of the solution to the dynamical system.

 ```sh
 # d : number of dictionary
 # m : dimension of h (hidden variable)
 # n : dimension of x (sequential data)
 # n_classes : number of classes
 
beta = tf.Variable(tf.random.uniform(minval=-1, maxval=1,shape = [d, m]), dtype='float32')
B = tf.Variable(tf.random.uniform(minval=-1, maxval=1,shape = [n, m]), dtype='float32')
A = tf.Variable(tf.random.uniform(minval=-1, maxval=1,shape = [m, n_classes]), dtype='float32')
b = tf.Variable(tf.random.uniform(minval=0, maxval=1,shape = [1, n_classes]), dtype='float32')
 ```

Then it is trained using the gradient-based method "ADAM" with the constant learning rate. Depending on your problem, change learning rate and optimizer in keras. 

 ```
lr = 0.01
bce = tf.keras.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(learning_rate=lr)

  ```


<!-- Results -->
## Results  

We presents the accuracy of classification and depict the learned vector fields. 
![OSC_final](https://user-images.githubusercontent.com/35155480/125504060-f261f77a-819f-4dd4-807a-764580da004e.png)

This is plotting the learned vector field of hidden variable h. The trajectories are the solution to the forced ODE. Note that the terminal position will be used to make a predcition. And the domain is partitioned by the corresponding colored sections of classes. As shown is four subplots, the termial stages of trajectories are matching with the underlying color. 

For more examples, please refer to the [paper](https://arxiv.org/pdf/2011.11096.pdf).





<!-- CONTACT -->
## Contact
Ryeongkyung Yoon - rkyoon@math.utah.edu

Project Link: [https://github.com/rkyoon12/NAED](https://github.com/rkyoon12/NAED)


<!--stackedit_data:
eyJoaXN0b3J5IjpbNzUwNDgyOTMwLC0yMDYxNDMyNzgsLTEzMT
I5MTU0MzAsLTEzMTI5MTU0MzAsLTk4MTkwMTEzN119
-->
