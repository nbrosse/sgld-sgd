import numpy as np
from sklearn.metrics import log_loss
import time
#import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sys

class Stopwatch:
    """Define tic() and toc() for calculating time"""
    def __init__(self):
        self.current = time.time()

    def tic(self):
        """Reset stopwatch"""
        self.current = time.time()

    def toc(self):
        """Return elapsed time"""
        elapsed = time.time() - self.current
        # Reset timer
        self.current = time.time()
        return elapsed

class LogisticRegression:
    """
    Methods for performing Bayesian logistic regression for large datasets.

    Logistic regression is trained using stochastic gradient Langevin dynamics

    References: 
        1. Logistic regression - https://www.stat.cmu.edu/~cshalizi/uADA/12/lectures/ch12.pdf
        2. Stochastic gradient Langevin dynamics - 
                http://people.ee.duke.edu/~lcarin/398_icmlpaper.pdf
    """

    def __init__(self,X_train,X_test,y_train,y_test):
        """
        Initialise the logistic regression object.

        Parameters:
        X_train - matrix of explanatory variables for training (assumes numpy array of floats)
        X_test - matrix of explanatory variables for testing (assumes numpy array of ints)
        y_train - vector of response variables for training (assumes numpy array of ints)
        y_train - vector of response variables for testing (assumes numpy array of ints)
        """
        # Set error to be raised if there's an over/under flow
        np.seterr( over = 'raise', under = 'raise' )
        self.X = X_train
        self.y = y_train
        self.X_test = X_test
        self.y_test = y_test

        # Set dimension constants
        self.N = self.X.shape[0]
        self.d = self.X.shape[1]
        self.test_size = self.X_test.shape[0]
        
        # Initialise containers
        # Logistic regression parameters (assume bias term encoded in design matrix)
        self.beta = np.random.rand(self.d)
        self.beta_mode = np.zeros(self.d)
        self.full_post = np.zeros(self.d)
        # Storage for beta samples and gradients of the log posterior during fitting
        self.sample = None
        self.grad_sample = None
        # Storage for logloss and time values during fitting
        self.training_loss = []
        self.n_iters = None
        
    def truncate(self,train_size,test_size):
        self.X = self.X[:train_size,:]
        self.y = self.y[:train_size]
        self.X_test = self.X_test[:test_size,:]
        self.y_test = self.y_test[:test_size]
        self.N = train_size
        self.test_size = test_size


    def fit_sgldfp(self,stepsize,n_iters=10**4,minibatch_size=500):
        """
        Fit Bayesian logistic regression model using train and test set.

        Uses stochastic gradient Langevin dynamics algorithm fixed point

        Parameters:
        stepsize - stepsize to use in stochastic gradient descent
        n_iters - number of iterations of stochastic gradient descent (optional)
        minibatch_size - minibatch size in stochastic gradient descent (optional)
        """
        # Start chain from mode
        # Run fit_sgd() before
        self.beta = self.beta_mode.copy()
        # Holds log loss values once fitted
        self.training_loss = []
        # Number of iterations before the logloss is stored
        self.loss_thinning = 10
        # Initialize sample storage
        self.n_iters = n_iters
        self.sample = np.zeros( ( self.n_iters, self.d ) )
        self.grad_sample = np.zeros( ( self.n_iters, self.d ) )
        # Calculate likelihood at beta mode
        self.full_post_computation()
        print ("Fitting chain...")
        print ("{0}\t{1}".format( "iteration", "Test log loss" ))
        timer = Stopwatch()
        for i in np.arange(self.n_iters):
            # Every so often output log loss on test set and store 
            if i % self.loss_thinning == 0:
                elapsed_time = timer.toc()
                current_loss = self.logloss()
                self.training_loss.append( [current_loss,elapsed_time] )
                print ("{0}\t\t{1}\t\t{2}".format( i, current_loss, elapsed_time ))
                timer.tic()
            self.sample_minibatch(minibatch_size)
            self.sample[i,:] = self.beta
            # Calculate gradients at current point
            dlogbeta, dlogbetaopt = self.dlogpostcv()
            dlogbetacv = self.full_post + ( dlogbeta - dlogbetaopt ) 
            self.grad_sample[i,:] = dlogbetacv
            # Update parameters using SGD
            eta = np.sqrt( stepsize ) * np.random.normal( size = self.d )
            self.beta += stepsize / 2 * dlogbetacv + eta

    def fit_sgld(self,stepsize,n_iters=10**4,minibatch_size=500):
        """
        Fit Bayesian logistic regression model using train and test set.

        Uses stochastic gradient Langevin dynamics algorithm

        Parameters:
        stepsize - stepsize to use in stochastic gradient descent
        n_iters - number of iterations of stochastic gradient descent (optional)
        minibatch_size - minibatch size in stochastic gradient descent (optional)
        """
        # Start chain from mode
        # Run fit_sgd() before
        self.beta = self.beta_mode.copy()
        # Holds log loss values once fitted
        self.training_loss = []
        # Number of iterations before the logloss is stored
        self.loss_thinning = 10
        # Initialize sample storage
        self.n_iters = n_iters
        self.sample = np.zeros( ( self.n_iters, self.d ) )
        self.grad_sample = np.zeros( ( self.n_iters, self.d ) )
        print ("Fitting chain...")
        print ("{0}\t{1}".format( "iteration", "Test log loss" ))
        timer = Stopwatch()
        for i in np.arange(self.n_iters):
            # Every so often output log loss on test set and store 
            if i % self.loss_thinning == 0:
                elapsed_time = timer.toc()
                current_loss = self.logloss()
                self.training_loss.append( [current_loss,elapsed_time] )
                print ("{0}\t\t{1}\t\t{2}".format( i, current_loss, elapsed_time ))
                timer.tic()
            self.sample_minibatch(minibatch_size)
            # Calculate gradients at current point
            dlogbeta = self.dlogpost()
            self.grad_sample[i,:] = dlogbeta
            self.sample[i,:] = self.beta
            # Update parameters using SGD
            eta = np.sqrt( stepsize ) * np.random.normal( size = self.d )
            self.beta += stepsize / 2 * ( dlogbeta ) + eta

    def fit_sgd(self,stepsize,n_iters=10**4,minibatch_size=500):
        """
        Fit Bayesian logistic regression model using train and test set.

        Uses stochastic gradient Langevin dynamics algorithm

        Parameters:
        stepsize - stepsize to use in stochastic gradient descent
        n_iters - number of iterations of stochastic gradient descent (optional)
        minibatch_size - minibatch size in stochastic gradient descent (optional)
        """
        # Holds log loss values once fitted
        self.training_loss = []
        # Number of iterations before the logloss is stored
        self.loss_thinning = 10
        # Initialize sample storage
        self.n_iters = n_iters
        self.sample = np.zeros( ( self.n_iters, self.d ) )
        self.grad_sample = np.zeros( ( self.n_iters, self.d ) )
        print ("Fitting using optimization procedure")
        print ("{0}\t{1}".format( "iteration", "Test log loss" ))
        timer = Stopwatch()
        self.beta = np.random.rand(self.d)
        for i in np.arange(self.n_iters):
            # Every so often output log loss and elapsed time on test set and store 
            if i % self.loss_thinning == 0:
                elapsed_time = timer.toc()
                current_loss = self.logloss()
                self.training_loss.append( [current_loss,elapsed_time] )
                print ("{0}\t\t{1}\t\t{2}".format( i, current_loss, elapsed_time ))
                timer.tic()
            self.sample_minibatch(minibatch_size)
            # Calculate gradients at current point
            dlogbeta = self.dlogpost()
            self.grad_sample[i,:] = dlogbeta
            self.sample[i,:] = self.beta
            # Update parameters using SGD
            self.beta += stepsize / 2 * dlogbeta
        self.beta_mode = self.beta


    def logloss(self):
        """Calculate the log loss on the test set, used to check convergence"""
        y_pred = np.zeros(self.test_size, dtype = int)
        for i in range(self.test_size):
            x = np.squeeze( np.copy( self.X_test[i,:] ) )
            y_pred[i] = int( np.dot( self.beta, x ) >= 0.0 )
        return log_loss( self.y_test, y_pred )


    def loglossp(self,beta):
        """
        Calculate the log loss on the test set for specified parameter values beta
        
        Parameters:
        beta - a vector of logistic regression parameters (float array)
        """
        y_pred = np.zeros(self.test_size, dtype = int)
        for i in range(self.test_size):
            x = np.squeeze( np.copy( self.X_test[i,:] ) )
            y_pred[i] = int( np.dot( beta, x ) >= 0.0 )
        return log_loss( self.y_test, y_pred )


    def dlogpost(self):
        """
        Calculate gradient of the log posterior wrt the parameters using a minibatch of data

        Returns:
        dlogbeta - gradient of the log likelihood wrt the parameter beta 
        """
        minibatch_size = len(self.minibatch)
        dlogbeta = np.zeros( self.d )
        # Calculate sum of gradients at each point in the minibatch
        for i in self.minibatch:
            x = np.squeeze( np.copy( self.X[i,:] ) )
            y = self.y[i]
            # Calculate gradient of the log density at current point, use to update dlogbeta
            # Handle overflow gracefully by catching numpy's error
            # (seterr was defined at start of class)
            dlogbeta += ( y - 1 / ( 1 + np.exp( - np.dot( self.beta, x ) ) ) ) * x
        # Adjust log density gradients so they're unbiased
        dlogbeta *= self.N / minibatch_size
        # Add gradient of log prior (assume Laplace prior with scale 1)
#        dlogbeta -= np.sign(self.beta)
        # Add gradient of log prior (assume Gaussian prior with scale 1)
        dlogbeta -= self.beta
        return dlogbeta


    def dlogpostcv(self):
        """
        Calculate gradient of the log posterior wrt the parameters using a minibatch of data

        Returns:
        dlogbeta - gradient of the log likelihood wrt the parameter beta 
        """
        minibatch_size = len(self.minibatch)
        dlogbeta = np.zeros( self.d )
        dlogbetaopt = np.zeros( self.d )
        # Calculate sum of gradients at each point in the minibatch
        for i in self.minibatch:
            x = np.squeeze( np.copy( self.X[i,:] ) )
            y = self.y[i]
            # Calculate gradient of the log density at current point, use to update dlogbeta
            # Handle overflow gracefully by catching numpy's error
            # (seterr was defined at start of class)
            dlogbeta += ( y - 1 / ( 1 + np.exp( - np.dot( self.beta, x ) ) ) ) * x
            dlogbetaopt += ( y - 1 / ( 1 + np.exp( - np.dot( self.beta_mode, x ) ) ) ) * x
        # Adjust log density gradients so they're unbiased
        dlogbeta *= self.N / minibatch_size
        dlogbetaopt *= self.N / minibatch_size
        # Add gradient of log prior (assume Laplace prior with scale 1)
#        dlogbeta -= np.sign(self.beta)
        # Add gradient of log prior (assume Gaussian prior with scale 1)
        dlogbeta -= self.beta        
#        dlogbetaopt -= np.sign(self.beta_mode)
        dlogbetaopt -= self.beta_mode
        return dlogbeta, dlogbetaopt

    def sample_minibatch(self, minibatch_size):
        """Sample the next minibatch"""
        self.minibatch = np.random.choice( np.arange( self.N ), minibatch_size, replace = False )

    def full_post_computation(self):
        self.minibatch = np.arange(self.N)
        dlogbeta, dlogbetaopt = self.dlogpostcv()
        self.full_post = dlogbetaopt

#%% Restriction of the state space

X_train = np.load( 'cover_type/X_train.dat' )
X_test = np.load( 'cover_type/X_test.dat' )
y_train = np.load( 'cover_type/y_train.dat' )
y_test = np.load( 'cover_type/y_test.dat' )

d = X_train.shape[1]
N_tab = np.array([10**3, 10**4, 10**5, X_train.shape[0]], dtype=np.int32)

dd = 27

X_train_dict = {}
X_test_dict = {}

#eig_val_tab = np.zeros((len(N_tab), d))

for i, N in enumerate(N_tab):
    X_tr = X_train[:N,:]
    _, eig_vec = np.linalg.eig(X_tr.T @ X_tr)
    X_train_dict[str(N)] = X_tr @ eig_vec[:,:dd]
    X_test_dict[str(N)] = X_test @ eig_vec[:,:dd]
    
#%% Computation of the modes

#X_train = np.load( 'cover_type/X_train.dat' )
#X_test = np.load( 'cover_type/X_test.dat' )
#y_train = np.load( 'cover_type/y_train.dat' )
#y_test = np.load( 'cover_type/y_test.dat' )

#N_tab = np.array([10**3, 10**4, 10**5, X_train.shape[0]], dtype=np.int32)
#
#d = X_train.shape[1]
#n_iter = 10**3
#
#
#beta_mode_tab = np.zeros((len(N_tab), dd))
#full_post_tab = np.zeros((len(N_tab), dd))
#
#for i, N in enumerate(N_tab):
#    N_trunc = N_tab[i]
#    X_tr = X_train_dict[str(N)]
#    X_te = X_test_dict[str(N)]
#    lr = LogisticRegression( X_tr, X_te, y_train, y_test )
#    step = 1./float(N_trunc)
#    lr.truncate(N_trunc, X_test.shape[0])
#    lr.fit_sgd(step,n_iters=n_iter,minibatch_size=500)
#    
#    X = np.array(lr.X)
#    Y = lr.y
#    
#    def U(x):
#       r = (1./2.)*np.linalg.norm(x)**2 - Y.T @ X @ x + np.sum(np.log(1.+np.exp(X @ x)))
#       return r
#    
#    def gradU(x):
#       grad = - X.T @ Y + X.T @ (1./(1+np.exp(-X @ x))) + x
#       return grad
#
#    resultat = minimize(U, x0=lr.beta_mode, jac=gradU)
#    beta_mode = resultat['x']
#    beta_mode_tab[i,:] = beta_mode
#    
#    # Sanity check
#    lr.beta_mode = beta_mode
#    lr.full_post_computation()
#    print('iteration ', i)
#    print('--------------------------------')
#    print(lr.full_post)
#    full_post_tab[i,:] = lr.full_post
#
#np.save('beta_mode_tab_dd.npy', beta_mode_tab)

#%% Test running SGLD and SLDFP

X_train = np.load( 'cover_type/X_train.dat' )
X_test = np.load( 'cover_type/X_test.dat' )
y_train = np.load( 'cover_type/y_train.dat' )
y_test = np.load( 'cover_type/y_test.dat' )

beta_mode_tab = np.load('beta_mode_tab_dd.npy')

N_tab = np.array([10**3, 10**4, 10**5, X_train.shape[0]], dtype=np.int32)
n_iter_tab = 10**2 * N_tab

#str_N = sys.argv[1]
str_N = 'N3'

if str_N=='N3':
    i = 0
elif str_N=='N4':
    i = 1
elif str_N=='N5':
    i = 2
else:
    i = 3
    
N_trunc = N_tab[i]
n_iter = n_iter_tab[i]
beta_mode= beta_mode_tab[i,:]

X_train = X_train_dict[str(N_trunc)]
X_test = X_test_dict[str(N_trunc)]

lr = LogisticRegression( X_train, X_test, y_train, y_test )
step = 1./float(N_trunc)
lr.truncate(N_trunc, X_test.shape[0])
lr.beta_mode = beta_mode
lr.fit_sgld(step,n_iters=n_iter,minibatch_size=50)
grad_sample_sgld = lr.grad_sample
sample_sgld = lr.sample
#var_grad = np.mean(np.var(lr.grad_sample,axis=0))
#var_traj = np.var(lr.sample, axis=0)    
lr.fit_sgldfp(step,n_iters=n_iter,minibatch_size=50)
grad_sample_fp = lr.grad_sample
sample_fp = lr.sample
#var_grad_fp = np.mean(np.var(lr.grad_sample,axis=0))
#var_traj_fp = np.var(lr.sample, axis=0)
#np.savez(str_N, var_grad=var_grad, var_traj=var_traj, var_grad_fp=var_grad_fp, 
#         var_traj_fp=var_traj_fp)
np.savez(str_N, grad_sample_sgld=grad_sample_sgld, sample_sgld=sample_sgld, 
         sample_fp=sample_fp, grad_sample_fp=grad_sample_fp)
