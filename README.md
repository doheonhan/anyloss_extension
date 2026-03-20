# Anyloss_extension

## |Paper| 
* [Original] AnyLoss: Transforming Classification Metrics into Loss Functions (KDD24, https://arxiv.org/abs/2405.14745) 
* [Extension] Generation of Loss Functions from Matrix-Based Binary Classification Metrics (TKDD, Not Published Yet)

## |How to use|
### 1. The code for the overview is shown below.
* The code in detail: AnyLoss_Code.ipynb
* For Quick Use (Python Code with Tensorflow): Define AnyLoss using "def" and put the loss name in the "[   ]" when compiling. (See Below)
  
```
s = 1e-5
import tensorflow as tf
################################ WBCE ################################
def WBCE(y_true, y_pred):
    N = batch    # batch_size
    y1 = tf.reduce_sum(y_true)
    y0 = N-y1
    w1 = y0/N #N/y1
    w0 = y1/N #N/y0
    return -tf.reduce_mean(w1*y_true*tf.math.log(y_pred+s)+w0*(1-y_true)*tf.math.log(1-y_pred+s))

################################ TN/FP/FN/TP ################################
def confusion_matrix(y_true, y_pred):
    N = batch    # batch_size
    y1 = tf.reduce_sum(y_true)
    y0 = N-y1
    TN = N-tf.reduce_sum(y_true)-tf.reduce_sum(y_pred)+tf.reduce_sum(y_true*y_pred)
    FP = tf.reduce_sum(y_pred)-tf.reduce_sum(y_true*y_pred)
    FN = tf.reduce_sum(y_true)-tf.reduce_sum(y_true*y_pred)
    TP = tf.reduce_sum(y_true*y_pred)
    return N, y1, y0, TN, FP, FN, TP

################################ Any_Fbeta ################################
def Any_Fbeta(y_true, y_pred):
    b = 1 
    y_pred = 1/(1+tf.math.exp(-L*(y_pred-0.5)))
    N, y1, y0, TN, FP, FN, TP = confusion_matrix(y_true, y_pred)
    F_beta = ((1+b**2)*TP) / ((b**2)*y1 + tf.reduce_sum(y_pred)+s)  # (1+b**2)TP/((1+b**2)TP+FP+b**2FN)
    return 1-F_beta

################################ WBCEFL ################################
def WBCEFL(y_true, y_pred):
    b = 1
    WBCEloss = WBCE(y_true, y_pred)
    N, y1, y0, TN, FP, FN, TP = confusion_matrix(y_true, y_pred)
    F_beta = ((1+b**2)*TP) / ((b**2)*y1 + tf.reduce_sum(y_pred)+s)  # (1+b**2)TP/((1+b**2)TP+FP+b**2FN)
    return (1-r)*WBCEloss+(r)*(1-F_beta)

################################ Any_Gmean ################################
def Any_Gmean(y_true, y_pred):
    y_pred = 1/(1+tf.math.exp(-L*(y_pred-0.5)))
    N, y1, y0, TN, FP, FN, TP = confusion_matrix(y_true, y_pred)
    sur_gmean = (TP*TN)/(y1*y0+s)
    return 1-sur_gmean

################################ WBCEGL ################################
def WBCEGL(y_true, y_pred):
    WBCEloss = WBCE(y_true, y_pred)
    N, y1, y0, TN, FP, FN, TP = confusion_matrix(y_true, y_pred)
    sur_gmean = (TP*TN)/(y1*y0+s)
    return (1-r)*WBCEloss+(r)*(1-sur_gmean)

################################ Any_BAccu ################################
def Any_BAccu(y_true, y_pred):
    y_pred = 1/(1+tf.math.exp(-L*(y_pred-0.5)))
    N, y1, y0, TN, FP, FN, TP = confusion_matrix(y_true, y_pred)
    baccu = (y0*TP+y1*TN) / (2*y1*y0+s)
    return 1-baccu

################################ WBCEBL ################################
def WBCEBL(y_true, y_pred):
    WBCEloss = WBCE(y_true, y_pred)
    N, y1, y0, TN, FP, FN, TP = confusion_matrix(y_true, y_pred)
    baccu = (y0*TP+y1*TN) / (2*y1*y0+s)
    return (1-r)*WBCEloss+(r)*(1-baccu)

L = Hyperparameter, positive real number, [2, 70], an integer for convenience
r = Hyperparameter, positive real number, (0, 1)
model.compile(loss=[  ], optimizer=opt, metrics=['accuracy']) # [  ]: Any_Fbeta // WBCEFL // ...
```

### 2. Description of Files
* Refer to the file title (indicating section of the paper)

### 3. Misc.
* Result folder: Results from experiments (csv)
* Bayesian Sign Test: bayesiantests.py
