import tensorflow as tf
from tf_utils.tf_utils import *
from tf_utils.tf_vars import *

LAMBDA = 0.1

def rand_orth(m,n):
    a = np.random.normal(0.0, 1.0, (m,n))
    u, _, v = np.linalg.svd(a, full_matrices=False)
    if m<=n:
        return v
    else:
        return u

def lip_conv_layer(x, f2, dims=None, la=LAMBDA, name='lip', strides = (1,1), f = tf.nn.relu, max_sv=2):
    x = tf.cast(x, dtype=tf.complex64)
    #NOTE: x should be (batch_size * filters * w * h) rather than the usual (batch_size * w * h * filters)
    [f1,w,h] = map(int, x.get_shape()[1:])
    # x : b * f * w * h
    # do FFT
    xhat = tf.fft2d(x)
    # xhat : b * f * w * h
    # reshape
    xc = tf.reshape(xhat, [-1, f1, 1, w*h])
    # xc : b * f * 1 * wh
    xt = tf.transpose(xc, perm=[0, 3, 2, 1])
    # xt : b * wh * 1 * f
    d = min(f1, f2)
    # ? Is this along rows or columns?
    # we want a stack of these!
    U_init = np.asarray([rand_orth(f1,d) for i in range(w*h)], dtype=np.complex64)
    U = get_scope_variable('U', scope_name=name, initializer = U_init, dtype=tf.complex64) # , shape=[w*h, f1, d]
    diag = get_scope_variable('d', scope_name=name, shape=[w*h, d], initializer = tf.ones_initializer(dtype=tf.complex64), dtype=tf.complex64)
    D = tf.map_fn(tf.diag, diag)
    V_init = np.asarray([rand_orth(d,f2) for i in range(w*h)], dtype=np.complex64)
    V = get_scope_variable('V', scope_name=name, initializer = V_init, dtype=tf.complex64) # , shape=[w*h, d, f2]
    Ur = tf.reshape(U, [1,w*h,f1,d])
    Dr = tf.reshape(D, [1,w*h,d,d])
    Vr = tf.reshape(V, [1,w*h,d,f2])
    print("U,D,V", Ur, Dr, Vr)
    yt = tf.matmul(tf.matmul(tf.matmul(xt, Ur), Dr), Vr)
    # yt : b * wh * 1 * f2
    yt2 = tf.reshape(yt, [-1, w, h, f2])
    # yt2 : b * w * h * f2
    yhat = tf.transpose(yt2, [0, 3, 1, 2])
    # yhat : b * f2 * w * h
    y = tf.real(tf.ifft2d(yhat))
    # y : b * f2 * w * h
    y = f(y)
    y = tf.layers.average_pooling2d(y, strides, strides)
    orth_loss = la * (tf.nn.l2_loss(tf.matmul(V, tf.conj(tf.transpose(V, [0,2,1]))) - tf.eye(d, dtype=tf.complex64)) + 
                      tf.nn.l2_loss(tf.matmul(tf.conj(tf.transpose(U, [0,2,1])), U) - tf.eye(d, dtype=tf.complex64)))
    d_loss = tf.reduce_sum(tf.nn.relu(tf.real(diag) - max_sv))
    tf.add_to_collection('losses', d_loss)
    tf.add_to_collection('losses', orth_loss)
    # also need loss for locality.
    return y

def lip_linear_layer(x, f2, la=LAMBDA, name='lip', f = lambda x: x, max_sv=2):
    x = tf.cast(x, tf.complex64)
    f1 = int(x.get_shape()[1])
    # x : b * f
    d = min(f1, f2)
    # ? Is this along rows or columns?
    # we want a stack of these!
    U_init = np.ndarray.astype(rand_orth(f1,d), np.complex64)
    U = get_scope_variable('U', scope_name=name, initializer = U_init, dtype=tf.complex64) # , shape=[f1, d]
    diag = get_scope_variable('d', scope_name=name, shape=[d], initializer = tf.ones_initializer(dtype=tf.complex64), dtype=tf.complex64)
    D = tf.diag(diag)
    V_init = np.ndarray.astype(rand_orth(d, f2), np.complex64)
    V = get_scope_variable('V', scope_name=name, initializer = V_init, dtype=tf.complex64) # , shape=[d, f2]
    #Ur = tf.reshape(U, [f1,d])
    #Dr = tf.reshape(D, [d,d])
    #Vr = tf.reshape(V, [d,f2])
    yt = tf.real(tf.matmul(tf.matmul(tf.matmul(x, U), D), V))
    # yt : b * f2
    y = f(yt)
    d_loss = tf.reduce_sum(tf.nn.relu(tf.real(diag) - max_sv))
    orth_loss = la * (tf.nn.l2_loss(tf.matmul(V, tf.conj(tf.transpose(V,[0,2,1]))) - tf.eye(d, dtypetf.complex64)) + 
                      tf.nn.l2_loss(tf.matmul(tf.conj(tf.transpose(U, [0,2,1])), U) - tf.eye(d, dtype=tf.complex64)))
    tf.add_to_collection('losses', orth_loss)
    return y
                  

def make_model_from_logits_lip(model):
    def m(x,y):
        predictions = model(x)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predictions), name = 'loss')
        acc = accuracy(y, predictions, vector=True)
        tf.add_to_collection('losses', loss)
        total_loss = sum(tf.get_collection('losses'))
        return {'loss': loss, 'total_loss': total_loss, 'inference': predictions, 'accuracy': acc}
        #, 'regularizer' : reg, 'ws': ws, 'ind_inference' : p_ind, 'ind_correct' : ind_correct}
    return m
