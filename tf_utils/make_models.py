import tensorflow as tf
from .tf_utils import *

def make_model_from_logits(model, vector=True):
    def m(x,y):
        #print("Model:")
        #print(model)
        predictions = model(x)
        #labels are one-hot
        loss = tf.reduce_mean((tf.nn.softmax_cross_entropy_with_logits if vector else tf.nn.sparse_softmax_cross_entropy_with_logits)(labels=y, logits=predictions), name = 'loss') 
        acc = accuracy(y, predictions, vector=vector)
        tf.add_to_collection('losses', loss)
        #added for comparing accuracies. This is batch_size * t
        #https://www.tensorflow.org/performance/xla/broadcasting
        #batch_size, t* batch_size. Must be this order to broadcast correctly
        #ind_correct = tf.cast(tf.transpose(tf.equal(tf.argmax(y,1), tf.transpose(tf.argmax(p_ind,1)))), tf.float32)
        #tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return {'loss': loss, 'inference': predictions, 'accuracy': acc}
        #, 'regularizer' : reg, 'ws': ws, 'ind_inference' : p_ind, 'ind_correct' : ind_correct}
    return m

