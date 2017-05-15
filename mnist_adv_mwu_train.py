import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
#from tensorflow.examples.tutorials.mnist import input_data

from cleverhans.utils_mnist import data_mnist
#from cleverhans.utils_tf import model_train, model_eval, batch_eval
from cleverhans.attacks import fgsm
from cleverhans.utils import cnn_model

import os.path
from random import random

import keras.backend as K

from mwu.mwu import *
#from adversarial.mwu_adv import *
from mnist.mnist_utils import *
from mnist.mnist_mwu_model import *
from adversarial.adv_model import *
from tf_utils.tf_utils import *
from tf_utils.make_models import *

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', 'results/mnist_adv_mwu/', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'model.ckpt', 'Filename to save model under.')
#flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 100, 'Size of training batches. Must divide evenly into the dataset sizes. (FIX this later)')
#for batch_size = 100, is 600 times nb_epochs
#STEPS
flags.DEFINE_integer('max_steps', 3600, 'Number of steps to run trainer.')
flags.DEFINE_integer('print_steps', 100, 'Print progress every...')
flags.DEFINE_integer('eval_steps', 600, 'Run evaluation every...')
flags.DEFINE_integer('save_steps', 1200, 'Run evaluation every...')
flags.DEFINE_integer('summary_steps', 1200, 'Run summary every...')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
flags.DEFINE_integer('verbosity', 1, 'How chatty')
flags.DEFINE_float('label_smooth', 0, 'How much to clip y values (0 for no clipping)')
flags.DEFINE_float('epsilon', 0.3, 'Strength of attack')
flags.DEFINE_string('clip', 'T', 'Whether to clip values to [0,1]')
flags.DEFINE_string('fake_data', False, 'Use fake data.  ')

flags.DEFINE_integer('u', 20, 'u')
flags.DEFINE_integer('u2', 20, 'u2')


def train_mwu(f, adv_f, label_smooth = 0, batch_size = 100, eval_steps = 600, learning_rate=0.1, epsilon=0.3, print_steps=100, save_steps =1200, train_dir = 'train_single/', filename = 'model.ckpt', summary_steps=1200, max_steps=3600, verbosity=1):
    train_data, test_data = get_bf_mnist(label_smooth=label_smooth)
    sess = tf.Session()
    x, y = make_phs()
    adv_model, epvar = make_adversarial_model(make_model_from_logits(f), adv_f, x, y)
    ph_dict = {'x':x, 'y':y, 'epsilon': epvar}
    #return d, ph_dict, epsilon
    #adv_model, ph_dict, epvar = make_adversarial_model(f, adv_f)
    evals = [Eval(test_data, batch_size, ['adv_accuracy'], eval_feed={'epsilon': i*0.1}, eval_steps = eval_steps, name="test (adversarial %f)" % (i*0.1)) for i in range(1,6)]
    Ws = tf.get_collection('Ws')
    bs = tf.get_collection('bs')
    #print("collections:"+str((Ws,bs)))
    opter = lambda: MWOptimizer(Ws, bs, learning_rate=learning_rate, smoothing=0)
    addons = [Train(opter, 
                    batch_size, 
                    train_feed={'epsilon' : epsilon},
                    loss = 'combined_loss', 
                    print_steps=print_steps),
              Saver(save_steps = save_steps, checkpoint_path = filename),
                #SummaryWriter(summary_steps = summary_steps, feed_dict = {}), #'keep_prob': 1.0
              Eval(test_data, batch_size, ['accuracy'], eval_feed={}, eval_steps = eval_steps, name="test (real)")] + evals
    trainer = Trainer(adv_model, max_steps, train_data, addons, ph_dict, train_dir = train_dir, verbosity=verbosity, sess=sess)
    trainer.init_and_train()
    trainer.finish()

def main(_):
    train_mwu(mnist_mwu_model(u=FLAGS.u,u2=FLAGS.u2), fgsm, label_smooth = FLAGS.label_smooth, batch_size = FLAGS.batch_size, eval_steps = FLAGS.eval_steps, learning_rate=FLAGS.learning_rate, epsilon=FLAGS.epsilon, print_steps=FLAGS.print_steps, save_steps =FLAGS.save_steps, train_dir = FLAGS.train_dir, filename = FLAGS.filename, summary_steps = FLAGS.summary_steps, max_steps = FLAGS.max_steps, verbosity = FLAGS.verbosity)
 #             'train_adv_mwu_50/', filename = 'model.ckpt', summary_steps=1200, max_steps=3600, verbosity=1)

if __name__=='__main__':
    app.run()
