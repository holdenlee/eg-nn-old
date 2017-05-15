import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
#from tensorflow.examples.tutorials.mnist import input_data

import os.path

import keras.backend as K

from .cifar_model import *
from .cifar_utils import *
from tf_utils.keras_utils import *
from tf_utils.make_models import *
from tf.utils.tf_utils import *
from utils.utils import *

FLAGS = flags.FLAGS

BATCH_SIZE = 128
STEPS_PER_EPOCH = 50000 // BATCH_SIZE

#NOTE: THIS NEEDS ./
flags.DEFINE_string('train_dir', './cifar10_basic/', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'model.ckpt', 'Filename to save model under.')
#flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Size of training batches.') 
#Must divide evenly into the dataset sizes. (FIX this later)')
#for batch_size = 100, is 600 times nb_epochs
#STEPS
flags.DEFINE_integer('max_steps', 100 * STEPS_PER_EPOCH, 'Number of steps to run trainer.')
flags.DEFINE_integer('print_steps', STEPS_PER_EPOCH, 'Print progress every...')
flags.DEFINE_integer('eval_steps', STEPS_PER_EPOCH, 'Run evaluation every...')
flags.DEFINE_integer('save_steps', STEPS_PER_EPOCH, 'Run evaluation every...')
flags.DEFINE_integer('summary_steps', 2 * STEPS_PER_EPOCH, 'Run summary every...')
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate for training')
flags.DEFINE_integer('verbosity', 1, 'How chatty')
flags.DEFINE_float('label_smooth', 0, 'How much to clip y values (0 for no clipping)')
flags.DEFINE_float('epsilon', 0.3, 'Strength of attack')
flags.DEFINE_string('clip', 'T', 'Whether to clip values to [0,1]')
flags.DEFINE_string('fake_data', False, 'Use fake data.  ')
flags.DEFINE_string('testing', 'F', 'Use fake data.  ')

def main(_):
    train_data, test_data = get_bf_cifar(augmentation=True, batch_size=FLAGS.batch_size)
    #FLAGS.label_smooth, FLAGS.testing
    sess = start_keras_session(1)
    model = cifar_model(dropout=True, softmax=False)
    d, ph_dict = make_with_cifar_inputs(make_model_from_logits(model, vector=False))
    #evals = [Eval(test_data, FLAGS.batch_size, ['adv_accuracy'], eval_feed={'epsilon': i*0.1}, eval_steps = FLAGS.eval_steps, name="test (adversarial %f)" % (i*0.1)) for i in range(1,6)]
    addons = [GlobalStep(),
              #TrackAverages(), #do this before train (why?)
              Train(lambda gs: tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate, decay=1e-6), FLAGS.batch_size, train_feed = {}, loss = 'loss', print_steps = FLAGS.print_steps),
                    #train_feed={'epsilon' : FLAGS.epsilon}, loss = 'combined_loss', print_steps=FLAGS.print_steps),
              # AdadeltaOptimizer(learning_rate=FLAGS.learning_rate, rho=0.95, epsilon=1e-08)
              Histograms(), #includes gradients, so has to be done after train
              Saver(save_steps = FLAGS.save_steps, checkpoint_path = FLAGS.filename), #os.path.join(FLAGS.train_dir, 
              SummaryWriter(summary_steps = FLAGS.summary_steps, feed_dict = {}), #'keep_prob': 1.0
              Eval(test_data, FLAGS.batch_size, ['accuracy'], eval_feed={}, eval_steps = FLAGS.eval_steps, name="test")] #+ evals
    trainer = Trainer(d, FLAGS.max_steps, train_data, addons, ph_dict, train_dir = FLAGS.train_dir, verbosity=FLAGS.verbosity, sess=sess)
    trainer.init_and_train()
    trainer.finish()

if __name__=='__main__':
    app.run()
