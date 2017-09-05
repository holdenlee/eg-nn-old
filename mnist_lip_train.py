from mnist.mnist_model_lip import *
from lipschitz.lipschitz import *
from tf_utils.tf_utils import *
from mnist.mnist_models import *
import tensorflow as tf

from tensorflow.python.platform import app
from tensorflow.python.platform import flags


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



if __name__=='__main__':
    train_data, test_data = get_bf_mnist(normalize=True)
    x,y = make_phs()
    model, Ws, bs = make_model_from_logits_lip(mnist_model_lip)(x,y)
    opter = lambda: tf.train.GradientDescentOptimizer(learning_rate=0.1)
    evaler = Eval(test_data, batch_size, ['accuracy'], eval_feed={}, eval_steps = eval_steps, name="test")
    addons = [Train(opter, 
                    batch_size, 
                    train_feed={}, 
                    loss = 'total_loss', 
                    print_steps=print_steps),
              evaler]
    trainer = Trainer(model, max_steps, train_data, addons, ph_dict, train_dir = train_dir, verbosity=verbosity)
    trainer.init()
    #print("Weights:",trainer.sess.run([W,b]))
    trainer.train()
    #[weights, biases] = trainer.sess.run([W,b])
    #print("Weights:",weights, biases)
    #print(np.sum(weights, axis=0))
    #print(np.max(weights, axis=0))
    trainer.finish()
