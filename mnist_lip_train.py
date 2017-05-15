from mnist.mnist_model_lip import *
from lipschitz.lipschitz import *
from tf_utils.tf_utils import *
from mnist.mnist_models import *
import tensorflow as tf

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
