# Intro

Exponentiated gradient for neural networks, implemented in tensorflow.

Acheives 98.5% on MNIST.

See also [David Parks's implementation](https://github.com/davidparks21/experimental_neural_network_matlab).

# Usage

```
opter = MWOptimizer(Ws, bs, learning_rate=lr, smoothing=smoothing)
```
Here Ws are the weight parameters (dimensions `li*n`), bs are the bias parameters (dimensions `2n`, because they include both the positive and negative bias terms).

# Notes

* U-value is the most important hyperparameter: how much to scale outputs?
* Initialize with Dirichlet distribution. 

# Future work

* Can we make EG more competitive against regular GD? Why is it not doing as well?
	* Learning rate schedule.
	* Look at evolution of weights.
* Optimize part of network with EG, part with SGD? Which part is more suitable for EG?
* Try smoothing, sleeping.
    * Try domain adaptation. Does this help solve the catastrophic forgetting problem?
* More sophisticated optimization algorithms (ex. with momentum).
* How does this do against adversarial examples?
