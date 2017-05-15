from .mnist_utils import *
from .mnist_models import *
from adversarial.adv_model import *

def make_adversarial_model_from_logits_with_inputs(model, adv_f):
    x, y = make_phs()
    d, epsilon = make_adversarial_model(make_model_from_logits(model), adv_f, x, y)
    ph_dict = {'x':x, 'y':y, 'epsilon': epsilon}
    return d, ph_dict, epsilon

