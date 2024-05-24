from project_target import project_target, project_current
import torch
from torch.nn.functional import kl_div, softmax, log_softmax

def loss(Vmin, Vmax, n_atoms, discount, memory, batch_size, target_model, curr_model):
    '''Loss of Categorical DQN is KL Divwergence between target and current distributions'''


    target_dist = project_target(Vmin, Vmax, n_atoms, discount, memory, batch_size, target_model)

    current_dist = project_current(memory, batch_size, curr_model, n_atoms)
    current_dist.data.clamp_(0.01, 0.99)

    loss = -(target_dist * current_dist.log()).sum(1).mean()

    return loss