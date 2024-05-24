import torch
import replay_memory
import numpy as np

def project_target(Vmin, Vmax, n_atoms, discount, memory, batch_size, target_model):

    #Sample gives me a list of named tuples, each sample is a named tuple of 5 elements
    batch = memory.sample(batch_size)

    #Unzipping the batch into its components, this creates one named tuple with all the batch elements
    #*batch unpacks the list into inidividual named tuples, *zip collects the corresponding elements of each named Tuple
    batch = replay_memory.Memory(*zip(*batch))

    rewards = torch.FloatTensor(batch.reward)
    next_state = torch.FloatTensor(np.float32(batch.next_state))
    #have to convert to float because its currently true and false
    dones = torch.FloatTensor(np.float32(batch.done))


    #The space between each discrete atom in the distribution, should be a floating point
    delta_z = float(Vmax - Vmin) / (n_atoms - 1)

    support = torch.linspace(Vmin, Vmax, n_atoms)

    #Distribution of probabilities
    probabilities = target_model(next_state)

    values = probabilities * support

    #Max(1) gets indices, sum across the atoms, take the maximum and get indices
    #Actions are second dimention, so we find the max action and it's index
    action = values.sum(2).max(1)[1]

    #Have to project action to batch x one chosen action x #atoms, gives me an action for each batch sample
    action = action.unsqueeze(1).unsqueeze(1).expand(values.size(0), 1, values.size(2))

    #Gather the distribution over atoms for the selected action at each batch element, get rid of the action dimension
    #Action dist is the atom prob distribution over returns for the best predicted actions by the target_model
    action_dist = values.gather(1, action).squeeze(1)

    #Need to add the reward signal to every single atom in the action_dist
    rewards = rewards.unsqueeze(1).expand_as(action_dist)
    dones = dones.unsqueeze(1).expand_as(action_dist)
    support = support.unsqueeze(0).expand_as(action_dist)

    #Gives us the direction we want to update towards
     #Target projected values (what we observed and want to nudge towards)
    Tz = rewards + (1 - dones) * discount * support
    #Clamp between the minimum and maximum values
    Tz = Tz.clamp(min=Vmin, max=Vmax)

    #Normalized distance of atoms of the update direction, its the distance between the Tz values
    b = (Tz - Vmin) / delta_z

    #Lowe and upper bounds of atoms
    l = b.floor().long()
    u = b.ceil().long()

    #Creating an offset, shape is batch_size x #atoms
    offset = torch.linspace(0, (batch_size - 1) * n_atoms, batch_size).long().unsqueeze(1).expand(batch_size, n_atoms)

    #Creating an empty distribution 
    proj_dist = torch.zeros(action_dist.size())

    #projecting the best action distribution into the reward distribution
    proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (action_dist * (u.float() - b)).view(-1))
    proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (action_dist * (b - l.float())).view(-1))


    return proj_dist


def project_current(memory, batch_size, curr_model, n_atoms):

    #Sample gives me a list of named tuples, each sample is a named tuple of 5 elements
    batch = memory.sample(batch_size)

    #Unzipping the batch into its components, this creates one named tuple with all the batch elements
    #*batch unpacks the list into inidividual named tuples, *zip collects the corresponding elements of each named Tuple
    batch = replay_memory.Memory(*zip(*batch))

    state = torch.FloatTensor(np.float32(batch.state))
    actions = torch.LongTensor(batch.action)

    values = curr_model(state)

    actions = actions.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, n_atoms)

    values = values.gather(1, actions).squeeze(1)

    return values




    








