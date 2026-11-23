import numpy as np

def get_next_action(model, observation, num_observations, hidden_size, num_actions):
    model = np.array(model)
    
    end_i2h = num_observations * hidden_size
    end_h2o = end_i2h + hidden_size * num_actions
    end_bias1 = end_h2o + hidden_size

    i2h = model[:end_i2h].reshape(num_observations, hidden_size)
    h2o = model[end_i2h:end_h2o].reshape(hidden_size, num_actions)

    bias1 = model[end_h2o:end_bias1].reshape(1, hidden_size)
    bias2 = model[end_bias1:].reshape(num_actions)

    activations = np.tanh(observation @ i2h + bias1)
    outputs = activations @ h2o + bias2

    out = np.argmax(outputs)
    
    return out