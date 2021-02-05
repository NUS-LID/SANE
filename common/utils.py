import tensorflow as tf
import numpy as np

def clip_reward(reward):
    return np.clip(reward, -1, 1)
    
def add_fc_layer(input, 
                hidden_units, 
                noise_head, 
                common_variance, 
                switch_init, 
                noisy,
                layer_wise_variance,
                single_param, 
                state_dependent,
                cond_variable, 
                activation_fn=None, 
                epsilons_given=False, 
                w_epsilon_placeholder=None, 
                b_epsilon_placeholder=None,
                name="default"):
    """
    Adds a noisy, SANE or fully connected layer to the network
    """
    if noisy :
        out = noisy_layer(input, 
                        size=hidden_units,
                        name="noisy"+name,
                        common_variance=common_variance,
                        layer_wise_variance=layer_wise_variance,
                        noise_head=noise_head, 
                        single_param=single_param,
                        state_dependent=state_dependent,
                        switch_init = switch_init,
                        epsilons_given=epsilons_given,
                        w_epsilon_placeholder=w_epsilon_placeholder, 
                        b_epsilon_placeholder=b_epsilon_placeholder,
                        cond_variable=cond_variable,
                        activation_fn=activation_fn)
    else :
        out = tf.layers.dense(inputs=input, 
                            units=hidden_units, 
                            name="actual_params" + name, 
                            activation=activation_fn)
    return out

def generate_noisy_latent_layers(input, noisy, noise_latent_size, layer_wise_variance) :
    """
    Architecture of the perturbation module
    """
    if not noisy :
        return None, None
    noise_latent = tf.layers.dense(inputs=input, 
                                    units=noise_latent_size, 
                                    kernel_initializer=tf.variance_scaling_initializer(scale=2), 
                                    name="noise_head", 
                                    activation=tf.nn.relu)
    if not layer_wise_variance :
        common_variance = tf.layers.dense(inputs = noise_latent, 
                                        units=1, 
                                        name="common_variance", 
                                        kernel_initializer=tf.variance_scaling_initializer(scale=2), 
                                        activation=None)
    else : 
        common_variance = None
    return noise_latent, common_variance

def noisy_layer(x, size, name, common_variance, layer_wise_variance, noise_head, single_param, state_dependent, switch_init, epsilons_given, w_epsilon_placeholder, b_epsilon_placeholder,cond_variable,
                bias=True, activation_fn=None, factored_noise=True):
    if state_dependent :
        if single_param : 
            ret = noisy_single_param(x, size, name, common_variance, False, noise_head, switch_init,  bias, factored_noise, epsilons_given, w_epsilon_placeholder, b_epsilon_placeholder, cond_variable)
        elif layer_wise_variance : 
            ret = noisy_single_param(x, size, name, None, layer_wise_variance, noise_head, switch_init, bias, factored_noise, epsilons_given, w_epsilon_placeholder, b_epsilon_placeholder, cond_variable)
        else :  
            ret = state_dependent_noise(x, size, name, noise_head, switch_init, bias, factored_noise, epsilons_given, w_epsilon_placeholder, b_epsilon_placeholder, cond_variable)
    elif single_param: 
        ret = noisy_networks_noise(x, size, name, switch_init, bias, factored_noise, epsilons_given, w_epsilon_placeholder, b_epsilon_placeholder, cond_variable, common_variance)
    else :
        ret = noisy_networks_noise(x, size, name, switch_init, bias, factored_noise, epsilons_given, w_epsilon_placeholder, b_epsilon_placeholder, cond_variable)

    if activation_fn is None :
        return ret
    else :
        return activation_fn(ret)

def noisy_single_param (x, size, name, common_variance, layer_wise_variance, noise_head, switch_init, 
                        bias, factored_noise, epsilons_given, w_epsilon_placeholder, b_epsilon_placeholder, cond_variable) :
    def f(x):
        return tf.multiply(tf.sign(x), tf.pow(tf.abs(x), 0.5))
    mu_init = tf.random_uniform_initializer(minval=-1*1/np.power(x.get_shape().as_list()[1], 0.5),     
                                                maxval=1*1/np.power(x.get_shape().as_list()[1], 0.5))

    if epsilons_given :
        w_epsilon, b_epsilon = gaussian_noise_vector_to_matrix(w_epsilon_placeholder, b_epsilon_placeholder, size, f)
        w_epsilon, b_epsilon = tf.tile(w_epsilon,[tf.shape(x)[0],1,1]), tf.tile(b_epsilon, [tf.shape(x)[0],1])    # repeat same noise for batch   
    else :
        if factored_noise : 
            w_epsilon, b_epsilon = factored_gaussian_noise(tf.shape(x)[0], tf.shape(x)[1], size, f)
        else :
            w_epsilon, b_epsilon = independent_gaussian_noise(tf.shape(x)[0], tf.shape(x)[1], size, f)
    with tf.variable_scope('perturbed_weights') :
        if switch_init == 1:
            w_mu = tf.get_variable(name + "/w_mu",  [1, x.get_shape()[1], size], initializer=tf.variance_scaling_initializer(scale=2))
        elif switch_init == 2:
            w_mu = tf.get_variable(name + "/w_mu",  [1, x.get_shape()[1], size])
        else :
            w_mu = tf.get_variable(name + "/w_mu",  [1, x.get_shape()[1], size], initializer=mu_init)
    if layer_wise_variance :
        log_w_sigma = tf.layers.dense(inputs = noise_head, 
                                      units=1, 
                                      name=name +"log_w_sigma", 
                                      kernel_initializer=tf.variance_scaling_initializer(scale=2), 
                                      activation=None)
    else :
        log_w_sigma = common_variance
    w_sigma = tf.reshape(log_w_sigma, [-1,1,1])
    w = w_mu + tf.cond(cond_variable[0], lambda : tf.multiply(w_sigma, w_epsilon), lambda :tf.zeros_like(w_mu))
    x = tf.expand_dims(x,1)
    ret = tf.cond(cond_variable[0], lambda : tf.reshape(tf.matmul(x, w), shape=[-1, size]), lambda : tf.matmul(x, w))
    if bias:
        with tf.variable_scope('perturbed_weights') :
            if switch_init == 1:
                b_mu = tf.get_variable(name + "/b_mu",  [1, size], initializer=tf.variance_scaling_initializer(scale=2))
            elif switch_init == 2 :
                b_mu = tf.get_variable(name + "/b_mu",  [1, size])
            else :
                b_mu = tf.get_variable(name + "/b_mu",  [1, size], initializer=mu_init)
        if layer_wise_variance :
            log_b_sigma = tf.layers.dense(inputs = noise_head, 
                                          units=1, 
                                          name=name +"log_b_sigma", 
                                          kernel_initializer=tf.variance_scaling_initializer(scale=2), 
                                          activation=None)
        else :
            log_b_sigma = common_variance
        b_sigma = log_b_sigma
        b = b_mu +  tf.cond(cond_variable[0], lambda : tf.multiply(b_sigma, b_epsilon) , lambda :tf.zeros_like(b_mu))
        ret = ret + b
    return tf.reshape(ret, [-1,size])

def state_dependent_noise(x, size, name, noise_head, switch_init, 
                          bias, factored_noise, epsilons_given, w_epsilon_placeholder, b_epsilon_placeholder, cond_variable) :
    def f(x):
        return tf.multiply(tf.sign(x), tf.pow(tf.abs(x), 0.5))
    mu_init = tf.random_uniform_initializer(minval=-1*1/np.power(x.get_shape().as_list()[1], 0.5),     
                                                maxval=1*1/np.power(x.get_shape().as_list()[1], 0.5))
    sigma_init = tf.constant_initializer(0.5/np.power(x.get_shape().as_list()[1], 0.5))

    if epsilons_given :
        w_epsilon, b_epsilon = gaussian_noise_vector_to_matrix(w_epsilon_placeholder, b_epsilon_placeholder, size, f)
        w_epsilon, b_epsilon = tf.tile(w_epsilon,[tf.shape(x)[0],1,1]), tf.tile(b_epsilon, [tf.shape(x)[0],1])    # repeat same noise for batch     
    else :
        if factored_noise : 
            w_epsilon, b_epsilon = factored_gaussian_noise(tf.shape(x)[0], tf.shape(x)[1], size, f)
        else :
            w_epsilon, b_epsilon = independent_gaussian_noise(tf.shape(x)[0], tf.shape(x)[1], size, f)

    if switch_init ==1:
        w_mu = tf.get_variable(name + "/w_mu",  [1, x.get_shape()[1], size], initializer=tf.variance_scaling_initializer(scale=2))
    elif switch_init == 2 :
        w_mu = tf.get_variable(name + "/w_mu",  [1, x.get_shape()[1], size])
    else :
        w_mu = tf.get_variable(name + "/w_mu",  [1, x.get_shape()[1], size], initializer=mu_init)

    log_w_sigma = tf.layers.dense(inputs = noise_head, 
                                      units=int(x.get_shape().as_list()[1]*size), 
                                      name=name +"log_w_sigma", 
                                      kernel_initializer=tf.variance_scaling_initializer(scale=2), 
                                      activation=None)
    
    

    w_sigma = tf.reshape(log_w_sigma, [-1, x.get_shape().as_list()[1], size])
    w = w_mu + tf.cond(cond_variable[0], lambda : tf.multiply(w_sigma, w_epsilon), lambda :tf.zeros_like(w_mu))
    x = tf.expand_dims(x,1)
    ret = tf.cond(cond_variable[0], lambda : tf.reshape(tf.matmul(x, w), shape=[-1, size]), lambda : tf.matmul(x, w))

    if bias:
        if switch_init == 1:
            b_mu = tf.get_variable(name + "/b_mu",  [1, size], initializer=tf.variance_scaling_initializer(scale=2))
        elif switch_init == 2:
            b_mu = tf.get_variable(name + "/b_mu",  [1, size])
        else :
            b_mu = tf.get_variable(name + "/b_mu",  [1, size], initializer=mu_init)

        log_b_sigma = tf.layers.dense(inputs = noise_head, 
                                      units=size, 
                                      name=name +"log_b_sigma", 
                                      kernel_initializer=tf.variance_scaling_initializer(scale=2), 
                                      activation=None)

        b_sigma = log_b_sigma
        b = b_mu + tf.cond(cond_variable[0], lambda : tf.multiply(b_sigma, b_epsilon), lambda :tf.zeros_like(b_mu))
        ret = ret + b

    return  tf.reshape(ret, [-1,size])

def noisy_networks_noise(x, size, name, switch_init, bias, factored_noise, epsilons_given, w_epsilon_placeholder, b_epsilon_placeholder, cond_variable, common_variance = None) :
    def f(x):
        return tf.multiply(tf.sign(x), tf.pow(tf.abs(x), 0.5))
    mu_init = tf.random_uniform_initializer(minval=-1*1/np.power(x.get_shape().as_list()[1], 0.5),     
                                                maxval=1*1/np.power(x.get_shape().as_list()[1], 0.5))
    sigma_init = tf.constant_initializer(0.5/np.power(x.get_shape().as_list()[1], 0.5))

    if epsilons_given :
        w_epsilon, b_epsilon = gaussian_noise_vector_to_matrix(w_epsilon_placeholder, b_epsilon_placeholder, size, f)
    else :
        if factored_noise : 
            w_epsilon, b_epsilon = factored_gaussian_noise(1, tf.shape(x)[1], size, f)
        else :
            w_epsilon, b_epsilon = independent_gaussian_noise(1, tf.shape(x)[1], size, f)

    w_epsilon, b_epsilon = tf.reshape(w_epsilon, (x.get_shape()[1],size)), tf.reshape(b_epsilon, (size,))
    if switch_init == 1:
        w_mu = tf.get_variable(name + "/w_mu",  [x.get_shape()[1], size], initializer=tf.variance_scaling_initializer(scale=2))
    elif switch_init == 2:
        w_mu = tf.get_variable(name + "/w_mu",  [x.get_shape()[1], size])
    else :
        w_mu = tf.get_variable(name + "/w_mu",  [x.get_shape()[1], size], initializer=mu_init)

    if common_variance is not None :
        log_w_sigma = common_variance
    else :
        log_w_sigma = tf.get_variable(name + "/w_sigma", [x.get_shape()[1], size], initializer=sigma_init)
    w_sigma = log_w_sigma
    w = w_mu + tf.cond(cond_variable[0], lambda : tf.multiply(w_sigma, w_epsilon), lambda :tf.zeros_like(w_mu))
    ret = tf.cond(cond_variable[0],lambda :  tf.reshape(tf.matmul(x, w), shape=[-1, size]), lambda : tf.matmul(x, w))
    if bias:
        if switch_init == 1:
            b_mu = tf.get_variable(name + "/b_mu",  [size], initializer=tf.variance_scaling_initializer(scale=2))
        elif switch_init == 2:
            b_mu = tf.get_variable(name + "/b_mu",  [size])
        else :
            b_mu = tf.get_variable(name + "/b_mu",  [size], initializer=mu_init)
        if common_variance is not None :
            log_b_sigma = common_variance
        else :
            log_b_sigma = tf.get_variable(name + "/b_sigma", [size], initializer=sigma_init)
        b_sigma = log_b_sigma
        b = b_mu + tf.cond(cond_variable[0], lambda : tf.multiply(b_sigma, b_epsilon), lambda :tf.zeros_like(b_mu))
        ret = ret + b

    return  tf.reshape(ret, [-1,size])

def sample_noise(shape):
    noise = tf.random_normal(shape)
    return noise

def sample_noise_cpu(shape):
    noise = np.random.normal(size=shape)
    return noise

def factored_gaussian_noise(batch_size, in_size, out_size, f):
    p = sample_noise([batch_size, in_size, 1])
    q = sample_noise([batch_size, 1, out_size ])
    f_p = f(p); f_q = f(q)
    w_epsilon = f_p*f_q; b_epsilon = tf.reshape(f_q, [-1, out_size])
    return w_epsilon, b_epsilon

def independent_gaussian_noise(batch_size, in_size, out_size, f) :
    weight_noise = sample_noise([batch_size, in_size, out_size])
    bias_noise = sample_noise([batch_size, out_size])
    return weight_noise, bias_noise

def factored_gaussian_noise_vectors_cpu(batch_size, in_size, out_size, f) :
    p = sample_noise_cpu([batch_size, in_size, 1])
    q = sample_noise_cpu([batch_size, 1, out_size ])
    return p, q

def gaussian_noise_vector_to_matrix(p, q, out_size, f) :
    f_p = f(p); f_q = f(q)
    w_epsilon = f_p*f_q; b_epsilon = tf.reshape(f_q, [-1, out_size])
    return w_epsilon, b_epsilon

def update_min_max_frame(all_sigmas, sigma, small_sigmas, max_sigmas_to_keep, min_sigma_frames, curr_frame, min_sigma_states, currr_state, large_sigmas, max_sigma_frames, max_sigma_states):
    all_sigmas.append(sigma)
    if len(small_sigmas) < max_sigmas_to_keep :
        if len(small_sigmas) == 0 :
            small_sigmas.append(sigma)
            min_sigma_frames.append(curr_frame)
            min_sigma_states.append(currr_state)
        else :
            index = np.searchsorted(small_sigmas, sigma)
            small_sigmas = small_sigmas[0:index] + [sigma] + small_sigmas[index:]
            min_sigma_frames = min_sigma_frames[0:index] + [curr_frame] + min_sigma_frames[index:]
            min_sigma_states = min_sigma_states[0:index] + [currr_state] + min_sigma_states[index:]
    else :
        index = np.searchsorted(small_sigmas, sigma)
        if index < max_sigmas_to_keep:
            small_sigmas =  small_sigmas[0:index] + [sigma] + small_sigmas[index:-1]
            min_sigma_frames = min_sigma_frames[0:index] + [curr_frame] + min_sigma_frames[index:-1]
            min_sigma_states = min_sigma_states[0:index] + [currr_state] + min_sigma_states[index:-1]


    if len(large_sigmas) < max_sigmas_to_keep :
        if len(large_sigmas) == 0 :
            large_sigmas.append(sigma)
            max_sigma_frames.append(curr_frame)
            max_sigma_states.append(currr_state)

        else :
            index = np.searchsorted(large_sigmas, sigma)
            large_sigmas = large_sigmas[0:index] + [sigma] + large_sigmas[index:]
            max_sigma_frames = max_sigma_frames[0:index] + [curr_frame] + max_sigma_frames[index:]
            max_sigma_states = max_sigma_states[0:index] + [currr_state] + max_sigma_states[index:]
    else :
        index = np.searchsorted(large_sigmas, sigma)
        if index < max_sigmas_to_keep and index !=0:
            large_sigmas =  large_sigmas[1:index] + [sigma] + large_sigmas[index:]
            max_sigma_frames = max_sigma_frames[1:index] + [curr_frame] + max_sigma_frames[index:]
            max_sigma_states = max_sigma_states[1:index] + [currr_state] + max_sigma_states[index:]
    return min_sigma_frames, max_sigma_frames, small_sigmas, large_sigmas, max_sigma_states, min_sigma_states
