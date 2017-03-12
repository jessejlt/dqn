import numpy as np
import tensorflow as tf
import gym

env = gym.make('CartPole-v0')
env.reset()

# Hyperparameters

# Number of hidden layer neurons
hidden_layer_neurons = 10
# Number of epochs before parameter update
batch_size = 5
# Step size for gradients descent
learning_rate = 1e-2
# Discount factor reward
discount_factor = 0.99
# Input dimensions - left, right, left-angle, right-angle
input_dimensions = 4

# A note on Tensorflow placeholders vs variables.
# Variables are for training weights and biases.
# Placeholders are where data is fed during training.
# So placeholder values are only known at runtime.

tf.reset_default_graph()
# Definition of the network as it goes from taking an observation of the
# environment to giving a probability of chosing an action.
observations = tf.placeholder(
    tf.float32, [None, input_dimensions], name="input_x")
W1 = tf.get_variable(
    "W1",
    shape=[input_dimensions, hidden_layer_neurons],
    initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations, W1))
W2 = tf.get_variable(
    "W2",
    shape=[hidden_layer_neurons, 1],
    initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1, W2)
probability = tf.nn.sigmoid(score)  # clamp to [-1,1]

# Now our learning policy
tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal")

# Our loss function
loglik = tf.log(input_y * (input_y - probability) + (1 - input_y) * (
    input_y + probability))
loss = -tf.reduce_mean(loglik * advantages)
new_grads = tf.gradients(loss, tvars)

# Apply our gradients
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
W1Grad = tf.placeholder(tf.float32, name="batch_grad1")
W2Grad = tf.placeholder(tf.float32, name="batch_grad2")
batch_grad = [W1Grad, W2Grad]
update_grads = adam.apply_gradients(zip(batch_grad, tvars))


def discount_rewards(r):
    """
    Actions taken near the end of the epoch are weighed negatively
    as they are more likely to have contributed to the failure,
    meaning that the pole failed to stay aloft.
    While this isn't always true, it's a reasonable place to start.
    """
    discounted_r = np.zeros_like(r)  # returns array shaped like r
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * discount_factor + r[t]
        discounted_r[t] = running_add
    return discounted_r


# Our agent
xs, hs, dlogps, drs, ys, tfps = [], [], [], [], [], []
running_reward = None
reward_sum = 0
epoch = 1
epochs = 10000
init = tf.global_variables_initializer()

with tf.Session() as sess:
    rendering = False
    sess.run(init)
    observation = env.reset()

    # reset the gradient placeholder. Gradients will be collected
    # in grad_buffer until we're ready to update the policy network.
    grad_buffer = sess.run(tvars)
    for ix, grad in enumerate(grad_buffer):
        grad_buffer[ix] = grad * 0

    while epoch <= epochs:

        # Rendering is slow. Let's not do so until the network
        # has learned fairly well
        if reward_sum / batch_size > 100 or rendering is True:
            env.render()
            rendering = True

        # Ensure our observation is in the correct shape.
        x = np.reshape(observation, [1, input_dimensions])

        # Run the policy network and select an action
        tf_prob = sess.run(probability, feed_dict={observations: x})
        action = 1 if np.random.uniform() < tf_prob else 0

        xs.append(x)  # observation
        y = 1 if action == 0 else 0  # fake label
        ys.append(y)

        # update teh environment to get new measurements
        observation, reward, done, info = env.step(action)
        reward_sum += reward
        drs.append(reward)

        if done:
            epoch += 1

            # stack all inputs, hidden states, action gradients, and rewards
            # for this epoch
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            tfp = tfps
            xs, hs, dlogps, drs, ys, tfps = [], [], [], [], [], []

            # compute discounted reward in reverse cronological order
            discounted_epr = discount_rewards(epr)

            # Size the rewards to be unit normal to reduce gradient
            # estimator variance
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            t_grad = sess.run(
                new_grads,
                feed_dict={
                    observations: epx,
                    input_y: epy,
                    advantages: discounted_epr
                })
            for ix, grad in enumerate(t_grad):
                grad_buffer[ix] += grad

            # Update policy with gradients
            if epoch % batch_size == 0:
                sess.run(
                    update_grads,
                    feed_dict={W1Grad: grad_buffer[0],
                               W2Grad: grad_buffer[1]})
                for ix, grad in enumerate(grad_buffer):
                    grad_buffer[ix] = grad * 0

                # Provide summary
                running_reward = reward_sum if running_reward is None else running_reward * 0.9 + reward_sum * 0.01
                print('Average reward for episode ', reward_sum / batch_size,
                      '. Total average reward ', running_reward / batch_size)

                if reward_sum / batch_size > 200:
                    print('Task solved in ', epoch)
                    break

                reward_sum = 0

            observation = env.reset()

print('Epochs complete ', epoch)
