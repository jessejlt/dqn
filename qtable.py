import gym
import numpy as np

env = gym.make('FrozenLake-v0')

# Init lookup table
print('Observation Space =', env.observation_space.n, ' Action Space =',
      env.action_space.n)
Q = np.zeros([env.observation_space.n, env.action_space.n])

# learning rate
lr = .85
# discount rate
y = .99
# number of iterations
epochs = 2000
# list containing total rewards and steps per epoch
rList = []

for i in range(epochs):
    # reset env and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    # Q learning
    while j < 99:
        env.render()
        j += 1
        # choose an action by greedily picking from Q table + stochastic
        # TODO what is this array comprehension syntax
        a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (
            1. / (i + 1)))
        # get new state and reward from environment
        # observation, reward, value, done, info
        sl, r, d, _ = env.step(a)
        # update Q-table with new knowledge TODO insert equation
        Q[s, a] = Q[s, a] + lr * (r + y * np.max(Q[sl, :]) - Q[s, a])
        rAll += r
        s = sl
        if d is True:  # d must equal death or end of game
            break
    rList.append(rAll)

print('Score over time=', str(sum(rList)/epochs))
print('Final Q-Table Values', Q)
