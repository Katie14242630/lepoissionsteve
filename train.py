import numpy as np
from pacman_env import PacmanEnv
import pickle

env = PacmanEnv()

Q = {}
actions = [0, 1, 2, 3]
alpha = 0.1
gamma = 0.9
epsilon = 0.4  # fixed exploration rate
episodes = 1000

def get_q(state):
    if state not in Q:
        Q[state] = np.zeros(len(actions))
    return Q[state]

for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    while not done:
        step_count += 1
        if step_count > 1000:
            break

        if np.random.rand() < epsilon:
            action = np.random.choice(actions)
        else:
            action = np.argmax(get_q(state))

        next_state, reward, done = env.step(action)
        total_reward += reward

        q_values = get_q(state)
        next_q = get_q(next_state)

        q_values[action] += alpha * (reward + gamma * np.max(next_q) - q_values[action])
        state = next_state

    if episode % 50 == 0:
        print(f"Episode {episode}: Total Reward = {total_reward} | Steps = {step_count}")

# Save Q-table
with open("q_table.pkl", "wb") as f:
    pickle.dump(Q, f)
