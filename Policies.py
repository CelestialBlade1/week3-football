import random
import numpy as np

class R_policy:
    policy_num: int
    state: tuple

    def __init__(self, policy_num, state):
        self.policy_num = policy_num
        self.state = state

    # Please note that I have only implemented the Park the Bus strategy due to lack of time
    def chosen_policy(self):
        R_updated_state = self.state
        if self.policy_num == 1:
            pass
        elif self.policy_num == 2:
            if self.state[2] == 8:
                new_R_pos = 12
                R_updated_state = (self.state[0], self.state[1], new_R_pos, self.state[3])
            elif self.state[2] == 12:
                new_R_pos = 8
                R_updated_state = (self.state[0], self.state[1], new_R_pos, self.state[3])
        return R_updated_state

class action_policy:
    state: tuple
    action_space: list
    V: dict

    def __init__(self, state, action_space, V):
        self.state = state
        self.action_space = action_space
        self.V = V

    # Function to implement the Epsilon greedy approach to choosing an action
    def Epsilon_Greedy_action(self):
        epsilon = 0.1
        if random.random() < epsilon:
            # Explore: choose a random action
            return random.choice(self.action_space)
        else:
            return self.Optimal_action()

    # Function to choose the optimal action based on the rewards for different positions
    def Optimal_action(self):
        state_moves = [-1, +1, -4, +4, -1, +1, -4, +4]
        rewards = [-10, -1, +10]
        values = []

        for a in self.action_space:
            if a < 4:
                B1_pos = self.state[0] + state_moves[a]
                new_state = (B1_pos, self.state[1], self.state[2], self.state[3])
                if 0 < B1_pos < 17:
                    x = rewards[1] + self.V.get(new_state, 0)
                else:
                    x = rewards[0] + self.V.get(self.state, 0)
                values.append(x)

            elif 4 <= a < 8:
                B2_pos = self.state[1] + state_moves[a]
                new_state = (self.state[0], B2_pos, self.state[2], self.state[3])
                if 0 < B2_pos < 17:
                    x = rewards[1] + self.V.get(new_state, 0)
                else:
                    x = rewards[0] + self.V.get(self.state, 0)
                values.append(x)

            elif a == 8:
                if self.state[3] == 1:
                    new_state = (self.state[0], self.state[1], self.state[2], 2)
                elif self.state[3] == 2:
                    new_state = (self.state[0], self.state[1], self.state[2], 1)
                x = rewards[1] + self.V.get(new_state, 0)
                values.append(x)

            elif a == 9:
                new_state = self.state
                x = rewards[2] + self.V.get(new_state, 0)
                values.append(x)

        action = np.argmax(values)
        return action


