from Transition import *
from Policies import *

class TD_agent:

    p_value: float
    q_value: float
    opponent_policy: int
    num_episodes : int
    alpha : float
    gamma : float
    epsilon : float

    # Initializing all the required variables
    def __init__(self, p_value, q_value, opponent_policy, num_episodes, alpha, gamma, epsilon):
        self.p = p_value
        self.q = q_value
        self.policy = opponent_policy
        self.num_episodes = num_episodes
        self.state_space = self.generate_states()
        self.action_space = list(range(10))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def generate_states(self):
        states = []
        for b1 in range(1,17):
            for b2 in range(1,17):
                for r in range(1,17):
                    for posession in [1,2]:
                        states.append((b1,b2,r,posession))
        return states
    
    # The terminal state is being decided on the basis of the reward returned
    def check_terminal_state(self,reward):

        if reward == -1 :
            flag = False
        elif reward == -10 or reward == 10:
            flag = True
        return flag
    
    # Function to print the current state for debugging the code
    def print_current_state(self, state):

        print(f"B1: {state[0]}\n")
        print(f"B2: {state[1]}\n")
        print(f"R : {state[2]}\n")
        print(f"Possession: {state[3]}\n")
    
    # Implementation of the TD(0) algorithm
    def td_zero(self):
        # Creating the value function for the state space
        V = {state: 0 for state in self.state_space}
        i= 0
        # An outer while loop to check for convergence of the Value function
        while True:

            # Running multiple episodes to obtain the appropriate Value function
            for episode in range(self.num_episodes):
                state = (5, 9, 8, 1)  # Starting state
                reward = -1
                R_pos_initial = state[2] # Record of the initial position of R

                # Choosing of the subsequent position of R and the action.
                R_updated_state = R_policy(self.policy, state).chosen_policy() 
                action = action_policy(state,self.action_space, V).Epsilon_Greedy_action()

                max_delta = 0  # Track the maximum delta within the episode

                while not self.check_terminal_state(reward):
                    transition_agent = Transition(R_updated_state, action, self.p, self.q, R_pos_initial)
                    # Transition function determines the next state and the appropriate reward
                    next_state, reward = transition_agent.get_transition()

                    delta = reward + self.gamma * V[next_state] - V[state] # TD error
                    V[state] += self.alpha * delta

                    # Update the maximum delta
                    max_delta = max(max_delta, abs(delta))

                    # Update for the next iteration
                    state = next_state
                    #self.print_current_state(state)
                    R_pos_initial = state[2]
                    R_updated_state = R_policy(self.policy, state).chosen_policy()
                    action = action_policy(state,self.action_space, V).Epsilon_Greedy_action()

                    

                #Check for convergence after each episode
                if max_delta < self.epsilon:
                   break
            
            # Check for convergence after completion of all episodes
            state = (5, 9, 8, 1)
            R_pos_initial = state[2]
            R_updated_state = R_policy(self.policy, state).chosen_policy()
            action = action_policy(state,self.action_space, V).Epsilon_Greedy_action()
            transition_agent = Transition(R_updated_state, action, self.p, self.q, R_pos_initial)      
            next_state, reward = transition_agent.get_transition()

            delta = reward + self.gamma * V[next_state] - V[state]

            if delta < self.epsilon :
                print(f"We have achieved the optimal Value function in the {i+1} iteration")
                break
            else:
                i+=1
                print(f"Completed {i} iteration")

        return V
    
    # Function to evaluate policy
    def evaluate_policy(self, V, num_episodes=100):
        print("Evaluating policy")
        total_goals = 0
        for i in range(num_episodes):
            state = (5, 9, 8, 1)  # Starting state
            reward = -1 # Initializing reward to enter the loop
            R_pos_initial = state[2]
            R_updated_state = R_policy(self.policy, state).chosen_policy()
            action = action_policy(state,self.action_space, V).Optimal_action()

            while not self.check_terminal_state(reward):
                transition_agent = Transition(R_updated_state, action, self.p, self.q, R_pos_initial)
                next_state, reward = transition_agent.get_transition()

                # Update for the next iteration
                state = next_state
                #self.print_current_state(state)
                R_pos_initial = state[2]
                R_updated_state = R_policy(self.policy, state).chosen_policy()
                action = action_policy(state,self.action_space, V).Optimal_action()

                # Checking whether a goal has been scored based on the reward
                if self.check_terminal_state(reward) and reward > 0:
                    total_goals += 1
                    break
                
            

        return total_goals / num_episodes