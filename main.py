import argparse
import matplotlib.pyplot as plt
from TDA import *

def parse_args():
    parser = argparse.ArgumentParser(description= "2v1 Football Shootout MDP Solver")
    parser.add_argument('--p', type = float, required= True, help ="Skill level parameter p (0 <= p <= 0.5)")
    parser.add_argument('--q', type = float, required= True, help ="Skill level parameter q (0.6 <= p <= 1)")
    parser.add_argument('--opponent_pol_num', type = int, required= True, help ="Opponent policy type [1, 2, 3]")

    return parser. parse_args()


def plot_graphs():
    
    # Graph 1: Varying p
    ps = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    q = 0.7
    goals_vs_p = []
    for p in ps:
        td_agent = TD_agent(p,q,opponent_policy = 2, num_episodes= 1000, alpha = 0.1, gamma = 0.9, epsilon = 0.01)
        V = td_agent.td_zero()
        goals = td_agent.evaluate_policy(V)
        goals_vs_p.append(goals)
    
    plt.figure()
    plt.plot(ps, goals_vs_p, marker='o')
    plt.xlabel('p')
    plt.ylabel('Probability of Winning')
    plt.title('Probability of Winning vs p (q = 0.7)')
    plt.grid()
    plt.show()

    # Graph 2: Varying q
    qs = [0.6, 0.7, 0.8, 0.9, 1]
    p = 0.3
    goals_vs_q = []
    for q in qs:
        td_agent = TD_agent(p,q,opponent_policy = 2, num_episodes= 1000, alpha = 0.1, gamma = 0.9, epsilon = 0.01)
        V = td_agent.td_zero()
        goals = td_agent.evaluate_policy(V)
        goals_vs_q.append(goals)

    plt.figure()
    plt.plot(qs, goals_vs_q, marker='o')
    plt.xlabel('q')
    plt.ylabel('Probability of Winning')
    plt.title('Probability of Winning vs q (p = 0.3)')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    plot_graphs()
    #args = parse_args()
    #td_agent = TD_agent(p,q,opponent_policy = 2, num_episodes= 1000, alpha = 0.1, gamma = 0.9, epsilon = 0.001)
    #V = td_agent.td_zero()
    #goals = td_agent.evaluate_policy(V)

    #print(goals)
