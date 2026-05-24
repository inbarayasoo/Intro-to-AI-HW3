from mdp import Action, MDP, format_transition_function
from simulator import Simulator
from typing import Dict, List, Tuple
import numpy as np
import copy

def best_action(mdp, U, row, col):
    best_a = None
    best_a_val = float('-inf')

    for outer_action, inner_dict in mdp.transition_function.items():
        utility = calc_utility(mdp, U, row, col, outer_action)
        if utility > best_a_val:
            best_a = outer_action
            best_a_val = utility
    return best_a, best_a_val


def calc_utility(mdp, U, row, col, outer_action):
    utility = 0
    for i, action in enumerate(mdp.actions):
        probability = mdp.transition_function[outer_action][i]
        next_state = mdp.step((row, col), action)
        utility = utility + float(probability) * float(U[next_state[0]][next_state[1]])

    return utility


def calc_probability_matrix(mdp, policy):
    n = mdp.num_row * mdp.num_col
    probability_matrix: np.ndarray = np.zeros((n,n))
    for r in range(mdp.num_row):
        for c in range(mdp.num_col):
            if mdp.board[r][c] == "WALL" or (r, c) in mdp.terminal_states:
                continue

            state_index = r * mdp.num_col + c
            for i, action in enumerate(mdp.actions):
                policy_action = Action[policy[r][c]] if isinstance(policy[r][c], str) else policy[r][c]
                probability = mdp.transition_function[policy_action][i]
                next_state = mdp.step((r, c), action)
                next_state_index = next_state[0] * mdp.num_col + next_state[1]
                probability_matrix[state_index, next_state_index] += probability

    return probability_matrix


def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the utility for each of the MDP's state obtained at the end of the algorithms' run.
    #
    U_tag = copy.deepcopy(U_init)
    while True:
        U = copy.deepcopy(U_tag)
        l = 0
        for r in range(mdp.num_row):
            for c in range(mdp.num_col):
                if mdp.board[r][c] == "WALL":
                    U_tag[r][c] = None
                elif (r, c) in mdp.terminal_states:
                    U_tag[r][c] = mdp.board[r][c]
                else:
                    best_a, best_a_val = best_action(mdp, U, r, c)
                    U_tag[r][c] = float(mdp.board[r][c]) + mdp.gamma * best_a_val
                    if abs(U_tag[r][c] - U[r][c]) > l:
                        l = abs(U_tag[r][c] - U[r][c])
        if l < (epsilon * (1 - mdp.gamma))/mdp.gamma:
            break

    U_final = U_tag
    return U_final


def get_policy(mdp, U):
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    policy: np.ndarray = np.empty((mdp.num_row, mdp.num_col), dtype=object)

    for r in range(mdp.num_row):
        for c in range(mdp.num_col):
            if mdp.board[r][c] == "WALL":
                policy[r][c] = None
            elif (r, c) in mdp.terminal_states:
                policy[r][c] = None
            else:
                best_a, best_a_val = best_action(mdp, U, r, c)
                policy[r][c] = best_a

    return policy


def policy_evaluation(mdp, policy):
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #
    P = calc_probability_matrix(mdp, policy)

    n = mdp.num_row * mdp.num_col
    R: np.ndarray = np.zeros((n, 1))
    for r in range(mdp.num_row):
        for c in range(mdp.num_col):
            state_index = r * mdp.num_col + c
            if mdp.board[r][c] == "WALL":
                R[state_index][0] = 0
            else:
                R[state_index][0] = mdp.board[r][c]

    U_vector: np.ndarray = np.dot(np.linalg.inv(np.eye(n) - mdp.gamma * P), R)

    U_final = U_vector.reshape(mdp.num_row, mdp.num_col)

    for r in range(mdp.num_row):
        for c in range(mdp.num_col):
            if mdp.board[r][c] == "WALL":
                U_final[r][c] = None

    return U_final


def policy_iteration(mdp, policy_init):
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #
    policy_tag = copy.deepcopy(policy_init)
    U_tag = policy_evaluation(mdp, policy_tag)
    changed = True

    while changed:
        U = U_tag
        policy_tag = get_policy(mdp, U)
        U_tag = policy_evaluation(mdp, policy_tag)
        changed = False
        for r in range(mdp.num_row):
            for c in range(mdp.num_col):
                if U_tag[r][c] > U[r][c]:
                    changed = True

    optimal_policy = policy_tag
    return optimal_policy


def mc_algorithm(
        sim,
        num_episodes,
        gamma,
        num_rows=3,
        num_cols=4,
        actions=[Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT],
        policy=None,
):
    # Given a simulator, the number of episodes to run, the number of rows and columns in the MDP, the possible actions,
    # and an optional policy, run the Monte Carlo algorithm to estimate the utility of each state.
    # Return the utility of each state.

    sum_path_count: np.ndarray = np.zeros((num_rows, num_cols))
    num_first_visit: np.ndarray = np.zeros((num_rows, num_cols))

    for episode_index, episode_gen in enumerate(sim.replay(num_episodes=num_episodes)):
        path_count: np.ndarray = np.zeros((num_rows, num_cols))
        visit_flag: np.ndarray = np.zeros((num_rows, num_cols))

        stack_list_episode = []
        for step_index, step in enumerate(episode_gen):
            stack_list_episode.append(step)

        sum_reward = 0
        while stack_list_episode:
            last_step = stack_list_episode.pop()
            state, reward, action, actual_action = last_step
            sum_reward = reward + gamma * sum_reward
            path_count[state[0]][state[1]] = sum_reward
            visit_flag[state[0]][state[1]] = 1

        for r in range(num_rows):
            for c in range(num_cols):
                sum_path_count[r][c] += path_count[r][c]
                num_first_visit[r][c] += visit_flag[r][c]

    for r in range(num_rows):
        for c in range(num_cols):
            if num_first_visit[r][c] == 0:
                num_first_visit[r][c] = 1
    V = np.divide(sum_path_count, num_first_visit)

    return V
