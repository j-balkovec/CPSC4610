# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
from qtconsole.qtconsoleapp import qt_aliases

# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

import logging
import os

# --------------------- CONFIG --------------------- #

# make sure the 'logs' directory exists
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)


def setup_logger(name, log_filename):
    log_file = os.path.join(log_directory, log_filename)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # file handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # prevent duplicate handlers
    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger


# separate loggers for each agent
vi_agent_logger = setup_logger('Value Iteration Agent', 'ValueIterationAgent.log')
async_vi_agent_logger = setup_logger('AsyncValue Iteration Agent', 'AsyncValueIterationAgent.log')
ps_vi_agent_logger = setup_logger('Prioritized Sweeping Value IterationAgent', 'PrioritizedSweepingValueIterationAgent.log')
# --------------------- CONFIG --------------------- #

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.

        // Jakob Balkovec
        Bellman Equation:

            V(s) = max_a [ Σ P(s' | s, a) * (R(s, a, s') + γ * V(s')) ]
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        # >>
        vi_agent_logger.info(" --init-- initializing \"ValueIterationAgent\"")
        # >>

        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):

        # >>
        vi_agent_logger.info(" --invoke-- invoked \"runValueIteration\"")
        # >>

        for i in range(self.iterations):
            # >>
            vi_agent_logger.info(f" --iteration {i + 1} -- starting")
            # >>

            # store updated values temporarily
            new_vals = util.Counter()

            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    continue
                max_q_val = float("-inf") # some insanely large n, where n != answer for all actions

                # compute the max q-val for each action
                for action in self.mdp.getPossibleActions(state):
                    q_val = self.computeQValueFromValues(state, action)
                    max_q_val = max(max_q_val, q_val)

                    # >>
                    vi_agent_logger.debug(f" --state: {state}, action: {action}, Q-value: {q_val}")
                    # >>

                # >>
                vi_agent_logger.debug(f" --state: {state}, max Q-value: {max_q_val}")
                # >>

                # update for state
                new_vals[state] = max_q_val

            # >>
            vi_agent_logger.info(f" --iteration {i + 1} -- updated values: {new_vals}")
            # >>

            # update the values
            self.values = new_vals

            # >>
            vi_agent_logger.info(" --complete-- value iteration completed")
            # >>

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    # Jakob Balkovec, Sat Feb 22nd
    def bellman_eq(self, prob, reward, next_state):
        """
        Applies the Bellman equation to calculate the contribution of a
        state-action-next_state transition to the Q-value.

        prob: transition probability to the next state
        reward: immediate reward for the transition
        next_state: the next state reached after taking the action
        """
        result = prob * (reward + self.discount * self.values[next_state])

        vi_agent_logger.debug(f" --bellman-- \"result : {result}\")")
        return result

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        q_val = 0
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, next_state)
            q_val += self.bellman_eq(prob, reward, next_state) # neater

        vi_agent_logger.debug(f" --value-- \"result : {q_val}\")")
        return q_val

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        # initial
        best_action = None
        best_q_val = float("-inf")

        for action in self.mdp.getPossibleActions(state):
            q_val = self.computeQValueFromValues(state, action)
            if q_val > best_q_val:
                best_q_val = q_val
                best_action = action

        vi_agent_logger.debug(f" --computeQval-- \"best_q_val : {best_q_val}\")")
        vi_agent_logger.debug(f" --action-- \"best_action : {best_action}\")")

        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        """Returns the policy at the state (no exploration)."""
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        # >>
        async_vi_agent_logger.info(" --init-- initializing \"AsynchronousValueIterationAgent\"")
        # >>

        # policy dictionary
        self.policy = {}
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def bellman_eq(self, prob, reward, next_state):
        """
        Applies the Bellman equation to calculate the contribution of a
        state-action-next_state transition to the Q-value.

        prob: transition probability to the next state
        reward: immediate reward for the transition
        next_state: the next state reached after taking the action
        """
        result = prob * (reward + self.discount * self.values[next_state])

        async_vi_agent_logger.debug(f" --bellman-- \"result : {result}\")")
        return result

    def runValueIteration(self):

        # get all states in the mdp
        states = self.mdp.getStates()

        # value iteration
        for i in range(self.iterations):
            # select one to update (cyclically)
            state = states[i % len(states)] # mod to getan index

            # if terminal --> skip the update
            if self.mdp.isTerminal(state):
                continue

            best_val = float("-inf")
            best_action = None
            for action in self.mdp.getPossibleActions(state):
                val = 0
                for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    reward = self.mdp.getReward(state, action, next_state)
                    val += self.bellman_eq(prob, reward, next_state)

                # >>
                async_vi_agent_logger.debug(f"State {state}, Action {action}, Value {val}")
                # >>

                # update
                if val > best_val:
                    best_val = val
                    best_action = action

            # update the val for the state
            self.values[state] = best_val

            # guard for debugging
            if not hasattr(self, 'policy'):
                async_vi_agent_logger.debug(f"policy initialized to: \"[policy]\"")
                self.policy = {}  # Ensure that policy exists

            self.policy[state] = best_action
        # >>
        async_vi_agent_logger.debug(f"final policy: {self.policy}")
        # >>

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        # >>
        ps_vi_agent_logger.info(" --init-- initializing \"PrioritizedSweepingValueIterationAgent\"")
        # >>

        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # dictionary to store predecessors of each state
        predecessors = {}
        states = self.mdp.getStates()

        # get predecessors for each state
        for state in states:
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        if prob > 0:
                            if next_state not in predecessors:
                                predecessors[next_state] = set() # track unique
                            predecessors[next_state].add(state)

        # initialize the pqueue
        # >>
        ps_vi_agent_logger.info("--init-- initializing \"priority queue\"")
        # >>

        pq = util.PriorityQueue()
        for state in states:
            if not self.mdp.isTerminal(state):
                max_q_val  = max(
                    [self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)],
                    default=0
                )
                diff = abs(self.values[state] - max_q_val)
                pq.push(state, -diff) # debug: push neg diff to get max heap behaviour

                # >>
                ps_vi_agent_logger.debug(f"state: \"{state}\" - pushed to PQ with diff: \"{-diff}\"")
                # >>

        # value iteration

        # >>
        vi_agent_logger.info(f" --iteration-- starting")
        # >>
        for _ in range(self.iterations):
            if pq.isEmpty():
                # >>
                ps_vi_agent_logger.warning("PQ empty >> ending iteration.")
                # >>

                break
            state = pq.pop()

            # >>
            ps_vi_agent_logger.debug(f"popped state {state} from PQ")
            # >>

            # update if not terminal
            if not self.mdp.isTerminal(state):
                self.values[state] = max(
                    [self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)],
                    default=0
                )
                ps_vi_agent_logger.debug(f"state: \"{state}\" --> updated value: \"{self.values[state]}\"")

            # update predecessors
            if state in predecessors:
                for pred in predecessors[state]:
                    max_q_val = max(
                        [self.computeQValueFromValues(pred, action) for action in self.mdp.getPossibleActions(pred)],
                        default=0
                    )
                    diff = abs(self.values[pred] - max_q_val)
                    if diff > self.theta:
                        pq.update(pred, -diff)

                        # >>
                        ps_vi_agent_logger.debug(f"predecessor: \"{pred}\" --> updated PQ with diff: \"{-diff}\"")
                        # >>