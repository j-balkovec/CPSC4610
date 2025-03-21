# qlearningAgents.py
# ------------------
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
from pyexpat import features

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

import logging, os

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
q_learning_agent = setup_logger('Q Learning Agent', 'QLearningAgent.log')
approx_q_agent = setup_logger('Approx Q Agent', 'QLearningAgent.log')
# --------------------- CONFIG --------------------- #


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.q_values = {}


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        # >>
        q_learning_agent.debug(f'getting Q-value for state \"{state}\", action \"{action}\"')
        # >>

        return self.q_values.get((state, action), 0.0)


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        legal_actions = self.getLegalActions(state)
        if not legal_actions:
            return 0.0

        value = max(self.getQValue(state, action) for action in legal_actions)

        # >>
        q_learning_agent.debug(f'--value-- computed value from Q-values for state \"{state}\": \"{value}\"')
        # >>

        return value

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        legal_actions = self.getLegalActions(state)
        if not legal_actions:
            return None
        best_value = self.computeValueFromQValues(state)
        best_actions = [action for action in legal_actions if self.getQValue(state, action) == best_value]

        # 0 to 100 random int
        action = best_actions[random.randint(0, 100) % len(best_actions)]

        # >>
        q_learning_agent.debug(f' --action-- computed action from Q-values for state \"{state}\": \"{action}\"')
        # >>
        return action

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None

        if not legalActions:
            return None

        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
            # >>
            q_learning_agent.debug(f'randomly selected action \"{action}\" for state \"{state}\"')
            # >>
            return action

        action = self.computeActionFromQValues(state)

        # >>
        q_learning_agent.debug(f'Selected best action \"{action}\" for state \"{state}\"')
        # >>

        return action


    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        sample = reward + self.discount * self.computeValueFromQValues(nextState)
        new_q_value = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * sample
        self.q_values[(state, action)] = new_q_value

        # >>
        q_learning_agent.debug(f'updated Q-value for state \"{state}\", action \"{action}\": \"{new_q_value}\"')
        # >>

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        features = self.featExtractor.getFeatures(state, action)
        q_val = sum(self.weights[feature] * value for feature, value in features.items())

        # >>
        approx_q_agent.debug(f"--value-- q_val: \"{q_val}\"")
        # >>

        return q_val

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        diff = (reward + self.discount * self.computeValueFromQValues(nextState) - self.getQValue(state, action))
        features = self.featExtractor.getFeatures(state, action)

        # >>
        approx_q_agent.debug(f"--update-- diff: \"{diff}\", features: \"{features}\"")
        # >>

        for feature, value in features.items():
            self.weights[feature] += self.alpha * diff * value

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # no need to print to console
            # >>
            approx_q_agent.debug(f"final weights: \"{self.weights}\"")
            # >>
