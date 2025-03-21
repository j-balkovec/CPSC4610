# multiAgents.py
# --------------
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
from contourpy.util.data import simple

from util import manhattanDistance
from pacman import GameState
from game import Directions
import random, util
import math

from game import Agent

import logging
import os

# --------------------- CONFIG --------------------- #

# make sure the 'logs' directory exists
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)


def setup_logger(name, log_filename):
    log_file = os.path.join(log_directory, log_filename)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # file handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # prevent duplicate handlers
    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger


# separate loggers for each agent
ra_logger = setup_logger('ReflexAgent', 'ReflexAgent.log')
mma_logger = setup_logger('MinMaxAgent', 'MinMaxAgent.log')
aba_logger = setup_logger('ABAgent', 'ABAgent.log')
em_logger = setup_logger('EMAgent', 'EMAgent.log')


# --------------------- CONFIG --------------------- #

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        ra_logger.info("evaluating action: %s", action.lower())

        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # >>
        ra_logger.info("new pacman position: %s", newPos)
        ra_logger.info("new scared times: %s", newScaredTimes)
        # >>

        # consts
        rewards = {"SCARED_GHOST": 20.0}
        penalty = {"GHOST": 100.0}
        weights = {"FOOD": 10.0, "GHOST_AVOIDANCE": 5.0, "CAPSULE": 15.0}

        # get base score from the game state
        score = successorGameState.getScore()

        # >>
        ra_logger.info("base score: %s", score)
        # >>

        # ------ FOOD ------
        foodList = newFood.asList()
        if foodList is not None:
            closestFoodDist = min((manhattanDistance(newPos, food) for food in foodList),
                                  default=float('inf'))  # to fix the min() arg is an empty seq
            score += weights["FOOD"] / (closestFoodDist + 1.0)  # div by zero avoidance

            # >>
            ra_logger.info("closest food distance: %s", closestFoodDist)
            ra_logger.info("updated score (food): %s", score)
            # >>
        # ------ FOOD ------

        # ------ GHOSTS ------
        for i, ghostState in enumerate(newGhostStates):
            ghostPos = ghostState.getPosition()
            ghostDist = manhattanDistance(newPos, ghostPos)

            # >>
            ra_logger.info("ghost %s position: %s", i, ghostPos)
            ra_logger.info("ghost %s distance: %s", i, ghostDist)
            # >>

            if newScaredTimes[i] > 0:
                score += penalty["GHOST"] / (ghostDist + 1.0)  # largo penalty for being close to a ghost

                # >>
                ra_logger.info("updated score (scared ghost): %s", score)
                # >>

            else:
                # penalize
                if ghostDist < 2:
                    score -= rewards["SCARED_GHOST"] / (ghostDist + 1.0)

                else:
                    score -= weights["GHOST_AVOIDANCE"] / (ghostDist + 1.0)
        # ------ GHOSTS ------

        # ------ THE THING THAT MAKES THE GHOSTS EATABLE ------
        capsules = currentGameState.getCapsules()
        if capsules is not None:
            closestCapsuleDist = min((manhattanDistance(newPos, capsule) for capsule in capsules),
                                     default=float('inf'))  # to fix the min() arg is an empty seq
            score += weights["CAPSULE"] / (closestCapsuleDist + 1.0)

            # >>
            ra_logger.info("closest capsule distance: %s", closestCapsuleDist)
            ra_logger.info("updated score (capsule): %s", score)
            # >>

        # ------ THE THING THAT MAKES THE GHOSTS EATABLE ------

        # >>
        ra_logger.info("final evaluation score: %s", score)
        # >>

        return score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """

        def minimax(agentIndex: int, depth: int, state: GameState) -> float:
            """
            pre: agentIndex, depth, and state are valid inputs
            post: returns the evaluated value based on the minimax algorithm considering both pacman and ghosts
            desc: Implements the minimax algorithm for decision-making, handling both pacman (maximizing) and ghosts (minimizing)
            """
            # >>
            mma_logger.info("Running minimax: agentIndex=%d, depth=%d", agentIndex, depth)
            # >>

            # base case: terminal (win, lose) | depth maxed out
            if state.isWin() or state.isLose() or depth == self.depth:
                eval_value = self.evaluationFunction(state)

                # >>
                mma_logger.info("Terminal state reached: eval=%f", eval_value)
                # >>

                return eval_value

            # pacman (max out player)
            if agentIndex == 0:
                return maxValue(agentIndex, depth, state)

            # ghosts (min out player)
            else:
                return minValue(agentIndex, depth, state)

        def maxValue(agentIndex: int, depth: int, state: GameState):
            """
            pre: agentIndex, depth, and state are valid inputs
            post: returns the maximum value for pacman's move or evaluation function if no legal actions are available
            desc: Computes the maximum value for pacman at a given depth, considering possible actions
            """
            bestValue = float("-inf")
            legalActions = state.getLegalActions(agentIndex)

            # >>
            mma_logger.info("Pacman (max): depth=%d, legalActions=%s", depth, legalActions)
            # >>

            # if no legal actions -> return eval
            if not legalActions:
                eval_value = self.evaluationFunction(state)

                # >>
                mma_logger.info("No legal actions: returning eval=%f", eval_value)
                # >>

                return eval_value

            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                value = minimax(1, depth, successor)  # move to first ghost
                bestValue = max(bestValue, value)

                # >>
                mma_logger.info("Pacman considering action=%s, value=%f", action, value)
                # >>

            return bestValue

        def minValue(agentIndex: int, depth: int, state: GameState):
            """
            pre: agentIndex, depth, and state are valid inputs
            post: returns the minimum value for ghosts' move or evaluation function if no legal actions are available
            desc: Computes the minimum value for the ghost agent at a given depth, considering possible actions
            """
            bestValue = float("inf")
            legalActions = state.getLegalActions(agentIndex)

            # >>
            mma_logger.info("Ghost (min): agentIndex=%d, depth=%d, legalActions=%s", agentIndex, depth, legalActions)
            # >>

            # if no legal actions -> return eval
            if not legalActions:
                eval_value = self.evaluationFunction(state)

                # >>
                mma_logger.info("No legal actions: returning eval=%f", eval_value)
                # >>

                return eval_value

            nextAgent = agentIndex + 1
            if nextAgent >= state.getNumAgents():  # reset to pacman if no ghosts
                nextAgent = 0
                depth += 1  # increase depth after all agents have moved

            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                value = minimax(nextAgent, depth, successor)
                bestValue = min(bestValue, value)

                # >>
                mma_logger.info("Ghost considering action=%s, value=%f", action, value)
                # >>

            return bestValue

        # Root Call
        bestAction = None
        bestValue = float("-inf")

        # pacman's legal moves
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)

            # start with the first ghost
            value = minimax(1, 0, successor)

            # >>
            mma_logger.info("Root: Pacman action=%s, value=%f", action, value)
            # >>

            if value > bestValue:
                bestValue = value
                bestAction = action

        # >>
        mma_logger.info("Best action chosen: %s with value=%f", bestAction, bestValue)
        # >>

        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    # >>
    aba_logger.info("running alpha-beta agent")

    # >>

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # >>
        aba_logger.info("getting action for gameState=%s", gameState)

        # >>

        def alphaBeta(agentIndex, depth, state, alpha, beta):
            """
            pre: agentIndex, depth, and state are valid inputs, alpha and beta are initialized
            post: returns either the best value (for ghosts) or best action (for pacman)
            desc: Implements the alpha-beta pruning algorithm to optimize the minimax search
            """

            # base case: terminal (win, lose) | depth maxed out
            if state.isWin() or state.isLose() or depth == self.depth:
                result = self.evaluationFunction(state)
                # >>
                aba_logger.info("terminal state or max depth reached, evaluation value=%d", result)
                # >>
                return result

            # pacman (max out player)
            if agentIndex == 0:
                return maxValue(agentIndex, depth, state, alpha, beta)

            # ghosts (min out player)
            else:
                return minValue(agentIndex, depth, state, alpha, beta)

        def maxValue(agentIndex, depth, state, alpha, beta):
            """
            pre: agentIndex, depth, and state are valid inputs, alpha and beta are initialized
            post: returns the best action if at root depth, otherwise the best value
            desc: Calculates the maximum value for pacman's move while applying alpha-beta pruning
            """
            # >>
            aba_logger.info("entering maxValue: agentIndex=%d, depth=%d", agentIndex, depth)
            # >>

            bestValue = float("-inf")
            bestAction = None
            legalActions = state.getLegalActions(agentIndex)

            # if no legal actions -> return eval
            if not legalActions:
                result = self.evaluationFunction(state)
                # >>
                aba_logger.info("no legal actions, evaluation value=%d", result)
                # >>
                return result

            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                value = alphaBeta(1, depth, successor, alpha, beta)  # move to first ghost

                if value > bestValue:
                    bestValue, bestAction = value, action

                # AB pruning check
                alpha = max(alpha, bestValue)
                if alpha > beta:  # >= fails test case q3/6
                    # >>
                    aba_logger.info("pruning, alpha > beta: alpha=%d, beta=%d", alpha, beta)
                    # >>
                    break  # prune remaining branches

            return bestAction if depth == 0 else bestValue

        def minValue(agentIndex, depth, state, alpha, beta):
            """
            pre: agentIndex, depth, and state are valid inputs, alpha and beta are initialized
            post: returns the minimum value for ghosts or the evaluation function value if no actions are left
            desc: Calculates the minimum value for ghosts' moves while applying alpha-beta pruning
            """
            # >>
            aba_logger.info("entering minValue: agentIndex=%d, depth=%d", agentIndex, depth)
            # >>

            bestValue = float("inf")
            legalActions = state.getLegalActions(agentIndex)

            # if no legal actions -> return eval
            if not legalActions:
                result = self.evaluationFunction(state)
                # >>
                aba_logger.info("no legal actions for ghost, evaluation value=%d", result)
                # >>
                return result

            nextAgent = agentIndex + 1
            if nextAgent >= state.getNumAgents():  # reset to pacman
                nextAgent = 0
                depth += 1  # increase depth after all agents have moved

            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                value = alphaBeta(nextAgent, depth, successor, alpha, beta)

                bestValue = min(bestValue, value)

                # AB pruning check
                beta = min(beta, bestValue)
                if alpha > beta:  # >= fails test case q3/6
                    # >>
                    aba_logger.info("pruning, alpha > beta: alpha=%d, beta=%d", alpha, beta)
                    # >>
                    break  # prune remaining

            return bestValue

        # Root call
        bestAction = None
        bestValue = float("-inf")
        alpha, beta = float("-inf"), float("inf")

        # pacman's legal moves
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)

            # start with first ghost
            value = alphaBeta(1, 0, successor, alpha, beta)

            if value > bestValue:
                bestValue, bestAction = value, action

            alpha = max(alpha, bestValue)  # update alpha

        # >>
        aba_logger.info("best action for pacman: %s", bestAction)
        # >>
        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    #
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """

        # >>
        em_logger.info("getting action for gameState=%s", gameState)

        # >>

        def expectimax(state, depth, agentIndex):
            """
            pre: state is a valid game state, depth is an integer >= 0, agentIndex is a valid agent number.
            post: returns the expectimax evaluation value of the state.
            desc: recursively evaluates the game tree; Pacman (max node) picks the best value,
                  ghosts (chance nodes) compute an expected value.
            """

            # probability toggles
            # got the "wrong answer" for anything besides uniform, but it was fun playing around with it
            USE_UNIFORM = True
            USE_RANDOM_BIAS = False
            USE_GAUSSIAN = False
            USE_EXPONENTIAL = False

            # base case: terminal (win, lose) | depth maxed out
            if gameState.isWin() or gameState.isLose() or depth == self.depth * state.getNumAgents():
                result = self.evaluationFunction(state)
                # >>
                em_logger.info("terminal state or max depth reached, evaluation value=%d", result)
                # >>
                return result

            # max node > Pacman
            if agentIndex == 0:
                actions = state.getLegalActions(agentIndex)

                if not actions:
                    result = self.evaluationFunction(state)
                    # >>
                    em_logger.info("pacman has no legal actions at depth=%d, returning evaluation=%d", depth, result)
                    # >>
                    return result

                result = max(expectimax(state.generateSuccessor(agentIndex, action), depth + 1,
                                        (depth + 1) % state.getNumAgents())
                             for action in state.getLegalActions(agentIndex))

                # >>
                em_logger.info("pacman (max node) at depth=%d, bestValue=%d", depth, result)
                # >>
                return result

            # chance node > ghosts
            else:
                actions = state.getLegalActions(agentIndex)

                if not actions:
                    result = self.evaluationFunction(state)
                    # >>
                    em_logger.info("ghost has no legal actions at depth=%d, returning evaluation=%d", depth, result)
                    # >>
                    return result
            try:
                # ------ uniform probability ------
                if USE_UNIFORM:
                    probability = 1.0 / len(actions)
                # ------ uniform probability ------

                # ------ random bias ------
                elif USE_RANDOM_BIAS:
                    randomWeights = [random.uniform(0.1, 1.0) for _ in actions]
                    total = sum(randomWeights)
                    probability = randomWeights[
                                      random.randint(0, len(randomWeights) - 1)] / total  # random weight in the list
                # ------ random bias ------

                # ------ Gaussian distribution ------
                elif USE_GAUSSIAN:
                    mean = len(actions) / 2  # set mean to half the number of actions
                    std_dev = len(actions) / 4  # standard deviation
                    gaussianWeights = [math.exp(-0.5 * ((i - mean) / std_dev) ** 2) for i in range(len(actions))]
                    total = sum(gaussianWeights)
                    probability = gaussianWeights[
                                      random.randint(0, len(gaussianWeights) - 1)] / total  # random weight in the list
                # ------ Gaussian distribution ------

                # ------ exponential bias ------
                elif USE_EXPONENTIAL:
                    base = 2
                    expWeights = [base ** i for i in range(len(actions))]
                    total = sum(expWeights)
                    probability = expWeights[
                                      random.randint(0, len(expWeights) - 1)] / total  # random weight in the list
                # ------ exponential bias ------

                else:
                    raise NotImplementedError(
                        "turn probability toggle on: {uniform, gaussian, exponential, exponential bias}")

            except NotImplementedError as notImplemented:
                # >>
                em_logger.error(str(notImplemented))
                # >>
                print("<ERROR> check EMAgent.log <|> grep \"ERROR\" EMAgent.log")

            # >>
            em_logger.debug("probability: prob=%.2f", probability)
            # >>

            result = sum(expectimax(state.generateSuccessor(agentIndex, action), depth + 1,
                                    (depth + 1) % state.getNumAgents()) * probability
                         for action in actions)

            # >>
            em_logger.info("ghost (chance node) at depth=%d, expectedValue=%.2f", depth, result)
            # >>
            return result

        # find best action >>> eval all possible successor states
        legalActions = gameState.getLegalActions(0)  # all possible moves
        actionValues = {}  # store expectimax values

        for action in legalActions:
            successorState = gameState.generateSuccessor(0, action)
            actionValues[action] = expectimax(successorState, 1, 1)  # expectimax

        bestAction = max(actionValues, key=actionValues.get)  # pick the best one

        # >>
        em_logger.info("best action chosen: %s with value=%d", bestAction, actionValues[bestAction])
        # >>

        return bestAction


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:

        Notes:
            * Pacman's current score

            * Food distance and count
                - Pacman needs to eat food so we should favour states with closer and fewer food pellets

            * Ghost distance and timer
                - we need to avoid ghosts
                - we need to somehow distinguish between "dangerous" and "edible" ghosts (move away | move closer)

            * Gigantor food pellet/s
                - we should encourage eating the giant food pellets, that make the ghosts eatable

            * Movement
                - Pacman should rarely stand still

            Use manhattanDistance for simplicity

        Heuristic:
            We need to combine the factors from above into a formula

            // score & factor and interchangeable terms here
            heur(state) = baseScore + FoodSore + GhostScore + GigantorPelletScore + MobilityScore
    """

    # base score
    score = currentGameState.getScore()

    # pacman pos
    pacmanPos = currentGameState.getPacmanPosition()

    # ghost states
    ghostStates = currentGameState.getGhostStates()

    # food
    foodList = currentGameState.getFood().asList()

    # food but bigger
    giantFoodPellets = currentGameState.getCapsules() # took me a while to find this

    if foodList:
        minFoodDist = min(manhattanDistance(pacmanPos, food) for food in foodList)
        foodScore = 10.0 / (1 + minFoodDist)  # get food
    else:
        foodScore = 0

    ghostScore = 0
    for ghost in ghostStates:
        ghostPos = ghost.getPosition()
        ghostDist = manhattanDistance(pacmanPos, ghostPos)

        if ghost.scaredTimer > 0:
            ghostScore += 18.0 / (1 + ghostDist)  # eat ghosts
        else:
            if ghostDist < 3:  # heavily penalize if ghost is close & not scared
                ghostScore -= 100.0 / (1 + ghostDist)

    # encourage eating ghosts
    capsuleScore = -20 * len(giantFoodPellets)

    # open space over corners
    mobilityScore = len(currentGameState.getLegalActions()) * 1.5

    # final eval
    return score + foodScore + ghostScore + capsuleScore + mobilityScore


# Abbreviation
better = betterEvaluationFunction
