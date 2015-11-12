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
import time


class AsynchronousValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
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
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = collections.defaultdict(float)
        states = self.mdp.getStates()
        for state in states:
            self.values[state] = 0

        "*** YOUR CODE HERE ***"
        statesSize = len(states)

        for i in range(self.iterations):
            state = states[i % statesSize]

            if not self.mdp.isTerminal(state):
                A = self.computeActionFromValues(state)
                Q = self.computeQValueFromValues(state, A)
                self.values[state] = Q

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        transition = self.mdp.getTransitionStatesAndProbs(state, action)
        Q = 0.0

        for nextState in transition:
            Q += nextState[1] * (self.mdp.getReward(state) + self.discount * self.getValue(nextState[0]))

        return Q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        Q = float('-inf')
        A = 0
        actions = self.mdp.getPossibleActions(state)
        if len(actions) == 0:
            return None

        for action in actions:
            if self.computeQValueFromValues(state, action) > Q:
                Q = self.computeQValueFromValues(state, action)
                A = action
        return A

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = collections.defaultdict(float)
        states = self.mdp.getStates()
        for state in states:
            self.values[state] = 0

        "*** YOUR CODE HERE ***"

        predecessorsOfAllStates = {}

        # populate the dictionary with predecessors of each state
        for s in states:
          predecessorsOfAllStates[s] = self.getPredecessors(s)

        theQueue = util.PriorityQueue() #initialize an empty priority Queue

        for s in states:
          if not self.mdp.isTerminal(s):
            highestQ = self.highestQValue(s) #get the highest Q value resulted from all available actions from s
            diff = abs(self.values[s] - highestQ) #get the absolute value diff
            theQueue.update(s, -diff) #push s onto priority Queue with priority -diff, update its priority if needed


        for i in range(self.iterations):
          #if queue is empty, terminate
          if theQueue.isEmpty():
            return

          state = theQueue.pop()  #pop a state out
          self.values[state] = self.highestQValue(state) #update the value

          for p in list(predecessorsOfAllStates[state]):
            highestQ = self.highestQValue(p) #get the highest Q value resulted from all available actions from p
            diff = abs(self.values[p] - highestQ) #get the absolute value diff

            if diff > theta:
              theQueue.update(p, -diff) #push p into the priority queue with priority -diff


    #return the highest Q-value based on all possible actions of state
    def highestQValue(self, state):
        resultQ = float('-inf')
        actions = self.mdp.getPossibleActions(state)
        if len(actions) == 0:
            return None

        for action in actions:
            transition = self.mdp.getTransitionStatesAndProbs(state, action)
            Q = 0.0

            for nextState in transition:
              Q += nextState[1] * (self.mdp.getReward(state) + self.discount * self.getValue(nextState[0]))

            if Q > resultQ:
                resultQ = self.computeQValueFromValues(state, action)
        return resultQ


    #check if coordinates are legal
    def isAllowed(self, y, x):

        #how to access grid attributes?

        if y < 0 or y >= self.mdp.grid.height: return False
        if x < 0 or x >= self.mdp.grid.width: return False
        return self.mdp.grid[x][y] != '#'


    #return a set containing all predecessors of state
    def getPredecessors(self, state):
        predecessorSet = set()

        #how to find parents if state is terminal state?


        if not self.mdp.isTerminal(state):
          x, y = state

          northState = (self.isAllowed(y+1,x) and (x,y+1)) or state
          westState = (self.isAllowed(y,x-1) and (x-1,y)) or state
          southState = (self.isAllowed(y-1,x) and (x,y-1)) or state
          eastState = (self.isAllowed(y,x+1) and (x+1,y)) or state

          if not self.mdp.isTerminal(northState):
            if 'south' in self.mdp.getPossibleActions(northState):
              transition = self.mdp.getTransitionStatesAndProbs(northState, 'south')
              for nextState in transition:
                if (nextState[0] == state) and (nextState[1] > 0):
                  predecessorSet.add(northState)

          if not self.mdp.isTerminal(southState):
            if 'north' in self.mdp.getPossibleActions(southState):
              transition = self.mdp.getTransitionStatesAndProbs(southState, 'north')
              for nextState in transition:
                if (nextState[0] == state) and (nextState[1] > 0):
                  predecessorSet.add(southState)

          if not self.mdp.isTerminal(westState):
            if 'east' in self.mdp.getPossibleActions(westState):
              transition = self.mdp.getTransitionStatesAndProbs(westState, 'east')
              for nextState in transition:
                if (nextState[0] == state) and (nextState[1] > 0):
                  predecessorSet.add(westState)

          if not self.mdp.isTerminal(eastState):
            if 'west' in self.mdp.getPossibleActions(eastState):
              transition = self.mdp.getTransitionStatesAndProbs(eastState, 'west')
              for nextState in transition:
                if (nextState[0] == state) and (nextState[1] > 0):
                  predecessorSet.add(eastState)
       
        return predecessorSet




