# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    # create the stack
    stack = util.Stack()
    # insert the starting state
    node = Node(problem.getStartState())
    stack.push(node)
    while not stack.isEmpty():
        node = stack.pop()
        # print(node.current_state)
        if problem.isGoalState(node.current_state):
            # print(node.get_actions())
            return node.get_actions()
        for succ in problem.getSuccessors(node.current_state):
            # succ returns a triple (next state, action and cost)
            # print("starting successors")
            new_state, new_action, new_cost = succ
            if new_state not in node.get_states():
                new_node = add(node,new_state, new_action)
                # print("start state : {0}".format(new_node.starting_state))
                # print("new action path:{0}".format(new_node.action))
                stack.push(new_node)
            # else:
                # print("found a cycle!")
    return None




def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # create the stack
    queue = util.Queue()
    # insert the starting state
    node = Node(problem.getStartState())
    queue.push(node)
    all_states = [node.current_state]
    while not queue.isEmpty():
        node = queue.pop()
        # print("current state is {}".format(node.current_state))
        if problem.isGoalState(node.current_state):
            return node.get_actions()
        for succ in problem.getSuccessors(node.current_state):
            # succ returns a triple (next state, action and cost)
            new_state, new_action, new_cost = succ
            if new_state not in all_states:
                new_node = add(node, new_state, new_action)
                all_states.append(new_state)
                queue.push(new_node)
            # else:
            #     print("found a cycle!")
    return None

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # queue = util.Queue()
    # node = Node(problem.getStartState())
    # queue.push(node)
    # seen = {node.current_state: 0}
    # while not queue.isEmpty():
    #     n = queue.pop()
    #     if problem.getCostOfActions(n.get_actions()) <= seen[n.current_state]:
    #         if problem.isGoalState(n.current_state):
    #             return n.get_actions()
    #         for succ in problem.getSuccessors(n.current_state):
    #             new_state, new_action, new_cost = succ
    #             new_node = add(n, new_state, new_action)
    #             if not new_state in seen or problem.getCostOfActions(new_node.get_actions()) < seen[new_state]:
    #                 queue.push(new_node)
    #                 seen[new_state] = problem.getCostOfActions(new_node.get_actions())
    # return None

    # use a priority queue

    queue = util.PriorityQueue()
    node = Node(problem.getStartState())
    node.add_cost(0)
    queue.push(node, 0)
    all_states = {node.current_state:0}
    while not queue.isEmpty():
        node = queue.pop()
        if node.cost <= all_states[node.current_state]:
            if problem.isGoalState(node.current_state):
                return node.get_actions()
        # find the path
        # remove path from list
        # deal with successors
        # if succesor already in this root path dont add
        #  else add their path to list, add to queue.
        for succ in problem.getSuccessors(node.current_state):
                # succ returns a triple (next state, action and cost)
                # print("starting successors")
            new_state, new_action, new_cost = succ
            new_node = add(node, new_state, new_action)
            cost = problem.getCostOfActions(new_node.get_actions())
            new_node.add_cost(cost)
            if (new_state not in all_states) or (cost < all_states[new_state]):
                all_states[new_state] = cost

            # print("start state : {0}".format(new_node.starting_state))
            # print("new action path:{0}".format(new_node.action))
                queue.push(new_node, cost)
                # else:
                # print("found a cycle!")
    return None

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    f_1 = lambda state: heuristic(state, problem=problem)
    queue = util.PriorityQueue()
    node = Node(problem.getStartState())
    node.add_cost(0+f_1(node.current_state))
    queue.push(node, node.cost)
    all_states = {node.current_state: node.cost}
    while not queue.isEmpty():
        node = queue.pop()
        if node.cost <= all_states[node.current_state]:
            if problem.isGoalState(node.current_state):
                return node.get_actions()
        # find the path
        # remove path from list
        # deal with successors
        # if succesor already in this root path dont add
        #  else add their path to list, add to queue.
        for succ in problem.getSuccessors(node.current_state):
            # succ returns a triple (next state, action and cost)
            # print("starting successors")
            new_state, new_action, new_cost = succ
            new_node = add(node, new_state, new_action)
            cost = problem.getCostOfActions(new_node.get_actions()) + f_1(new_node.current_state)
            new_node.add_cost(cost)
            if (new_state not in all_states) or (cost < all_states[new_state]):
                all_states[new_state] = cost

                # print("start state : {0}".format(new_node.starting_state))
                # print("new action path:{0}".format(new_node.action))
                queue.push(new_node, cost)
                # else:
                # print("found a cycle!")
    return None
    # f_1 = lambda state: heuristic(state, problem=problem)
    #
    # # f_n = lambda state, prob, func1, func2: func1(state) + func2(state, problem=prob)
    # queue = util.PriorityQueueWithFunction(f_1)
    # # insert the starting state
    # node = Node(problem.getStartState())
    # queue.push(node)
    # all_states = [node.current_state]
    # while not queue.isEmpty():
    #     node = queue.pop()
    #     # print(node.current_state)
    #     if problem.isGoalState(node.current_state):
    #         # print(node.get_actions())
    #         return node.get_actions()
    #     for succ in problem.getSuccessors(node.current_state):
    #         # succ returns a triple (next state, action and cost)
    #         # print("starting successors")
    #         new_state, new_action, new_cost = succ
    #         if new_state not in all_states:
    #             new_node = add(node, new_state, new_action)
    #             # print("start state : {0}".format(new_node.starting_state))
    #             # print("new action path:{0}".format(new_node.action))
    #             all_states.append(new_state)
    #             queue.push(new_node)
    #         # else:
    #         #     print("found a cycle!")
    # return None


class Node:
    """a node that will state the current state and the path it leads to"""
    def __init__(self, state):
        # the current state
        self.starting_state = state
        self.all_states = [state]
        self.current_state = state
        self.action = []
        self.cost = 0

    def get_actions(self):
        return self.action
    def add_cost(self, cost):
        self.cost = cost
    def get_states(self):
        return self.all_states
    def __getitem__(self, item):
        return self.current_state[item]

def add(node, state, action):
    new_node = Node(state)
    new_node.starting_state = node.starting_state
    new_node.all_states = node.all_states + [state]
    new_node.action = node.action + [action]
    return  new_node



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
