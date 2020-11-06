# searchAgents.py
# ---------------
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
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""

from game import Directions
from game import Agent
from game import Actions
import util
import time
import search

class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError, fn + ' is not a search function in search.py.'
        func = getattr(search, fn)
        if 'heuristic' not in func.func_code.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError, heuristic + ' is not a function in searchAgents.py or search.py.'
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError, prob + ' is not a search problem type in SearchAgents.py.'
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception, "No search function provided for SearchAgent"
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn, (1, 1), None, False)

class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """

    def __init__(self, startingGameState):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height-2, self.walls.width-2
        self.corners = ((1,1), (1,top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print 'Warning: no food in corner ' + str(corner)
        self._expanded = 0 # DO NOT CHANGE; Number of search nodes expanded
        # Please add any code here which you would like to use
        # in initializing the problem
        "*** YOUR CODE HERE ***"
        # check to see if starting position is a goal position:
        # goal positions are the corner. goal state is to collect all the corners
        self.goal = [(1,1), (1,top), (right, 1), (right, top)]


        # states will be a tuple that holds current position and collected corners if reached

        self.startState = (self.startingPosition, False, False, False, False )


    def getStartState(self):
        """
        Returns the start state (in your state space, not the full Pacman state
        space)
        """
        return self.startState
    def isGoalState(self, state):
        """
        Returns whether this search state is a goal state of the problem.
        """
        current_pos, corner_1, corner_2, corner_3, corner_4 = state
        corner_list = [corner_1, corner_2, corner_3, corner_4]
        for c in range(len(self.corners)):
            if self.corners[c] == current_pos:
                corner_list[c] = True



        return corner_list[0] and corner_list[1] and corner_list[2] and corner_list[3]

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
            For a given state, this should return a list of triples, (successor,
            action, stepCost), where 'successor' is a successor to the current
            state, 'action' is the action required to get there, and 'stepCost'
            is the incremental cost of expanding to that successor
        """
        currentPosition, corner_1, corner_2, corner_3, corner_4 = state
        corner_list = [corner_1,corner_2, corner_3, corner_4]
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # Add a successor state to the successor list if the action is legal
            # Here's a code snippet for figuring out whether a new position hits a wall:
            x,y = currentPosition
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]
            if not hitsWall:
                # check to see if new position is in a goal position:
                for corner in range(len(self.corners)):
                    if self.corners[corner] == (x, y):
                        corner_list[corner] = True
                nextState = ((nextx,nexty), corner_list[0],corner_list[1],corner_list[2],corner_list[3])
                successors.append( ( nextState, action, 1) )

        self._expanded += 1 # DO NOT CHANGE
        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        if actions == None: return 999999
        x,y= self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)


def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    This function should always return    shortest path from the state to a goal of the problem; i.e., it should be
 a number that is a lower bound on the
    admissible.
    """
    corners = problem.corners # These are the corner coordinates
    walls = problem.walls # These are the walls of the maze, as a Grid (game.py)
    # heuristic is => manhantan distance of closest unseen corner
    # goal  {(1,1), (1,top), (right, 1), (right, top)}
    current_pos, corner_1, corner_2, corner_3, corner_4 = state
    corner_list = [corner_1, corner_2, corner_3, corner_4]
    # for c in range(len(corners)):
    #     if corners[c] == current_pos:
    #         corner_list[c] = True
    #add how many walls are directly
    # find shortest
    max_1 = 999999999999999999999999
    min_1 = []
    corner_left = 0
    corner_chosen = None
    j = None
    k = None
    for i in range(len(corners)):
        if not corner_list[i]:
            # penalty for walls
            # how many walls around current postion
            d = abs(current_pos[0] - corners[i][0]) + abs(current_pos[1] - corners[i][1])
            if d < max_1:
                max_1 = d
                corner_chosen = corners[i]
                j = i
            min_1.append(d)

    if min_1 == []:
        return 0

    corner_list[j] = True
    min_2 = max_1
    while corner_chosen is not None:
        next_min, corner_chosen, corner_list = check_distance(corners,corner_chosen, corner_list)
        min_2 += next_min


    return min_2
    # min_d, corner_seen = check_distance(corners, current_pos, corner_list)
    # if min_d == 10000000000:
    #     return 0
    # min_d2, corner_seen2= check_distance(corners, corner_seen, corner_list)
    # if min_d2 == 10000000000:
    #     return min_d
    # return min_d + min_d2


def check_distance(corners, current_pos, corner_list):
    max_1 = 999999999999999999999999
    min_1 = []
    corner_left = 0
    corner_chosen = None
    j = None
    for i in range(len(corners)):
        if not corner_list[i]:
            # penalty for walls
            # how many walls around current postion
            d = abs(current_pos[0] - corners[i][0]) + abs(current_pos[1] - corners[i][1])
            if d < max_1:
                max_1 = d
                corner_chosen = corners[i]
                j = i
            min_1.append(d)

    if min_1 == []:
        return 0, None, [True, True, True, True]

    corner_list[j] = True
    return max_1, corner_chosen, corner_list




class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem

class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        # cost

        self._expanded = 0 # DO NOT CHANGE
        self.heuristicInfo = {}
        food_list = startingGameState.getFood().asList()
        self.heuristicInfo['distances'] = {}
        # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1 # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost

class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem

def foodHeuristic(state, problem):
    """Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be admissible to ensure correctness.

    If using A* ever finds a solution that is worse uniform cost
    search finds, your heuristic is *not* admissible!  On the other
    hand, inadmissible heuristics may occasionally find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']

    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"
    dist = problem.heuristicInfo['distances']
    heur_prob = problem.heuristicInfo
    food = foodGrid.asList()
    food_not_found = [i for i in food if foodGrid[i[0]][i[1]]== True]
    def sort_min_distance(val, val2=position):
        return abs(val[0]-position[0]) - abs(val[1]-position[1])
    food_not_found.sort(key=sort_min_distance)
    if len(food_not_found) > 5:
        food_not_found = food_not_found[:5]
    # create mst
    if food_not_found == []:
        return 0
    min_d = 99999999999999999999999999999
    last_d = 0
    last_order = []
    last_weight = []
    last_food = None
    min_order = []
    order = []
    graph = create_graph(food_not_found + [position])
    order, min_order = find_perm(graph, [position], position, len(food_not_found)+1, [0])



    # for food in food_not_found:
    #     min_order = []
    #     order = []
    #     cur_dis = abs(food[0] - position[0]) - abs(food[1] - position[1])
    #     if dist.get(food, []) != []:
    #         print(dist[food])
    #         for orders in dist[food]:
    #             if set(food_not_found) == set(orders[0]):
    #                 order, min_order = orders
    #     if min_order == []:
    #         graph = create_graph(food_not_found)
    #         seen_list = [food]
    #         current_pos = food
    #         order, min_order = find_perm(graph, seen_list, current_pos, len(food_not_found), [0])
    #
    #     if sum(min_order) < min_d:
    #         min_d = sum(min_order)
    #         last_weight = min_order
    #         last_order = order
    #         last_d = cur_dis
    #         last_food = food
    # if dist.get(last_food, []) != []:
    #     if (last_order,last_weight) not in dist[last_food]:
    #         dist[last_food].append((last_order, last_weight))
    # else:
    #     dist[last_food] = [(last_order, last_weight)]
    # problem.heuristicInfo['distances'] = dist
    # return min_d + last_d
    return sum(min_order)





# order the 3 closests dotes.
    # find the three closets nodes
    # recursively find the children
    # food_not_found = [i for i in food if foodGrid[i[0]][i[1]]== True]
    # if len(food_not_found) > 10:
    #     depth = 5
    # else:
    #     depth = len(food_not_found)
    # min_d = check_distance_food(food_not_found, position, depth)
    # return min_d


    # max_1 = 999999999999999999999999
    # max_2 = None
    # max_3 = None
    # min_1 = []
    # food_chosen = None
    # food_chosen2 = None
    # food_chosen3 = None
    # found_food = [True if foodGrid[pos[0]][pos[1]] == False else False for pos in food]
    # j = None
    # k = None
    # n = None
    # for i in range(len(food)):
    #     if not found_food[i]:
    #         d = abs(position[0] - food[i][0]) + abs(position[1] - food[i][1])
    #         if d < max_1:
    #             if max_1 != 999999999999999999999999:
    #                 if max_2 :
    #                     max_3 = max_2
    #                     food_chosen3 = food_chosen2
    #                     n = k
    #                     print("the n is ",n)
    #                 max_2 = max_1
    #                 food_chosen2 = food_chosen3
    #                 k = j
    #                 print("the k is ", k)
    #
    #             max_1 = d
    #             food_chosen = food[i]
    #             j = i
    #             print("the j is ", j)
    #         min_1.append(d)
    #
    # if min_1 == []:
    #     return 0
    # min_list = [max_1, max_2, max_3]
    # chosen_list = [food_chosen, food_chosen2, food_chosen3]
    # check_min_list = [a for a in min_list if a is not None]
    # print("n is again ", n)
    # index_list = [j, k, n]
    #
    # check_chosen = [(chosen_list[x], index_list[x]) for x in range(len(min_list)) if min_list[x] is not None]
    # version_food_found = [found_food] * len(check_min_list)
    # overall_min = []
    # print("new set")
    # for m in range(len(check_min_list)):
    #     print(check_chosen[m][1])
    #     print("len of food_found", len(version_food_found[m]))
    #     version_food_found[m][check_chosen[m][1]] = True
    #     min_2 = check_min_list[m]
    #     next_min, check_chosen[m], version_food_found[m] = check_distance_food(food, check_chosen[m], version_food_found[m], depth=3)
    #     min_2 += next_min
    #     overall_min.append(min_2)
    # return min(overall_min)



#
# def check_distance_food(food, current_pos, depth):
#     """ food -> the food left to find
#       current pos -> the current position
#       depth -> what depth we are at ."""
#     # find longest consecutive path closest to pacy
#     # check depth
#     max_1 = 999999999999999999999999
#     max_2 = None
#     max_3 = None
#     min_1 = []
#     food_chosen = None
#     food_chosen2 = None
#     food_chosen3 = None
#     # j = None
#     # k = None
#     # n = None
#     # pick the three closests dots
#     for i in range(len(food)):
#             d = abs(current_pos[0] - food[i][0]) + abs(current_pos[1] - food[i][1])
#             if d < max_1:
#                 if max_1 != 999999999999999999999999:
#                     if max_2:
#                         max_3 = max_2
#                         food_chosen3 = food_chosen2
#                     max_2 = max_1
#                     food_chosen2 = food_chosen
#
#                 max_1 = d
#                 food_chosen = food[i]
#             min_1.append(d)
#
#     if min_1 == []:
#         # no more nodes return base case
#         return 0
#     min_list = [max_1, max_2, max_3]
#     chosen_list = [food_chosen, food_chosen2, food_chosen3]
#     # keep only the food distances that aren't none
#     check_min_list = [a for a in min_list if a is not None]
#     # index_list = [j, k, n]
#     # a list the food_chosen position
#     check_chosen = [(chosen_list[x]) for x in range(len(min_list)) if min_list[x] is not None]
#     overall_min = []
#     if depth == 0:
#         return min(check_min_list)
#     if depth > 0:
#         for m in range(len(check_min_list)):
#             current_food = food[:]
#             current_food.remove(check_chosen[m])
#             min_2 = check_min_list[m]
#             min_2 += check_distance_food(food, check_chosen[m], depth-1)
#             overall_min.append(min_2)
#     return min(overall_min)
#
#     # max_1 = 999999999999999999999999
#     # min_1 = []
#     # food_chosen = None
#     # j = None
#     # for i in range(len(food)):
#     #     if not food_list[i]:
#     #         # penalty for walls
#     #         # how many walls around current postion
#     #         d = abs(current_pos[0][0] - food[i][0]) + abs(current_pos[0][1] - food[i][1])
#     #         if d < max_1:
#     #             max_1 = d
#     #             food_chosen = food[i]
#     #             j = i
#     #         min_1.append(d)
#     #
#     # if min_1 == []:
#     #     return 0, (None,current_pos[1]), [True for i in food]
#     #
#     # food_list[j] = True
#     # return max_1, (food_chosen, current_pos[1]), food_list

def find_perm(graph, seen_list, current_pos, need_see, list_weight):
    # graph => u,v, weight ,
    if len(seen_list) == need_see:
        return seen_list, list_weight
    else:
        new_info = []
        for val in graph:
            if current_pos in val:
                new_seen = seen_list[:]
                new_weights = list_weight[:]
                u,v,weight = val
                if not(u in seen_list and v in seen_list):
                    new_weights.append(weight)
                    if u == current_pos:
                        new_seen.append(v)
                        new_pos = v
                    else:
                        new_seen.append(u)
                        new_pos = u
                    new_info.append([new_seen, new_weights, new_pos])
        overall_info = []
        for info in new_info:
            ordered_seen, listed_weights = find_perm(graph, info[0], info[2], need_see, info[1])
            overall_info.append([ordered_seen, listed_weights])
        min_1 = 9328409099999999999999999999
        final_order = []
        final_min = []
        for new_info in overall_info:
            if sum(new_info[1]) < min_1:
                min_1 = sum(new_info[1])
                final_min = new_info[1]
                final_order = new_info[0]
        return final_order, final_min

def create_graph(food_needed):
   # create the graph for this

    def sort_min_distance(val):
        return abs(val[0][0] - val[1][0]) + abs(val[0][1] - val[1][1])
    graph = []
    seen_nodes = []
    final_food = []
    food_after = [[[i,x] for x in food_needed if i != x] for i in food_needed]
    for food in food_after:
        for val2 in food:
            final_food.append(val2)
    final_food.sort(key=sort_min_distance)

    for food in final_food:
        if not(food[1], food[0]) in seen_nodes:
            seen_nodes.append((food[0], food[1]))
            graph.append([food[0], food[1], abs(food[0][0] - food[1][0]) + abs(food[0][1] - food[1][1])])

    return graph





class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"
    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while(currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState) # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception, 'findPathToClosestDot returned an illegal move: %s!\n%s' % t
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print 'Path found with cost %d.' % len(self.actions)

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        food_list = food.asList()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)
        max_d = 99999999999999999999999
        n = None
        for i in range(len(food_list)):
            d = mazeDistance(startPosition,food_list[i], gameState)
            if d < max_d:
                max_d = d
                n = i
        print("starting position : ", startPosition)
        print(food_list)
        problem.goal = food_list[n]
        print(food_list[n])

        "*** YOUR CODE HERE ***"
        actions = search.astar(problem)
        print(actions)
        food[food_list[n][0]][food_list[n][1]] = False
        return actions


class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x,y = state


        "*** YOUR CODE HERE ***"
        return self.goal == (x,y)

def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))

