


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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        old_pos = currentGameState.getPacmanPosition()


        def sort_min_distance(val, val2=newPos):
            return abs(val[0] - val2[0]) + abs(val[1] - val2[1])
        def sort_min_distance_2(val, val2=old_pos):
            return abs(val[0] - val2[0]) + abs(val[1] - val2[1])

        # print("success new pos, {}".format(newPos))
        old_food = currentGameState.getFood()
        old_food_list = old_food.asList()
        old_food_not_found = [i for i in old_food_list if old_food[i[0]][i[1]]]
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        ghost_pos = [g.getPosition() for g in newGhostStates]
        food = newFood.asList()
        food_not_found = [i for i in food if newFood[i[0]][i[1]]]
        # print("ghost state, {}".format(newGhostStates))
        num_food_left = successorGameState.getNumFood()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        min_moves_left = [val for val in newScaredTimes if val != 0]
        scared_gos_pos = [g.getPosition() for g in newGhostStates if g.scaredTimer != 0]
        if food_not_found == []:
            return 10000
        if min_moves_left == []:
            # check for nearest ghost
            ghost_pos.sort(key=sort_min_distance)
            if sort_min_distance(ghost_pos[0]) <= 8:
                if sort_min_distance(ghost_pos[0]) <= 4:
                    return -1000
        # lets see if there's a ghost nearby to eat
        else:
            # get the ones we can eat
            min_value = 9999
            step_left = 0
            for ghost in range(len(scared_gos_pos)):
                if sort_min_distance(scared_gos_pos[ghost]) < min_value:
                    min_value = sort_min_distance(scared_gos_pos[ghost])
                    step_left = min_moves_left[ghost]
            # check if we can reach the ghost
            if min_value <= step_left:
                return 1000
        # dealt with nearby ghost.
        # deal with with food now, get nearest food to pacman
        # check if an action brought reduce food.
        actions = currentGameState.getLegalActions()
        if old_food[newPos[0]][newPos[1]]:
            return 10
        else:
            #find closest dot and figure out which action gets you closest.
            old_food_not_found.sort(key=sort_min_distance_2)
            closets = old_food_not_found[0]
            dist_to_closest = sort_min_distance(closets,old_pos)
            best_act = None
            for a in actions:
                new_state = currentGameState.generatePacmanSuccessor(a)
                cur_pos = new_state.getPacmanPosition()
                if sort_min_distance(cur_pos, closets) < dist_to_closest:
                    best_act = a
                    dist_to_closest = sort_min_distance(cur_pos, closets)
            if best_act == action:
                return 5
            return 1





    # WINNING STATE
    # food_not_found.sort(key=sort_min_distance)
    # food_man_list= [sort_min_distance(food_not_found[i])for i in range(len(food_not_found))]

    # food_man = sum(food_man_list)
    # old_food_man = sum([sort_min_distance(old_food_not_found[i])for i in range(len(old_food_not_found))])
    # if food_man < old_food_man:
    #	return 10 + (1/(float(sort_min_distance(food_not_found[0]))))
    # return 1.0/(float(sort_min_distance(food_not_found[0])))
    # dis_to_nearest_food = -(float(sort_min_distance(food_not_found[0]))) + -(float(food_man))

    # # most important
    # # print("food, {}".format(newFood))
    #
    # # print("new scared times, {}".format(newScaredTimes))
    # # worse case is when ghost is close and not scared
    # #
    # # # best case when its close and its scared
    # # # how many points where gained ?
    # gained_score = successorGameState.getScore() - currentGameState.getScore()
    # # ghost_pellet = sum([True if val!=0 else False for val in newScaredTimes])
    # # min_moves_left = [val for val in newScaredTimes if val!=0]
    # # min_move = 0
    # # if len(min_moves_left)>0:
    # #     min_move = min(min_moves_left)
    # current_score = currentGameState.getScore()
    # if current_score == 0:
    #     current_score = 1
    # score = abs(gained_score/current_score)
    # if score < 0:
    #     final_score = - abs(score/current_score)
    # else:
    #     final_score = abs(score/current_score)
    #
    #
    # #     score = currentGameState.getScore() +1
    # # did this action cause a better outcome?
    #
    # # distance to nearest foods
    #
    #
    # # distance to ghosts
    #
    #
    #
    #
    #
    #
    #
    # "*** YOUR CODE HERE ***"
    # return final_score


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

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        self.num_agents = gameState.getNumAgents()
        best_move, evalu = self.generate_min_max(gameState, self.depth + 1, self.num_agents, 0)
        # print("the max value is {}".format(evalu))
        return best_move

    def update_turn(self, turn):
        cur_turn = turn
        if cur_turn == self.num_agents - 1:
            return 0
        return cur_turn + 1

    def terminal(self, gamestate, depth, turn):
        if depth == 0 and turn > 0:
            return True
        if depth == 1 and turn == 0:
            return True
        return gamestate.isWin() or gamestate.isLose()

    def generate_min_max(self, gamestate, depth, max_player, turn):
        best_move = None
        if self.terminal(gamestate, depth, turn):
            return best_move, self.evaluationFunction(gamestate)
        if turn == 0:
            value = -9999999
            depth -= 1

        else:
            value = 9999999
        # print("current depth {}".format(depth))
        # print("the value to start off with for turn {} is {}".format(turn, value))

        last_turn = turn
        for move in gamestate.getLegalActions(turn):
            next_state = gamestate.generateSuccessor(turn, move)
            next_move, next_val = self.generate_min_max(next_state, depth, max_player, self.update_turn(turn))
            # print("turn is {}".format(last_turn))

            if last_turn == 0 and value < next_val:
                value, best_move = next_val, move

            elif last_turn > 0 and value > next_val:
                # print("GOT HERE")
                value, best_move = next_val, move
        # if value == 9999999:
        # print("depth {} and value {}".format(depth, value))
        # print("DONE THIS")
        return best_move, value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        self.num_agents = gameState.getNumAgents()
        best_move, evalu = self.alpha_beta(gameState, self.depth + 1, self.num_agents, 0, -999999999, 999999999)
        # print("the max value is {}".format(evalu))
        return best_move

    def update_turn(self, turn):
        cur_turn = turn
        if cur_turn == self.num_agents - 1:
            return 0
        return cur_turn + 1

    def terminal(self, gamestate, depth, turn):
        if depth == 0 and turn > 0:
            return True
        if depth == 1 and turn == 0:
            return True
        return gamestate.isWin() or gamestate.isLose()

    def alpha_beta(self, gamestate, depth, max_player, turn, alpha, beta):
        best_move = None
        if self.terminal(gamestate, depth, turn):
            return best_move, self.evaluationFunction(gamestate)
        if turn == 0:
            value = -9999999
            depth -= 1

        else:
            value = 9999999
        # print("current depth {}".format(depth))
        # print("the value to start off with for turn {} is {}".format(turn, value))

        last_turn = turn
        for move in gamestate.getLegalActions(turn):
            next_state = gamestate.generateSuccessor(turn, move)
            next_move, next_val = self.alpha_beta(next_state, depth, max_player, self.update_turn(turn), alpha, beta)
            # print("turn is {}".format(last_turn))

            if last_turn == 0:
                if value < next_val:
                    value, best_move = next_val, move
                if value >= beta:
                    return best_move, value
                alpha = max(alpha, value)
            elif last_turn > 0:
                if value > next_val:
                    # print("GOT HERE")
                    value, best_move = next_val, move
                if value <= alpha:
                    return best_move, value
                beta = min(beta, value)

        # if value == 9999999:
        # print("depth {} and value {}".format(depth, value))
        # print("DONE THIS")
        return best_move, value


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        self.num_agents = gameState.getNumAgents()
        best_move, evalu = self.expectimax(gameState, self.depth + 1, self.num_agents, 0)
        # print("the max value is {}".format(evalu))
        return best_move

    def update_turn(self, turn):
        cur_turn = turn
        if cur_turn == self.num_agents - 1:
            return 0
        return cur_turn + 1

    def terminal(self, gamestate, depth, turn):
        if depth == 0 and turn > 0:
            return True
        if depth == 1 and turn == 0:
            return True
        return gamestate.isWin() or gamestate.isLose()

    def expectimax(self, gamestate, depth, max_player, turn):
        best_move = None
        if self.terminal(gamestate, depth, turn):
            return best_move, self.evaluationFunction(gamestate)
        if turn == 0:
            value = -9999999
            depth -= 1

        else:
            value = 0
        # print("current depth {}".format(depth))
        # print("the value to start off with for turn {} is {}".format(turn, value))

        last_turn = turn
        length_of_actions = float(len(gamestate.getLegalActions(turn)))
        for move in gamestate.getLegalActions(turn):
            next_state = gamestate.generateSuccessor(turn, move)
            next_move, next_val = self.expectimax(next_state, depth, max_player, self.update_turn(turn))
            # print("turn is {}".format(last_turn))

            if last_turn == 0 and value < next_val:
                value, best_move = next_val, move

            elif last_turn > 0:
                # print("GOT HERE")
                value = value + next_val * (1.0 / length_of_actions)
        # if value == 9999999:
        # print("depth {} and value {}".format(depth, value))
        # print("DONE THIS")
        return best_move, value


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    # best state is where there is no food left.

    food_grid = currentGameState.getFood()
    food_list = food_grid.asList()
    pacman_pos = currentGameState.getPacmanPosition()

    def sort_min_distance(val, val2=pacman_pos):
        return abs(val[0] - val2[0]) + abs(val[1] - val2[1])
    # get the current score
    score = currentGameState.getScore()
    if food_list == []:
        return 10000 + score

    # next check for worse states.
    # check if we are in a power state; check all the ghost scared time
    ghostStates = currentGameState.getGhostStates()
    ghost_pos = [g.getPosition() for g in ghostStates]
    food_not_found = [i for i in food_list if food_grid[i[0]][i[1]]]
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    min_moves_left = [val for val in scaredTimes if val != 0]
    scared_gos_pos = [g.getPosition() for g in ghostStates if g.scaredTimer != 0]
    not_scared_gos_pos = [g.getPosition() for g in ghostStates if g.scaredTimer == 0]
    if min_moves_left != []:
        # we have scared ghosts!. We like this state.
        # check to see if there are no monsters near us.
        if not_scared_gos_pos != []:
            # potentially monsters nearby
            for monster_pos in not_scared_gos_pos:
                if sort_min_distance(monster_pos) <= 4:
                    if sort_min_distance(monster_pos) == 0:
                        return - 10000
                    return -100 + score
            # otherwise no monster nearby. Check if there are scared ghost nearby
            for ghost_pos in scared_gos_pos:
                if sort_min_distance(ghost_pos) <= 4:
                    if sort_min_distance(ghost_pos) == 0:
                        return 600 + score
                    return 500/(sort_min_distance(ghost_pos)+1) + score
    # no scared ghost. Everyone is a monster. Let's see if there's pellets nearby
    nearby_pellet = currentGameState.getCapsules()
    # get the closets pellet.
    if nearby_pellet != []:
        nearby_pellet.sort(key=sort_min_distance)
        closest_pellet = nearby_pellet[0]
        # no_monsters = True
        # for monster in not_scared_gos_pos:
        #     if sort_min_distance(monster) <= 4:
        #         no_monsters = False
        # if no_monsters:
        #     if sort_min_distance(closest_pellet) <= 4:
        #         return 400 + score
        if sort_min_distance(closest_pellet) <= 4:
            if sort_min_distance(closest_pellet) == 0:
                return 500 + score
            return 400/(sort_min_distance(closest_pellet)+1) + score
    # no nearby pellets.  see if there are monster nearby
    if not_scared_gos_pos != []:
        # potentially monsters nearby
        for monster_pos in not_scared_gos_pos:
            if sort_min_distance(monster_pos) <= 4:
                if sort_min_distance(monster_pos) == 0:
                    return - 10000
                return -20*(5-sort_min_distance(monster_pos)) + score
    # This state will need to be determined by how much food is left, and how long it will
    # take to get there, and how many points we have.
    # see if there's somewhere we can go (directly to the left or right up or down and get food.
    legal_act = currentGameState.getLegalActions()
    number_food_left = currentGameState.getNumFood()
    food_nearby = [False] * len(legal_act)
    for i in range(len(legal_act)):
        temp_sec = currentGameState.generatePacmanSuccessor(legal_act[i])
        temp_food_left = temp_sec.getNumFood()
        if temp_food_left < number_food_left:
            food_nearby[i] = True
    num_food_nearby = sum(food_nearby)
    if num_food_nearby > 0:
        return score + 2*num_food_nearby + 1.0/float(number_food_left**2)
    # there is no food nearby. Let's try and get closer to our next food.
    food_not_found.sort(key=sort_min_distance)
    distance_to_closets = sort_min_distance(food_not_found[0])
    return score + 1.0/float(distance_to_closets**2) + 1.0/float(number_food_left**2)

















#     if len(food_not_found) > 5:
#         total_length = len(food_not_found)
#         new_food_not_found = food_not_found[:5]
#     else:
#         total_length = len(food_not_found)
#         new_food_not_found = food_not_found
#     # create mst
#     graph = create_graph(new_food_not_found + [pacman_pos])
#     order, min_order = find_perm(graph, [pacman_pos], pacman_pos, len(new_food_not_found) + 1, [0])
#     return score  - sum(min_order) + -total_length
#
#
#
#
# def find_perm(graph, seen_list, current_pos, need_see, list_weight):
#     # graph => u,v, weight ,
#     if len(seen_list) == need_see:
#         return seen_list, list_weight
#     else:
#         new_info = []
#         for val in graph:
#             if current_pos in val:
#                 new_seen = seen_list[:]
#                 new_weights = list_weight[:]
#                 u, v, weight = val
#                 if not (u in seen_list and v in seen_list):
#                     new_weights.append(weight)
#                     if u == current_pos:
#                         new_seen.append(v)
#                         new_pos = v
#                     else:
#                         new_seen.append(u)
#                         new_pos = u
#                     new_info.append([new_seen, new_weights, new_pos])
#         overall_info = []
#         for info in new_info:
#             ordered_seen, listed_weights = find_perm(graph, info[0], info[2], need_see, info[1])
#             overall_info.append([ordered_seen, listed_weights])
#         min_1 = 9328409099999999999999999999
#         final_order = []
#         final_min = []
#         for new_info in overall_info:
#             if sum(new_info[1]) < min_1:
#                 min_1 = sum(new_info[1])
#                 final_min = new_info[1]
#                 final_order = new_info[0]
#         return final_order, final_min
#
# def create_graph(food_needed):
#     # create the graph for this
#
#     def sort_min_distance(val):
#         return abs(val[0][0] - val[1][0]) + abs(val[0][1] - val[1][1])
#
#     graph = []
#     seen_nodes = []
#     final_food = []
#     food_after = [[[i, x] for x in food_needed if i != x] for i in food_needed]
#     for food in food_after:
#         for val2 in food:
#             final_food.append(val2)
#     final_food.sort(key=sort_min_distance)
#
#     for food in final_food:
#         if not (food[1], food[0]) in seen_nodes:
#             seen_nodes.append((food[0], food[1]))
#             graph.append([food[0], food[1], abs(food[0][0] - food[1][0]) + abs(food[0][1] - food[1][1])])
#
#     return graph


# Abbreviation
better = betterEvaluationFunction

