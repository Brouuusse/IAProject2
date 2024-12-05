import numpy as np
import scipy.stats
from pacman_module.game import Agent, Directions, manhattanDistance, PriorityQueue

from collections import deque
from scipy.spatial.distance import cdist

class BeliefStateAgent(Agent):
    """Belief state agent.

    Arguments:
        ghost: The type of ghost (as a string).
    """

    def __init__(self, ghost):
        super().__init__()
        self.ghost = ghost


    def transition_matrix(self, walls, position):
        """Builds the transition matrix with optimizations for performance."""
        grid_width, grid_height = walls.width, walls.height

        walls_array = np.array(walls.data)

        transition_matrix = np.zeros((grid_width, grid_height, grid_width, grid_height))

        valid_positions = np.array([(x, y) for x in range(grid_width) for y in range(grid_height) if not walls_array[x, y]])
        position_indices = {tuple(pos): idx for idx, pos in enumerate(valid_positions)}

        distance_matrix = cdist(valid_positions, valid_positions, metric='cityblock')

        for (current_x, current_y), current_idx in position_indices.items():
            neighbor_indices = np.where(distance_matrix[current_idx] == 1)[0]
            neighbors = valid_positions[neighbor_indices]

            distances_to_pacman = np.array([manhattanDistance(tuple(neighbor), position) for neighbor in neighbors])
            current_distance_to_pacman = manhattanDistance((current_x, current_y), position)

            for neighbor, neighbor_distance_to_pacman in zip(neighbors, distances_to_pacman):
                neighbor_x, neighbor_y = neighbor
                if neighbor_distance_to_pacman < current_distance_to_pacman:
                    transition_matrix[current_x, current_y, neighbor_x, neighbor_y] = 1
                else:
                    if self.ghost == "terrified":
                        transition_matrix[current_x, current_y, neighbor_x, neighbor_y] = 8
                    elif self.ghost == "afraid":
                        transition_matrix[current_x, current_y, neighbor_x, neighbor_y] = 2
                    else:
                        transition_matrix[current_x, current_y, neighbor_x, neighbor_y] = 1

            transition_matrix[current_x, current_y] /= np.sum(transition_matrix[current_x, current_y])
            
        return transition_matrix


    def observation_matrix(self, walls, evidence, position):
        """Builds the observation matrix

            O_t = P(e_t | X_t)

        given a noisy ghost distance evidence e_t and the current Pacman
        position.

        Arguments:
            walls: The W x H grid of walls.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.

        Returns:
            The W x H observation matrix O_t.
        """
        observation_matrix = np.zeros((walls.width, walls.height))
        probability_distribution = [0.0625, 0.25, 0.375, 0.25, 0.0625]
        
        for x in range(walls.width):
            for y in range(walls.height):
                if walls[x][y]:
                    continue
                manhattan_difference = abs(manhattanDistance((x, y), position) - evidence)
                if manhattan_difference <= 2:
                    probability_index = evidence - manhattanDistance((x, y), position) + 2
                    observation_matrix[x][y] = probability_distribution[probability_index]
        
        total_sum = np.sum(observation_matrix)
        if total_sum > 0:
            observation_matrix /= total_sum

        return observation_matrix



    def update(self, walls, belief, evidence, position):
        """Updates the previous ghost belief state.

        b_{t-1} = P(X_{t-1} | e_{1:t-1})

        Arguments:
            walls: The W x H grid of walls.
            belief: The belief state for the previous ghost position b_{t-1}.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.

        Returns:
            The updated ghost belief state b_t as a W x H matrix.
        """
        transition_matrix = self.transition_matrix(walls, position)
        observation_matrix = self.observation_matrix(walls, evidence, position)

        grid_width, grid_height = walls.width, walls.height
        updated_belief = np.zeros((grid_width, grid_height))

        valid_positions = np.array([[x, y] for x in range(grid_width) for y in range(grid_height) if not walls[x][y]])

        for current_x, current_y in valid_positions:
            transition_probability_sum = np.sum(transition_matrix[:, :, current_x, current_y] * belief)
            updated_belief[current_x, current_y] = observation_matrix[current_x, current_y] * transition_probability_sum

        updated_belief /= np.sum(updated_belief)

        return updated_belief


    def get_action(self, state):
        """Updates the previous belief states given the current state.

        ! DO NOT MODIFY !

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            The list of updated belief states.
        """

        walls = state.getWalls()
        beliefs = state.getGhostBeliefStates()
        eaten = state.getGhostEaten()
        evidences = state.getGhostNoisyDistances()
        position = state.getPacmanPosition()

        new_beliefs = [None] * len(beliefs)

        for i in range(len(beliefs)):
            if eaten[i]:
                new_beliefs[i] = np.zeros_like(beliefs[i])
            else:
                new_beliefs[i] = self.update(
                    walls,
                    beliefs[i],
                    evidences[i],
                    position,
                )

        return new_beliefs



class PacmanAgent(Agent):
    """Pacman agent that tries to eat ghosts given belief states."""

    def __init__(self):
        super().__init__()
        self.target = -1
    def _get_action(self, walls, beliefs, eaten, position):
            """
            Arguments:
                walls: The W x H grid of walls.
                beliefs: The list of current ghost belief states.
                eaten: A list of booleans indicating which ghosts have been eaten.
                position: The current position of Pacman.

            Returns:
                A legal move as defined in `game.Directions`.
            """
            best_action = Directions.STOP
            bestPos = (0, 0)
            closestDist = float("inf")
            if self.target == -1 or eaten[self.target]:
                for i in range(len(beliefs)):
                    maxBelief = 0
                    bestPosT = (0, 0)
                    if eaten[i]:
                        continue
                    for j in range(walls.width):
                        for k in range(walls.height):
                            sum = 0
                            totSum = 0
                            if walls[j][k]:
                                continue
                            lB = max(0, j-1)
                            uB = min(walls.width, j+2)
                            for l in range(lB, uB):
                                lB = max(0, k-1)
                                uB = min(walls.height, k+2)
                                for m in range(lB, uB):
                                    if walls[l][m]:
                                        continue
                                    sum += beliefs[i][l][m]
                                    totSum += 1
                            if sum > maxBelief:
                                maxBelief = sum
                                bestPosT = (j, k)
                    if manhattanDistance(position, bestPosT) < closestDist:
                        closestDist = manhattanDistance(position, bestPosT)
                        bestPos = bestPosT
                        self.target = i
            else:
                max_belief = 0
                for j in range(walls.width):
                    for k in range(walls.height):
                        if walls[j][k]:
                            continue
                        if beliefs[self.target][j][k] > max_belief:
                            max_belief = beliefs[self.target][j][k]
                            bestPosT = (j, k)
                bestPos = bestPosT
            pq = PriorityQueue()
            pq.push((position, [], 0), manhattanDistance(position, bestPos))
            visited = []
            belief = beliefs[self.target]
            x = belief * np.arange(walls.width)[:, np.newaxis]
            y = np.sum(belief * np.arange(walls.height), axis=1)
            bestPosT = bestPos
            bestPos = (int(np.sum(x)), int(np.sum(y)))
            if position == bestPos or walls[bestPos[0]][bestPos[1]]:
                bestPos = bestPosT
            dTPrio = []
            dirNorth = (bestPos[0] - position[0], Directions.NORTH)
            dirSouth = (position[0] - bestPos[0], Directions.SOUTH)
            dirEast = (bestPos[1] - position[1], Directions.EAST)
            dirWest = (position[1] - bestPos[1], Directions.WEST)
            dTPrio.append(dirNorth)
            dTPrio.append(dirSouth)
            dTPrio.append(dirEast)
            dTPrio.append(dirWest)
            dTPrio.sort(key=lambda x: x[0])
            while True:
                if pq.isEmpty():
                    return Directions.STOP
                _, ((x, y), path, c) = pq.pop()
                if (x, y) == bestPos and path:
                    best_action = path[0]
                    break
                if (x, y) in visited:
                    continue
                visited.append((x, y))
                directions = []
                for _, action in dTPrio:
                    if action == Directions.NORTH:
                        directions.append((x, y + 1, action))
                    elif action == Directions.SOUTH:
                        directions.append((x, y - 1, action))
                    elif action == Directions.EAST:
                        directions.append((x + 1, y, action))
                    elif action == Directions.WEST:
                        directions.append((x - 1, y, action))

                for xn, yn, action in directions:
                    if not walls[xn][yn] and (xn, yn):
                        pq.push(((xn, yn), path + [action], c + 1),
                                c + 1 + manhattanDistance((xn, yn), bestPos))
            return best_action
    # def _get_action(self, walls, beliefs, eaten, position):
    #     """
    #     Arguments:
    #         walls: The W x H grid of walls.
    #         beliefs: The list of current ghost belief states.
    #         eaten: A list of booleans indicating which ghosts have been eaten.
    #         position: The current position of Pacman.

    #     Returns:
    #         A legal move as defined in `game.Directions`.
    #     """
    #     if self.target == -1 or eaten[self.target]:
    #         best_position, self.target = self._get_best_target_position(walls, beliefs, eaten, position)
    #     else:
    #         best_position = self._get_best_position_for_target(walls, beliefs, position)
    #     return self._find_best_action_to_target(walls, position, best_position)


    # def _get_best_target_position(self, walls, beliefs, eaten, position):
    #     best_position = (0, 0)
    #     best_target = -1
        
    #     candidates = [
    #         (self._get_candidate_position_for_ghost(walls, beliefs, ghost_index), ghost_index)
    #         for ghost_index in range(len(beliefs))
    #         if not eaten[ghost_index]
    #     ]
        
    #     best_position, best_target = min(
    #         candidates, 
    #         key=lambda x: manhattanDistance(position, x[0]), 
    #         default=((0, 0), -1)
    #     )

    #     self.target = best_target
    #     return best_position, self.target

    
    
    # def _get_best_position_for_target(self, walls, beliefs, position):
    #     max_belief = 0
    #     best_position = (0, 0)
        
    #     target_beliefs = beliefs[self.target]
        
    #     for x in range(walls.width):
    #         for y in range(walls.height):
    #             if walls[x][y]: 
    #                 continue
    #             belief = target_beliefs[x][y]
    #             if belief > max_belief:
    #                 max_belief = belief
    #                 best_position = (x, y)
        
    #     return best_position

    
    
    
    # def _get_candidate_position_for_ghost(self, walls, beliefs, ghost_index):
    #     max_belief = 0
    #     candidate_position = (0, 0)

    #     ghost_beliefs = beliefs[ghost_index]
        
    #     for x in range(walls.width):
    #         for y in range(walls.height):
    #             if walls[x][y]:
    #                 continue

    #             belief_sum, neighbor_count = 0, 0
    #             x_min, x_max = max(0, x - 1), min(walls.width, x + 2)
    #             y_min, y_max = max(0, y - 1), min(walls.height, y + 2)
                
    #             for neighbor_x in range(x_min, x_max):
    #                 for neighbor_y in range(y_min, y_max):
    #                     if not walls[neighbor_x][neighbor_y]:
    #                         belief_sum += ghost_beliefs[neighbor_x][neighbor_y]
    #                         neighbor_count += 1
                
    #             if belief_sum > max_belief:
    #                 max_belief = belief_sum
    #                 candidate_position = (x, y)

    #     return candidate_position

    

    # def _find_best_action_to_target(self, walls, position, best_position):
    #     priority_queue = PriorityQueue()
    #     priority_queue.push((position, [], 0), manhattanDistance(position, best_position))
        
    #     visited_positions = set()
    #     direction_priorities = self._get_direction_priorities(position, best_position)

    #     while not priority_queue.isEmpty():
    #         _, ((current_x, current_y), current_path, cost) = priority_queue.pop()
            
    #         if (current_x, current_y) == best_position and current_path:
    #             return current_path[0]

    #         if (current_x, current_y) in visited_positions:
    #             continue
            
    #         visited_positions.add((current_x, current_y))
    #         self._expand_priority_queue(walls, current_x, current_y, current_path, cost, best_position, priority_queue, direction_priorities)

    #     return Directions.STOP



    # def _get_direction_priorities(self, position, best_position):
    #     delta_x = best_position[0] - position[0]
    #     delta_y = best_position[1] - position[1]

    #     direction_priorities = [
    #         (delta_x, Directions.NORTH),
    #         (-delta_x, Directions.SOUTH),
    #         (delta_y, Directions.EAST),
    #         (-delta_y, Directions.WEST)
    #     ]
        
    #     direction_priorities.sort(key=lambda direction: direction[0])
        
    #     return direction_priorities




    # def _expand_priority_queue(self, walls, current_x, current_y, current_path, cost, best_position, priority_queue, direction_priorities):
    #     direction_deltas = {
    #         Directions.NORTH: (0, 1),
    #         Directions.SOUTH: (0, -1),
    #         Directions.EAST: (1, 0),
    #         Directions.WEST: (-1, 0)
    #     }

    #     for _, action in direction_priorities:
    #         delta_x, delta_y = direction_deltas[action]
    #         next_x, next_y = current_x + delta_x, current_y + delta_y
            
    #         if 0 <= next_x < walls.width and 0 <= next_y < walls.height and not walls[next_x][next_y]:
    #             new_cost = cost + 1
    #             priority = new_cost + manhattanDistance((next_x, next_y), best_position)
                
    #             priority_queue.push(
    #                 ((next_x, next_y), current_path + [action], new_cost),
    #                 priority
    #             )



    # def _get_action(self, walls, beliefs, eaten, position):
    #     """
    #     Arguments:
    #         walls: The W x H grid of walls.
    #         beliefs: The list of current ghost belief states.
    #         eaten: A list of booleans indicating which ghosts have been eaten.
    #         position: The current position of Pacman.

    #     Returns:
    #         A legal move as defined in `game.Directions`.
    #     """
    #     best_action = Directions.STOP
    #     best_position = (0, 0)
    #     min_distance = float("inf")

    #     if self.target == -1 or eaten[self.target]:
    #         for ghost_index in range(len(beliefs)):
    #             if eaten[ghost_index]:
    #                 continue

    #             max_belief = 0
    #             candidate_position = (0, 0)
    #             for x in range(walls.width):
    #                 for y in range(walls.height):
    #                     if walls[x][y]:
    #                         continue

    #                     belief_sum = 0
    #                     neighbor_count = 0

    #                     x_min, x_max = max(0, x - 1), min(walls.width, x + 2)
    #                     y_min, y_max = max(0, y - 1), min(walls.height, y + 2)

    #                     for neighbor_x in range(x_min, x_max):
    #                         for neighbor_y in range(y_min, y_max):
    #                             if not walls[neighbor_x][neighbor_y]:
    #                                 belief_sum += beliefs[ghost_index][neighbor_x][neighbor_y]
    #                                 neighbor_count += 1

    #                     if belief_sum > max_belief:
    #                         max_belief = belief_sum
    #                         candidate_position = (x, y)

    #             if manhattanDistance(position, candidate_position) < min_distance:
    #                 min_distance = manhattanDistance(position, candidate_position)
    #                 best_position = candidate_position
    #                 self.target = ghost_index

    #     else:
    #         max_belief = 0
    #         for x in range(walls.width):
    #             for y in range(walls.height):
    #                 if walls[x][y]:
    #                     continue
    #                 belief = beliefs[self.target][x][y]
    #                 if belief > max_belief:
    #                     max_belief = belief
    #                     best_position = (x, y)

    #     priority_queue = PriorityQueue()
    #     priority_queue.push((position, [], 0), manhattanDistance(position, best_position))
    #     visited_positions = set()
    #     target_beliefs = beliefs[self.target]

    #     weighted_x = np.sum(target_beliefs * np.arange(walls.width)[:, np.newaxis], axis=1)
    #     weighted_y = np.sum(target_beliefs * np.arange(walls.height), axis=0)
    #     candidate_best_position = best_position
    #     best_position = (int(np.sum(weighted_x)), int(np.sum(weighted_y)))

    #     if position == best_position or walls[best_position[0]][best_position[1]]:
    #         best_position = candidate_best_position

    #     directions = [
    #         ((best_position[0] - position[0]), Directions.NORTH),
    #         ((position[0] - best_position[0]), Directions.SOUTH),
    #         ((best_position[1] - position[1]), Directions.EAST),
    #         ((position[1] - best_position[1]), Directions.WEST)
    #     ]
    #     directions.sort(key=lambda direction: direction[0])

    #     while not priority_queue.isEmpty():
    #         _, ((current_x, current_y), current_path, cost) = priority_queue.pop()

    #         if (current_x, current_y) == best_position and current_path:
    #             best_action = current_path[0]
    #             break

    #         if (current_x, current_y) in visited_positions:
    #             continue

    #         visited_positions.add((current_x, current_y))
    #         potential_moves = []

    #         for _, action in directions:
    #             if action == Directions.NORTH:
    #                 potential_moves.append((current_x, current_y + 1, action))
    #             elif action == Directions.SOUTH:
    #                 potential_moves.append((current_x, current_y - 1, action))
    #             elif action == Directions.EAST:
    #                 potential_moves.append((current_x + 1, current_y, action))
    #             elif action == Directions.WEST:
    #                 potential_moves.append((current_x - 1, current_y, action))

    #         for next_x, next_y, action in potential_moves:
    #             if not walls[next_x][next_y] and (next_x, next_y):
    #                 priority_queue.push(
    #                     ((next_x, next_y), current_path + [action], cost + 1),
    #                     cost + 1 + manhattanDistance((next_x, next_y), best_position)
    #                 )

    #     return best_action



    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        ! DO NOT MODIFY !

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        return self._get_action(
            state.getWalls(),
            state.getGhostBeliefStates(),
            state.getGhostEaten(),
            state.getPacmanPosition(),
        )
