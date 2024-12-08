import numpy as np
from scipy.stats import binom
from pacman_module.game import Agent, Directions, manhattanDistance, PriorityQueue

from scipy.spatial.distance import cdist

class BeliefStateAgent(Agent):
    """Belief state agent.

    Arguments:
        ghost: The type of ghost (as a string).
    """

    def __init__(self, ghost):
        super().__init__()
        self.ghost = ghost
        self.probability_distribution = binom.pmf(np.arange(0, 4 + 1), 4, 0.5)

    def transition_matrix(self, walls, position):
        """Builds the transition matrix

            T_t = P(X_t | X_{t-1})

        given the current Pacman position.

        Arguments:
            walls: The W x H grid of walls.
            position: The current position of Pacman.

        Returns:
            The W x H x W x H transition matrix T_t. The element (i, j, k, l)
            of T_t is the probability P(X_t = (k, l) | X_{t-1} = (i, j)) for
            the ghost to move from (i, j) to (k, l).
        """
        
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
        
        for x in range(walls.width):
            for y in range(walls.height):
                if walls[x][y]:
                    continue
                manhattan_distance = manhattanDistance((x, y), position)
                difference = evidence - manhattan_distance
                if abs(difference) <= 2:
                    probability_index = difference + 2
                    observation_matrix[x][y] = self.probability_distribution[probability_index]


        total_sum = np.sum(observation_matrix)
        if total_sum > 0:
            observation_matrix /= total_sum

        return observation_matrix
 
    def update(self, walls, belief, evidence, position):
        """Updates the previous ghost belief state

            b_{t-1} = P(X_{t-1} | e_{1:t-1})

        given a noisy ghost distance evidence e_t and the current Pacman
        position.

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
        self.has_moved = False

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
        
        if self.has_moved and self.target == -1 or eaten[self.target]:
            best_position, self.target = self._get_best_target_position(walls, beliefs, eaten, position)
        else:
            best_position = self._get_best_position_for_target(walls, beliefs, position)
            if not self.has_moved:
                self.has_moved = True
        return self._find_best_action_to_target(walls, position, best_position)


    def _get_best_target_position(self, walls, beliefs, eaten, position):
        best_position = (0, 0)
        best_target = -1
        
        candidates = [
            (self._get_candidate_position_for_ghost(walls, beliefs, ghost_index), ghost_index)
            for ghost_index in range(len(beliefs))
            if not eaten[ghost_index]
        ]
        
        if not candidates:
            return best_position, best_target
        
        best_position, best_target = min(
            candidates, 
            key=lambda x: manhattanDistance(position, x[0]), 
            default=((0, 0), -1)
        )

        self.target = best_target
        return best_position, self.target

    
    
    def _get_best_position_for_target(self, walls, beliefs, position):
        max_belief = 0
        best_position = (0, 0)
        
        target_beliefs = beliefs[self.target]
        
        for x in range(walls.width):
            for y in range(walls.height):
                if walls[x][y]: 
                    continue
                belief = target_beliefs[x][y]
                if belief > max_belief:
                    max_belief = belief
                    best_position = (x, y)
        
        return best_position
    
    def _get_candidate_position_for_ghost(self, walls, beliefs, ghost_index):
        max_belief = 0
        candidate_position = (0, 0)

        ghost_beliefs = beliefs[ghost_index]
        
        for x in range(walls.width):
            for y in range(walls.height):
                if walls[x][y]:
                    continue

                belief_sum, neighbor_count = 0, 0
                x_min, x_max = max(0, x - 1), min(walls.width, x + 2)
                y_min, y_max = max(0, y - 1), min(walls.height, y + 2)
                
                for neighbor_x in range(x_min, x_max):
                    for neighbor_y in range(y_min, y_max):
                        if not walls[neighbor_x][neighbor_y]:
                            belief_sum += ghost_beliefs[neighbor_x][neighbor_y]
                            neighbor_count += 1
                
                if belief_sum > max_belief:
                    max_belief = belief_sum
                    candidate_position = (x, y)

        return candidate_position

    def _find_best_action_to_target(self, walls, position, best_position):
        priority_queue = PriorityQueue()
        priority_queue.push((position, [], 0), manhattanDistance(position, best_position))
        
        visited_positions = set()
        direction_priorities = self._get_direction_priorities(position, best_position)

        while not priority_queue.isEmpty():
            _, ((current_x, current_y), current_path, cost) = priority_queue.pop()
            
            if (current_x, current_y) == best_position and current_path:
                return current_path[0]

            if (current_x, current_y) in visited_positions:
                continue
            
            visited_positions.add((current_x, current_y))
            self._expand_priority_queue(walls, current_x, current_y, current_path, cost, best_position, priority_queue, direction_priorities)

        return Directions.STOP

    def _get_direction_priorities(self, position, best_position):
        delta_x = best_position[0] - position[0]
        delta_y = best_position[1] - position[1]

        direction_priorities = [
            (delta_x, Directions.NORTH),
            (-delta_x, Directions.SOUTH),
            (delta_y, Directions.EAST),
            (-delta_y, Directions.WEST)
        ]
        
        direction_priorities.sort(key=lambda direction: direction[0])
        
        return direction_priorities

    def _expand_priority_queue(self, walls, current_x, current_y, current_path, cost, best_position, priority_queue, direction_priorities):
        direction_deltas = {
            Directions.NORTH: (0, 1),
            Directions.SOUTH: (0, -1),
            Directions.EAST: (1, 0),
            Directions.WEST: (-1, 0)
        }

        for _, action in direction_priorities:
            delta_x, delta_y = direction_deltas[action]
            next_x, next_y = current_x + delta_x, current_y + delta_y
            
            if 0 <= next_x < walls.width and 0 <= next_y < walls.height and not walls[next_x][next_y]:
                new_cost = cost + 1
                priority = new_cost + manhattanDistance((next_x, next_y), best_position)
                
                priority_queue.push(
                    ((next_x, next_y), current_path + [action], new_cost),
                    priority
                )

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
