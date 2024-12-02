import numpy as np
import scipy.stats
from pacman_module.game import Agent, Directions, manhattanDistance, PriorityQueue

from collections import deque

class BeliefStateAgent(Agent):
    """Belief state agent.

    Arguments:
        ghost: The type of ghost (as a string).
    """

    def __init__(self, ghost):
        super().__init__()

        self.ghost = ghost

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
        T_t = np.zeros((walls.width, walls.height, walls.width, walls.height))
        for i in range(walls.width):
            for j in range(walls.height):
                if walls[i][j]:
                    continue
                for k in range(walls.width):
                    for l in range(walls.height):
                        if walls[k][l]:
                            continue
                        if manhattanDistance((i, j), (k, l)) == 1:
                            if manhattanDistance((k, l), position) < manhattanDistance((i, j), position):
                                T_t[i][j][k][l] = 1
                            else:
                                if self.ghost == "terrified":
                                    T_t[i][j][k][l] = 8
                                elif self.ghost == "afraid":
                                    T_t[i][j][k][l] = 2
                                else:
                                    T_t[i][j][k][l] = 1
                T_t[i][j] = T_t[i][j]/np.sum(T_t[i][j])

        return T_t

    def observation_matrix(self, walls, evidence, position):
        """
        Optimized version of observation matrix O_t = P(e_t | X_t)
        
        Arguments:
            walls: The W x H grid of walls.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.
        
        Returns:
            The W x H observation matrix O_t.
        """
        width, height = walls.width, walls.height
        O_t = np.zeros((width, height))

        valid_positions = np.array([[x, y] for x in range(width) for y in range(height) if not walls[x][y]])

        true_distances = np.array([manhattanDistance(position, (x, y)) for x, y in valid_positions])
        noise = np.abs(evidence - true_distances)
        
        O_t[valid_positions[:, 0], valid_positions[:, 1]] = scipy.stats.norm.pdf(noise, loc=0, scale=1)
        
        O_t /= np.sum(O_t)

        return O_t

    def update(self, walls, belief, evidence, position):
        """Updates the previous ghost belief state

        b_{t-1} = P(X_{t-1} | e_{1:t-1})

        Arguments:
            walls: The W x H grid of walls.
            belief: The belief state for the previous ghost position b_{t-1}.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.

        Returns:
            The updated ghost belief state b_t as a W x H matrix.
        """
    
        T = self.transition_matrix(walls, position)
        O = self.observation_matrix(walls, evidence, position)

        width, height = walls.width, walls.height
        new_belief = np.zeros((width, height))

        valid_positions = np.array([[x, y] for x in range(width) for y in range(height) if not walls[x][y]])

        for x, y in valid_positions:
            transition_sum = np.sum(T[:, :, x, y] * belief)
            new_belief[x, y] = O[x, y] * transition_sum

        new_belief /= np.sum(new_belief)

        return new_belief

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
        
    def getLegalActions(self, position, walls):
        x, y = position
        actions = []

        if not walls[x][y + 1]:
            actions.append(Directions.NORTH)
        if not walls[x][y - 1]: 
            actions.append(Directions.SOUTH)
        if not walls[x - 1][y]:
            actions.append(Directions.WEST)
        if not walls[x + 1][y]:
            actions.append(Directions.EAST)

        if len(actions) == 0:
            actions.append(Directions.STOP)

        return actions
    
    def simulate_future_positions(self, position, walls, depth=3):
        possible_positions = {position}
        directions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]

        for _ in range(depth):
            next_positions = set()

            for current_pos in possible_positions:
                legal_actions = self.getLegalActions(current_pos, walls)

                for action in legal_actions:
                    if action == Directions.NORTH:
                        new_position = (current_pos[0], current_pos[1] + 1)
                    elif action == Directions.SOUTH:
                        new_position = (current_pos[0], current_pos[1] - 1)
                    elif action == Directions.EAST:
                        new_position = (current_pos[0] + 1, current_pos[1])
                    elif action == Directions.WEST:
                        new_position = (current_pos[0] - 1, current_pos[1])
                    else:
                        new_position = current_pos

                    next_positions.add(new_position)

            possible_positions = next_positions

        return list(possible_positions)

    def bfs_distance(self, start, goal, walls):
        """
        Calculate the shortest distance between start and goal using BFS.

        Arguments:
        - start: Tuple (x, y), starting position.
        - goal: Tuple (x, y), target position.
        - walls: Grid of booleans where True indicates a wall.

        Returns:
        - Integer representing the shortest path length, or float('inf') if no path exists.
        """
        if start == goal:
            return 0

        queue = deque([(start, 0)])  # Each entry is (position, distance)
        visited = set()
        visited.add(start)

        while queue:
            current, dist = queue.popleft()
            x, y = current

            # Explore neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (x + dx, y + dy)
                if neighbor == goal:
                    return dist + 1
                if (0 <= neighbor[0] < walls.width and
                    0 <= neighbor[1] < walls.height and
                    not walls[neighbor[0]][neighbor[1]] and
                    neighbor not in visited):
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return float('inf')  # No path found

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
        best_position = (0, 0)
        min_distance = float("inf")

        if self.target == -1 or eaten[self.target]:
            for ghost_index in range(len(beliefs)):
                max_belief = 0
                candidate_position = (0, 0)
                if eaten[ghost_index]:
                    continue
                for x in range(walls.width):
                    for y in range(walls.height):
                        belief_sum = 0
                        neighbor_count = 0
                        if walls[x][y]:
                            continue
                        x_min = max(0, x - 1)
                        x_max = min(walls.width, x + 2)
                        for neighbor_x in range(x_min, x_max):
                            y_min = max(0, y - 1)
                            y_max = min(walls.height, y + 2)
                            for neighbor_y in range(y_min, y_max):
                                if walls[neighbor_x][neighbor_y]:
                                    continue
                                belief_sum += beliefs[ghost_index][neighbor_x][neighbor_y]
                                neighbor_count += 1
                        if belief_sum > max_belief:
                            max_belief = belief_sum
                            candidate_position = (x, y)
                if manhattanDistance(position, candidate_position) < min_distance:
                    min_distance = manhattanDistance(position, candidate_position)
                    best_position = candidate_position
                    self.target = ghost_index
        else:
            max_belief = 0
            for x in range(walls.width):
                for y in range(walls.height):
                    if walls[x][y]:
                        continue
                    if beliefs[self.target][x][y] > max_belief:
                        max_belief = beliefs[self.target][x][y]
                        best_position = (x, y)

        priority_queue = PriorityQueue()
        priority_queue.push((position, [], 0), manhattanDistance(position, best_position))
        visited_positions = []
        target_beliefs = beliefs[self.target]
        weighted_x = target_beliefs * np.arange(walls.width)[:, np.newaxis]
        weighted_y = np.sum(target_beliefs * np.arange(walls.height), axis=1)
        candidate_best_position = best_position
        best_position = (int(np.sum(weighted_x)), int(np.sum(weighted_y)))

        if position == best_position or walls[best_position[0]][best_position[1]]:
            best_position = candidate_best_position

        direction_priorities = []
        direction_north = (best_position[0] - position[0], Directions.NORTH)
        direction_south = (position[0] - best_position[0], Directions.SOUTH)
        direction_east = (best_position[1] - position[1], Directions.EAST)
        direction_west = (position[1] - best_position[1], Directions.WEST)
        direction_priorities.append(direction_north)
        direction_priorities.append(direction_south)
        direction_priorities.append(direction_east)
        direction_priorities.append(direction_west)
        direction_priorities.sort(key=lambda direction: direction[0])

        while True:
            if priority_queue.isEmpty():
                return Directions.STOP
            _, ((current_x, current_y), current_path, cost) = priority_queue.pop()
            if (current_x, current_y) == best_position and current_path:
                best_action = current_path[0]
                break
            if (current_x, current_y) in visited_positions:
                continue
            visited_positions.append((current_x, current_y))
            potential_moves = []
            for _, action in direction_priorities:
                if action == Directions.NORTH:
                    potential_moves.append((current_x, current_y + 1, action))
                elif action == Directions.SOUTH:
                    potential_moves.append((current_x, current_y - 1, action))
                elif action == Directions.EAST:
                    potential_moves.append((current_x + 1, current_y, action))
                elif action == Directions.WEST:
                    potential_moves.append((current_x - 1, current_y, action))

            for next_x, next_y, action in potential_moves:
                if not walls[next_x][next_y] and (next_x, next_y):
                    priority_queue.push(
                        ((next_x, next_y), current_path + [action], cost + 1),
                        cost + 1 + manhattanDistance((next_x, next_y), best_position),
                    )
        return best_action

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
