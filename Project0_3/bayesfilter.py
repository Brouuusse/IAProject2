import numpy as np
import scipy.stats
from pacman_module.game import Agent, Directions, manhattanDistance

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



    def _get_action(self, walls, beliefs, eaten, position):
        if self.target == -1 or eaten[self.target]:
            ghost_positions = []
            for i, belief in enumerate(beliefs):
                if not eaten[i]:
                    max_belief_value = np.max(belief)
                    if max_belief_value > 0:
                        max_positions = [(x, y) for x in range(belief.shape[0]) for y in range(belief.shape[1]) if belief[x, y] == max_belief_value]
                        for pos in max_positions:
                            ghost_positions.append((i, pos))

            best_position = None
            best_ghost = None
            if ghost_positions:
                best_ghost, best_position = min(ghost_positions, key=lambda g: manhattanDistance(position, g[1]))

            self.target = best_ghost
        else:
            max_belief = 0
            for j in range(walls.width):
                for k in range(walls.height):
                    if walls[j][k]:
                        continue
                    if beliefs[self.target][j][k] > max_belief:
                        max_belief = beliefs[self.target][j][k]
                        best_position = (j, k)

        future_positions = self.simulate_future_positions(position, walls, depth=3)
        best_future_position = min(future_positions, key=lambda p: manhattanDistance(p, best_position))
        if manhattanDistance(position, best_position) <= manhattanDistance(best_future_position, best_position):
            return Directions.STOP
        
        legal_actions = self.getLegalActions(position, walls)
        best_action = None
        min_distance = float('inf')

        for action in legal_actions:
            if action == Directions.NORTH:
                new_position = (position[0], position[1] + 1)
            elif action == Directions.SOUTH:
                new_position = (position[0], position[1] - 1)
            elif action == Directions.EAST:
                new_position = (position[0] + 1, position[1])
            elif action == Directions.WEST:
                new_position = (position[0] - 1, position[1])
            else:
                new_position = position

            distance = manhattanDistance(new_position, best_position)

            if distance < min_distance:
                min_distance = distance
                best_action = action

        if not best_action:
            best_action = Directions.STOP
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
