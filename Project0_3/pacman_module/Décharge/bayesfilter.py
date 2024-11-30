import numpy as np
import scipy.stats
from pacman_module.game import Agent, Directions, manhattanDistance

#+ DICTIONNAIRE (retenir mvt pour ne pas refaire les mêmes "erreurs" (e.g : 4 nuages de points, Pacman au milieu))

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
        W, H = walls.width, walls.height
        T_t = np.zeros((W, H, W, H))

        def surrounding_tiles(x, y):
            return [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]

        def out_of_bounds(x, y, max_x, max_y):
            return x < 0 or y < 0 or x >= max_x or y >= max_y
        
        if (self.ghost == "fearless"):
            fear_factor = 0.0

        elif (self.ghost == "afraid"):
            fear_factor = 1.0

        else:
            fear_factor = 3.0


        for i in range(W):
            for j in range(H):
                if walls[i][j]:
                    continue
                weight = 0
                neighbouring_tiles = surrounding_tiles(i, j)

                for k, l in neighbouring_tiles:
                    if not (out_of_bounds(k, l, W, H) or walls[k][l]):
                        if manhattanDistance((k, l), position) < manhattanDistance((i, j), position):
                            weight += 2**(fear_factor)
                        else:
                            weight += 1

                for k, l in neighbouring_tiles:
                    if not (out_of_bounds(k, l, W, H) or walls[k][l]):
                        if manhattanDistance((k, l), position) < manhattanDistance((i, j), position):
                            T_t[i, j, k, l] = (2**(fear_factor)) / weight
                        else:
                            T_t[i, j, k, l] = 1.0 / weight

            T_t[i, j] /= np.sum(T_t[i, j])

        return T_t
    

    def observation_matrix(self, walls, evidence, position):
        """
        Builds the observation matrix O_t = P(e_t | X_t)
    
        Arguments:
            walls: The W x H grid of walls.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.
    
        Returns:
            The W x H observation matrix O_t.
        """
        width, height = walls.width, walls.height
        O_t = np.zeros((width, height))

        n, p = 4, 0.5
        probabilities = [scipy.stats.binom.pmf(i, n, p) for i in range(n + 1)]

        for x in range(width):
            for y in range(height):
                if walls[x][y]:
                    continue

                true_distance = manhattanDistance(position, (x, y))
                k = abs(evidence - true_distance)
            
                if k <= n:
                    O_t[x, y] = probabilities[k]

        O_t /= np.sum(O_t)

        return O_t
    
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
        T = self.transition_matrix(walls, position)
        O = self.observation_matrix(walls, evidence, position)

        width, height = walls.width, walls.height
        new_belief = np.full((width, height), 0.0)

        for x in range(width):
            for y in range(height):
                if walls[x][y]:
                    continue

                transition_sum = 0
                for i in range(width):
                    for j in range(height):
                        if walls[i][j]:
                            continue

                        transition_sum += T[i][j][x][y] * belief[i][j]

                new_belief[x][y] = O[x][y] * transition_sum

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

        legal_actions = self.getLegalActions(position, walls)
    
        max_belief_positions = []

        for i, belief in enumerate(beliefs):
            if not eaten[i]:
                max_belief_value = np.max(belief)
                max_belief_positions.append([(x, y) for x in range(belief.shape[0]) for y in range(belief.shape[1]) if belief[x, y] == max_belief_value])

        best_position = None
        if max_belief_positions:
            all_max_positions = [pos for sublist in max_belief_positions for pos in sublist]
            best_position = min(all_max_positions, key=lambda pos: manhattanDistance(position, pos))

        best_action = None
        if best_position:
            x, y = position
            best_x, best_y = best_position
        
            if best_x < x:
                best_action = Directions.WEST
            elif best_x > x:
                best_action = Directions.EAST
            elif best_y < y:
                best_action = Directions.SOUTH
            elif best_y > y:
                best_action = Directions.NORTH

        if best_action in legal_actions:
            return best_action
        else:
            return Directions.STOP

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
