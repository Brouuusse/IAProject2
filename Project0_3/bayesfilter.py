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
                                if self.ghostType == "terrified":
                                    T_t[i][j][k][l] = 8
                                elif self.ghostType == "afraid":
                                    T_t[i][j][k][l] = 2
                                else:
                                    T_t[i][j][k][l] = 1
                T_t[i][j] = T_t[i][j]/np.sum(T_t[i][j])

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

        for x in range(width):
            for y in range(height):
                if walls[x][y]:
                    continue

                true_distance = manhattanDistance(position, (x, y))

                noise_probability = scipy.stats.binom.pmf(evidence - true_distance + n, n, p)
                O_t[x, y] = noise_probability

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
        new_belief = np.zeros((width, height))

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
        """
        Détermine les mouvements possibles pour Pacman.

        Arguments:
            position: La position actuelle de Pacman (x, y).
        walls: Le W x H grid des murs.

        Returns:
            Une liste de mouvements possibles (Directions.NORTH, Directions.SOUTH, etc.).
        """
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
        direction_deltas = {
            Directions.NORTH: (0, 1),
            Directions.SOUTH: (0, -1),
            Directions.EAST: (1, 0),
            Directions.WEST: (-1, 0),
            Directions.STOP: (0, 0)
        }

        legal_actions = self.getLegalActions(position, walls)

        best_action = Directions.STOP
        min_distance = float('inf')

        for i, belief in enumerate(beliefs):
            if eaten[i]:
                continue

            max_belief_position = np.unravel_index(np.argmax(belief), belief.shape)

            for action in legal_actions:
                dx, dy = direction_deltas[action]
                new_position = (position[0] + dx, position[1] + dy)

                distance = manhattanDistance(new_position, max_belief_position)

                if distance < min_distance:
                    min_distance = distance
                    best_action = action

        # Retourne la meilleure action
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
