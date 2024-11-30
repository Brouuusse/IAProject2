import numpy as np
import scipy.stats
from pacman_module.game import Agent, Directions, manhattanDistance, PriorityQueue
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

        n, p = 6, 0.5
        probabilities = [scipy.stats.binom.pmf(i, n, p) for i in range(n + 1)]

        for x in range(width):
            for y in range(height):
                if walls[x][y]:
                    continue

                true_distance = manhattanDistance(position, (x, y))
                k = abs(evidence - true_distance)
            
                if k <= 2:
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

            pq = PriorityQueue()
            pq.push((position, [], 0), manhattanDistance(position, best_position))
            visited = set()
            belief = beliefs[self.target]
            
            x_weighted = belief * np.arange(walls.width)[:, np.newaxis]
            y_weighted = np.sum(belief * np.arange(walls.height), axis=1)
            weighted_position = (int(np.sum(x_weighted)), int(np.sum(y_weighted)))
            if position == weighted_position or walls[weighted_position[0]][weighted_position[1]]:
                best_position = best_position
            else:
                best_position = weighted_position

            direction_priorities = [
                (best_position[0] - position[0], Directions.NORTH),
                (position[0] - best_position[0], Directions.SOUTH),
                (best_position[1] - position[1], Directions.EAST),
                (position[1] - best_position[1], Directions.WEST),
            ]
            direction_priorities.sort(key=lambda x: x[0])

            while not pq.isEmpty():
                _, ((x, y), path, cost) = pq.pop()
                if (x, y) == best_position and path:
                    best_action = path[0]
                    break

                if (x, y) in visited:
                    continue
                visited.add((x, y))

                directions = []
                for _, action in direction_priorities:
                    if action == Directions.NORTH:
                        directions.append((x, y + 1, action))
                    elif action == Directions.SOUTH:
                        directions.append((x, y - 1, action))
                    elif action == Directions.EAST:
                        directions.append((x + 1, y, action))
                    elif action == Directions.WEST:
                        directions.append((x - 1, y, action))

                for xn, yn, action in directions:
                    if not walls[xn][yn] and (xn, yn) not in visited:
                        pq.push(((xn, yn), path + [action], cost + 1), cost + 1 + manhattanDistance((xn, yn), best_position))

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
