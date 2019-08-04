#tic tac toe gem yeah

# Imports
import time
import random
import numpy as np
import pickle as pkl
from collections import defaultdict


# possibly connect4. Only have to change game state and the diagonals
# one against one
# Turn based
# Min Cal

class GameState:
    """ Class for representing a state of the tic-tac-toe game. """
    _player_symbols = {0: " ", 1: "X", 2: "O"}

    def __init__(self, board=None, prev_player=2):
        """
        Initializes a GameState.
        If the given board parameter is None, initialize an empty board.

        Parameters
        ----------
        board: numpy.ndarray, shape=(3, 3) or None
            The cells of the tic-tac-toe board, with values either 0 (unoccupied) or a player number (1 or 2)
        prev_player: int
            1 or 2, the number of the player who is NOT about to make a move
        """
        # STUDENT CODE HERE
        self.board = board
        if self.board is None:
            self.board = np.zeros([3, 3])

        # The person whom is not making a move. In this case it is the 2nd person
        self.prev_player = prev_player

    def get_next_player(self):
        """
        Return the number of the next player, the one who is about to make a move.

        Returns
        -------
        int
        """
        # STUDENT CODE HERE
        if self.prev_player == 1:
            return 2
        else:
            return 1

    def has_ended(self):
        """
        Determine if the game has ended.

        If a row, column, or diagonal has been entirely filled by a player, return their player number
        If there is a draw (all cells are filled, no winner), return -1
        If the game is still going, return 0

        Returns
        -------
        int
        """
        """
        Psuedo Code:
        Goes through the array or board and checks to see if there is a win
        Row = (0 to 2)
        Col = (0 to 2)
        """
        # STUDENT CODE HERE
        # if 0 in self.board:
        #    return 0

        # Checking for rows

        for r in self.board:
            p_cnts = [0, 0]
            for num in r:
                if num == 1:
                    p_cnts[0] += 1
                elif num == 2:
                    p_cnts[1] += 1
            if p_cnts[0] == 3:
                return 1
            if p_cnts[1] == 3:
                return 2

        # Checking for the columns

        for c in range(self.board.shape[1]):
            p_cnts = [0, 0]
            column = self.board[:, c]
            for r in column:
                if r == 1:
                    p_cnts[0] += 1
                elif r == 2:
                    p_cnts[1] += 1
            if p_cnts[0] == 3:
                return 1
            if p_cnts[1] == 3:
                return 2

        # checking diagonals

        if (self.board[0, 0] == self.board[1, 1]) and self.board[2, 2] == self.board[0, 0]:
            return self.board[0, 0]

        if (self.board[2, 0] == self.board[1, 1]) and self.board[0, 2] == self.board[2, 0]:
            return self.board[2, 0]

        if np.all(self.board != 0):
            return -1
        return 0

    def get_possible_moves(self):
        """
        Return a list of all possible tic-tac-toe moves that the next player can make
        from this current state.

        Each move can be defined as a tuple of two integers, the row and column of the cell to choose,
        which are equivalent to the index of the cell in the board.

        i.e. For the following board,

         1 | 0 | 2
        -----------
         0 | 1 | 2
        -----------
         1 | 2 | 0

        Return [(0, 1), (1, 0), (2, 2)] or some variant.

        Returns
        -------
        List[tuple[int]]
            List of tuple pairs corresponding to possible moves.
        """
        # STUDENT CODE HERE
        possible_moves = []
        for r in range(self.board.shape[0]):
            for c in range(self.board.shape[1]):
                if self.board[r, c] == 0:
                    possible_moves.append((r, c))

        # This is the easier part I believe
        return possible_moves

    def get_new_state(self, move):
        """
        Create a new GameState for the move played, with new board/player values.
        (Remember to COPY the board if you wish to change it.)

        Parameters
        ----------
        move: tuple[int]
            A tuple (r, c) of the move played by the next_player.

        Returns
        -------
        GameState
            The new GameState.
        """
        # STUDENT CODE HERE
        if self.prev_player == 2:
            next_player = 1
        else:
            next_player = 2

        newboard = np.copy(self.board)
        newboard[move[0], move[1]] = next_player
        return GameState(newboard, next_player)

    def __repr__(self):
        return self.board, self.prev_player

    def __str__(self):
        # Prettify board
        board_display = ("\n" + "-" * 11 + "\n").join(
            [" " + " | ".join([GameState._player_symbols[x] for x in row]) + " " for row in self.board])
        string = board_display + "\nNext player: " + str(self.get_next_player())
        return string


class Node:
    """ Nodes that build up the tree to search through. """

    def __init__(self, gamestate):
        """
        Initializes a Node with a GameState and empty dictionary of children.
        #it is just the init function
        Parameters
        ----------
        gamestate: GameState
            The GameState associated with this node
        """
        # STUDENT CODE HERE
        self.gamestate = gamestate
        self.children = {}

    def get_children(self):
        """
        Expands the tree by getting all possible moves from self.gamestate,
        making new Nodes associated with each of these,
        and adding them to the self.children dictionary mapping move (tuple) to Node.

        Returns
        -------
        None
        """
        # STUDENT CODE HERE
        pos_moves = self.gamestate.get_possible_moves()
        for tup in pos_moves:
            # get the next state
            # make node
            # Add node to children
            newstate = self.gamestate.get_new_state(tup)
            val = Node(newstate)
            self.children[tup] = val


import collections
import numpy
import random


class MonteCarlo:
    """ Class that handles Monte Carlo searches over the game tree. """

    def __init__(self, initial_time=10, calc_time=2, c=1.4, max_expansions=1, max_moves=None):
        """
        Initializes a MonteCarlo instance, which involves initializing the following instance variables:
         - search parameters, as passed to the constructor
         - self.history, a list containing the root node (Node corresponding to the initial, empty GameState)
         - self.wins, self.plays, empty defaultdicts mapping Nodes to ints that store win/total play records

        Then run an initial search.

        Parameters
        ----------
        initial_time: float
            Seconds for which to run the initial search
        calc_time: float
            Seconds for which to "think" before the computer makes a move
        c: float
            Exploration parameter for calculation of UCT
        max_expansions: int or None
            Maximum number of nodes to expand before entering pure simulation phase, or None if unlimited
        max_moves: int or None
            Maximum number of moves to try per simulation, or None if unlimited
        """
        # STUDENT CODE HERE
        self.initial_time = initial_time
        self.calc_time = calc_time
        self.c = c
        self.max_expansions = max_expansions
        self.max_moves = max_moves
        # Let's ask to see if I am getting the right values for the
        self.history = [Node(GameState())]
        self.wins = collections.defaultdict(int)
        self.plays = collections.defaultdict(int)

        self.run_search(initial=True)

    def update(self, move):
        """
        When a move is made, get the next GameState and add its node to the game history.
        Return None if the move is invalid.

        Parameters
        ----------
        move: tuple[int]
            The move last made

        Returns
        -------
        Node
            The GameState that follows the move, now the last element of self.history
        """
        # STUDENT CODE HERE
        if len(self.history[-1].children) == 0:
            self.history[-1].get_children()
        if move in self.history[-1].children:
            newNode = self.history[-1].children[move]
            self.history.append(newNode)
            return newNode
        return None

    def win_prob(self, node):
        """
        Calculate the win probability associated with the given Node.

        Parameters
        ----------
        node: Node

        Returns
        -------
        float
            The node's wins divided by the node's plays, or 0 if there have been no plays.
        """
        if self.plays[node] == 0:
            return 0
        return self.wins[node] / self.plays[node]

    def uct(self, node, parent):
        """
        Calculates a node's UCT.

        Parameters
        ----------
        node: Node
        parent: Node
            The node for which "node" is a direct child.

        Returns
        -------
        float
        """
        # Calculate exploitation (win ratio)
        # STUDENT CODE HERE
        exploitation = self.win_prob(node)

        # Calculate exploration (how unexplored it is)
        # STUDENT CODE HERE
        if self.plays[node] == 0:
            return float("inf")
        exploration = self.c * np.sqrt(np.log(self.plays[parent]) / self.plays[node])

        return exploitation + exploration

    def search(self):
        """
        Perform one round of MCTS.

        1. Initialize a temporary history
        2. While the current node is not a terminal node (has_ended() returns 0), and we haven't yet reached the move limit:
            - Selection: choose moves by maximum UCT value for as long as we are
                         in explored territory, adding to the temporary history
            - Expansion: expand the current node, choosing a random move and adding to
                         the temporary history until the maximum number of expansions is reached
            - Simulation: choose random moves until the end of the game is reached
                          (do not add to the temporary history)
        3. Backpropagation: backpropagate through the temporary history and increment the number of plays for each node,
                            as well as the number of wins if its prev_player matches the game's overall winner or
                            if the game is a draw (by a lesser amount)
                            (the move that leads to this node is more likely to be picked by the previous player)
        """
        # STUDENT CODE HERE
        # self.history - already created
        temphist = [self.history[-1]]
        current = temphist[-1]

        moves = 0
        expansion = 0
        # Just adding this to add
        # Just trying this
        while current.gamestate.has_ended() == 0 and (self.max_moves == None or moves < self.max_moves):
            # Selection phase
            if self.plays[current] > 0:
                # Know there is already children
                # Turning this into a list
                keylist = list(current.children.keys())
                UCT = -float("inf")
                keepkey = None
                for key in keylist:
                    node = current.children[key]
                    uctval = self.uct(node, current)
                    if uctval > UCT:
                        UCT = uctval
                        keepkey = key
                temphist.append(current.children[keepkey])
                current = temphist[-1]

            # Does the expansion and simulation parts of the code

            else:
                if len(current.children) == 0:
                    current.get_children()
                randkey = random.choice(list(current.children.keys()))
                node = current.children[randkey]

                if self.max_expansions is None or expansion < self.max_expansions:
                    temphist.append(node)
                    expansion += 1
                current = node

            moves += 1

            # Loop throught temphist and check stuff
        # If winner matches previous player, then add 3 to wins or 1 if draw
        # add 1 to place

        # Believe I am now doing the simulation

        # backprop time
        for nod in temphist:
            self.plays[nod] += 1
            if current.gamestate.has_ended() == nod.gamestate.prev_player:
                self.wins[nod] += 3
            elif current.gamestate.has_ended() == -1:
                self.wins[nod] += 1
            else:
                self.wins[nod] += 0

    def run_search(self, max_time=None, initial=False):
        """
        Run search for the given amount of time.

        Parameters
        ----------
        max_time: float
            Number of seconds for which to run the search
        initial: bool
            Defines what max_time should default to if it is given as None
            Choose self.initial_time if initial is True, else self.calc_time

        Returns
        -------
        None
        """
        if max_time is None:
            max_time = self.initial_time if initial else self.calc_time
        if initial: print("Running initial search...")

        t0 = time.time()
        sims = 0
        while time.time() - t0 < max_time:
            self.search()
            sims += 1

        print("Ran simulations:", sims)

    def make_computer_move(self):
        """
        Run search for calc_time seconds, choose the best (highest win probability) move from the
        set of possible moves, update the history, and return the status of the new GameState.

        Returns
        -------
        int
            Status of the game, either the winner number, -1 if the game is drawn, or 0 otherwise
        """
        # STUDENT CODE HERE

        self.run_search(max_time=self.calc_time)
        # print(self.history[-1].gamestate, self.plays[self.history[-1]])
        keylist = self.history[-1].children.keys()
        # print(len(keylist))
        high_winprob = -1
        keepkey = None
        for key in keylist:
            node = self.history[-1].children[key]
            prob_val = self.win_prob(node)
            if prob_val > high_winprob:
                high_winprob = prob_val
                keepkey = key
        node = self.update(keepkey)
        return node.gamestate.has_ended()

    def make_player_move(self):
        """
        Get player input (repeat as long as input is invalid), and make a move.

        Returns
        -------
        int
            Status of the game, either the winner number, -1 if the game is drawn, or 0 otherwise
        """
        new_node = None
        while new_node is None:
            try:
                move = tuple(int(x) for x in input("Enter move: ").split())
                new_node = self.update(move)
            except:
                print("Invalid input. Enter a move as 'r c' (without quotes, separated by a single space)")
        return new_node.gamestate.has_ended()


def play_game(player1=True, player2=False):
    """
    Plays a game, displaying the board each turn.

    Parameters
    ----------
    player1: bool
        Whether or not the first player (X) should be human-controlled.
    player2: bool
        Whether or not the second player (O) should be human-controlled.
    """

    game = MonteCarlo(max_expansions=None)

    while True:
        # Player 1 move
        if player1:
            x = game.make_player_move()
        else:
            x = game.make_computer_move()
        print(game.history[-1].gamestate)
        if x != 0: break

        # Player 2 move
        if player2:
            x = game.make_player_move()
        else:
            x = game.make_computer_move()
        print(game.history[-1].gamestate)
        if x != 0: break

    if game.history[-1].gamestate.has_ended() == -1:
        print("Game ended in a draw.")
    else:
        print(game.history[-1].gamestate.prev_player, " won.")


#Should have the computer play the game
# Computer vs Computer: should tie
play_game(False, False)