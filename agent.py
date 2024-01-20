import random
from pygame.locals import *
from model import OthelloModel
import torch
import numpy as np
from board_utils import BoardUtils
import math


class RandomAgent:
    """
    Agent that makes random moves.
    """
    @staticmethod
    def choose_move(board, valid_move_func):
        """
        Choose a random move.
        Args:
            board (list[list[str]]): the board
            valid_move_func (function): function to check if a move is valid
        Returns:
            tuple: move
        """
        valid_moves = [(x, y) for x in range(len(board)) for y in range(len(board[0])) if valid_move_func(x, y)]
        if valid_moves:
            return random.choice(valid_moves)
        return None

class PlayerAgent:
    """
    Agent that takes moves from the player.
    """
    @staticmethod
    def choose_move(valid_move_func, pygame, cell_size):
        """
        Wait for player to choose a move.
        Args:
            valid_move_func (function): function to check if a move is valid
            pygame (pygame): pygame instance
            cell_size (int): number of pixels per cell
        Returns:
            tuple: move
        """
        while True:
            for event in pygame.event.get():
                if event.type == MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if y > 40:
                        row, col = x // cell_size, (y - 40) // cell_size
                        if valid_move_func(row, col):
                            return row, col
                elif event.type == QUIT:
                    pygame.quit()
                    exit()

class eOthelloAgent:
    """
    Agent that takes moves from the eOthello website.
    """
    @staticmethod
    def choose_move(game, curr_player, eo):
        """
        Wait for eOthello bot move.
        Args:
            game (OthelloGame): game instance
            curr_player (str): current player (X or O)
        Returns:
            tuple: move
        """
        board, move, last_player = eo.calculate_board_state()
        while curr_player != last_player:
            board, move, last_player = eo.calculate_board_state()
        game.board = board
        game.current_player = 'X' if curr_player == 'O' else 'O'
        return move

class MCTSAgent:
    """
    MCTS agent. Uses minimax at the end of the game.
    """
    class MCTSNode:
        """
        A simple class for MCTS nodes.
        """
        def __init__(self, game, last_move, parent=None):
            self.game = game.duplicate()
            if last_move is not None:
                self.game.make_move(last_move[0], last_move[1])
            self.last_move = last_move
            self.parent = parent
            self.children = []
            self.total = 0
            self.valuations = 0
            self.moves_left = self.game.get_valid_moves()
            random.shuffle(self.moves_left)
        
        def select_child(self, no_exploration=False):
            """
            Select a child node using the UCB1 formula.
            Returns:
                MCTSNode: child node
            """
            if no_exploration:
                alpha = 0
            else:
                alpha = 3.92 # 2*1.4^2, standard value
            ucb1 = np.array([c.valuations / c.total + math.sqrt((alpha * math.log(self.total)) / c.total) for c in self.children])
            return self.children[np.argmax(ucb1)]


    def __init__(self, model_path):
        self.model = OthelloModel()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)

    def board_to_state(self, board, turn):
        """
        Convert a board to an input to the model.
        Args:
            board (list[list[str]]): the board
            turn (int): 0 if O's turn, 1 if X's turn
        Returns:
            torch.FloatTensor: model input
        """
        board = np.array(board)
        state = np.zeros((1, 5, 8, 8))
        state[0, 0] = (board == 'X').astype(int)
        state[0, 1] = (board == 'O').astype(int)
        state[0, 2] = (board == ' ').astype(int)
        state[0, 3] = turn
        state[0, 4] = BoardUtils.weights_heuristic(board)
        return torch.FloatTensor(state).to(self.device)
    
    def choose_move(self, game, depth=100, threshold=14):
        """
        Choose a move using MCTS or minimax.
        Args:
            game (OthelloGame): game instance
            depth (int): number of iterations for MCTS
            threshold (int): number of moves left to switch to minimax
        Returns:
            tuple: move
        """
        if len(game.get_valid_moves()) == 1:
            return game.get_valid_moves()[0]
        if BoardUtils.moves_remaining_from_board(game.board) <= threshold:
            minimax_agent = MinimaxAgent()
            return minimax_agent.choose_move(game, 999)
        root = self.MCTSNode(game, None)
        return self.search(root, depth)
    
    def search(self, root, depth):
        """
        Run MCTS search.
        Args:
            root (MCTSNode): root node
            depth (int): number of iterations
        Returns:
            tuple: move
        """
        for _ in range(depth):
            node = root
            # Selection
            while len(node.moves_left) == 0 and len(node.children) != 0:
                node = node.select_child()
            # Expansion
            if len(node.moves_left) != 0:
                move = node.moves_left.pop()
                new_node = self.MCTSNode(node.game, move, node)
                node.children.append(new_node)
                node = new_node
            # Simulation
            if node.game.is_over():
                winner = node.game.winner()
                if winner == 'X':
                    valuation = 1
                elif winner == 'O':
                    valuation = 0
                else:
                    valuation = 0.5
            else:
                state = self.board_to_state(node.game.board, 1 if node.game.current_player == 'X' else 0)
                with torch.no_grad():
                    output = self.model(state)
                valuation = output[0][0].item()
            # Backpropagation
            while node != None:
                node.total += 1
                if node.game.current_player == 'X':
                    node.valuations += 1 - valuation
                else:
                    node.valuations += valuation
                node = node.parent
        return root.select_child(no_exploration=True).last_move

class MinimaxAgent:
    """
    A minimax agent with alpha-beta pruning.
    """
    def choose_move(self, game, depth):
        """
        Choose a move using minimax.
        Args:
            game (OthelloGame): game instance
            depth (int): search tree depth
        Returns:
            tuple: move
        """
        self.minimax_calls = 0
        _, move = self.minimax(game, depth, -999999, 999999)
        return move

    def minimax(self, game, depth, alpha, beta):
        """
        Recursive minimax search.
        Args:
            game (OthelloGame): game instance
            depth (int): search tree depth
            alpha (float): alpha value
            beta (float): beta value
        Returns:
            tuple: (score, move)
        """
        self.minimax_calls += 1
        if game.is_over():
            return BoardUtils.winner_from_board(game.board), None
        if depth == 0:
            return BoardUtils.weights_heuristic(game.board), None
        
        best_score = -999999 if game.current_player == 'X' else 999999
        best_move = None
        for move in game.get_valid_moves():
            game_dupe = game.duplicate()
            game_dupe.make_move(move[0], move[1])
            score, _ = self.minimax(game_dupe, depth-1, alpha, beta)
            if (score > best_score and game.current_player == 'X') or (score < best_score and game.current_player == 'O'):
                best_score = score
                best_move = move
            alpha = max(alpha, score) if game.current_player == 'X' else alpha
            beta = min(beta, score) if game.current_player == 'O' else beta
            if beta <= alpha:
                break
        return best_score, best_move

