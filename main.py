import pygame
from pygame.locals import *
from agent import RandomAgent as rand
from agent import PlayerAgent as player
from agent import MCTSAgent as mcts
from agent import eOthelloAgent as eothello
import copy
import random
import read_eothello
import numpy as np
import datetime

GAME_SIZE = 8
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 128, 0)
CELL_SIZE = 60
WIDTH, HEIGHT = GAME_SIZE * CELL_SIZE, GAME_SIZE * CELL_SIZE
pygame.init()

class Othello:
    """
    A class for all Othello game logic.
    """
    def __init__(self, board=None, current_player=None):
        if board is None:
            self.board = [[' ' for _ in range(GAME_SIZE)] for _ in range(GAME_SIZE)]
            self.board[GAME_SIZE // 2 - 1][GAME_SIZE // 2 - 1], self.board[GAME_SIZE // 2][GAME_SIZE // 2] = 'O', 'O'
            self.board[GAME_SIZE // 2 - 1][GAME_SIZE // 2], self.board[GAME_SIZE // 2][GAME_SIZE // 2 - 1] = 'X', 'X'
        else:
            self.board = board
        if current_player is None:
            self.current_player = 'X' # X is black, O is white
        else:
            self.current_player = current_player
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    
    def opponent(self):
        """
        Returns the opposite of the current player.
        Returns:
            str: opposite of current player (X or O)
        """
        return 'X' if self.current_player == 'O' else 'O'
    
    def valid_move(self, x, y):
        """
        Checks if a move is valid.
        Args:
            x (int): x cell
            y (int): y cell
        Returns:
            bool: whether the move is valid
        """
        if self.board[x][y] != ' ':
            return False
        for dx, dy in self.directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GAME_SIZE and 0 <= ny < GAME_SIZE and self.board[nx][ny] == self.opponent():
                while 0 <= nx < GAME_SIZE and 0 <= ny < GAME_SIZE and self.board[nx][ny] != ' ':
                    nx += dx
                    ny += dy
                    if 0 <= nx < GAME_SIZE and 0 <= ny < GAME_SIZE and self.board[nx][ny] == self.current_player:
                        return True
        return False
    
    def make_move(self, x, y):
        """
        Makes a move.
        Args:
            x (int): x cell
            y (int): y cell
        Returns:
            bool: whether the move was valid
        """
        if not self.valid_move(x, y):
            return False        
        self.board[x][y] = self.current_player
        for dx, dy in self.directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GAME_SIZE and 0 <= ny < GAME_SIZE and self.board[nx][ny] == self.opponent():
                to_flip = []
                while 0 <= nx < GAME_SIZE and 0 <= ny < GAME_SIZE and self.board[nx][ny] != ' ':
                    if self.board[nx][ny] == self.current_player:
                        for fx, fy in to_flip:
                            self.board[fx][fy] = self.current_player
                        break
                    to_flip.append((nx, ny))
                    nx += dx
                    ny += dy
        
        self.current_player = self.opponent()
        if not self.has_valid_moves():
            self.current_player = self.opponent()
        return True
    
    def is_full(self):
        """
        Checks if the board is full of coins.
        Returns:
            bool: whether the board is full
        """
        for row in self.board:
            for cell in row:
                if cell == ' ':
                    return False
        return True
    
    def winner(self):
        """
        Returns the winner of the game.
        Returns:
            str: winner (X or O)
        """
        x_count = sum(row.count('X') for row in self.board)
        o_count = sum(row.count('O') for row in self.board)
        if x_count > o_count:
            return 'X'
        elif x_count < o_count:
            return 'O'
        else:
            return 'Tie'
        
    def has_valid_moves(self):
        """
        Checks if the current player has any valid moves.
        Returns:
            bool: whether the current player has any valid moves
        """
        if self.is_full():
            return False
        for x in range(GAME_SIZE):
            for y in range(GAME_SIZE):
                if self.valid_move(x, y):
                    return True
        return False
    
    def is_over(self):
        """
        Checks if the game is over.
        Returns:
            bool: whether the game is over
        """
        if not self.has_valid_moves():
            self.current_player = self.opponent()
            if not self.has_valid_moves():
                return True
            self.current_player = self.opponent()
        return False
    
    def get_valid_moves(self):
        """
        Returns all valid moves.
        Returns:
            list[tuple]: list of valid moves
        """
        moves = []
        for x in range(GAME_SIZE):
            for y in range(GAME_SIZE):
                if self.valid_move(x, y):
                    moves.append((x, y))
        return moves
    
    def duplicate(self):
        """
        Returns a duplicate of the current game.
        Returns:
            Othello: duplicate of the current game
        """
        board = [[' ' for _ in range(GAME_SIZE)] for _ in range(GAME_SIZE)]
        for x in range(GAME_SIZE):
            for y in range(GAME_SIZE):
                board[x][y] = self.board[x][y]
        return Othello(copy.deepcopy(self.board), self.current_player)
    
    def __str__(self):
        """
        Converts the board into a printable string.
        Returns:
            str: printable board string
        """
        s = '  '
        for y in range(GAME_SIZE):
            s += str(y) + ' '
        s += '\n'
        for x in range(GAME_SIZE):
            s += str(x) + ' '
            for y in range(GAME_SIZE):
                s += self.board[x][y] + ' '
            s += '\n'
        return s
    

class OthelloGame:
    """
    A class for running an Othello game.
    """
    @staticmethod
    def play(model_path, display=True, game=None):
        """
        Runs a game of Othello.
        Args:
            model_path (str): path to torch model
            display (bool): whether to display the game with a GUI
            game (Othello): game instance
        Returns:
            str: winner (X or O)
            list[list[str]]: list of all board states
            list[int]: list of turns (1 for X, 0 for O)
            float: black-to-white ratio
        """
        if display:
            screen = pygame.display.set_mode((WIDTH, HEIGHT + 40))
            pygame.display.set_caption('Othello')
            font = pygame.font.SysFont(None, 36)
        states = []
        turns = []
        if game is None:
            game = Othello()
        mcts_agent = mcts(model_path)
        eo = read_eothello.eOthelloScraper()
        
        # Main game loop
        running = True
        while running:
            # Display the current state of the game
            if display:
                screen.fill(GREEN)
                if not game.is_over():
                    color_name = "Black" if game.current_player == 'X' else "White"
                    status_msg = f"{color_name}'s turn"
                    label = font.render(status_msg, 1, BLACK)
                    screen.blit(label, (10, 5))
                else:
                    winner = game.winner()
                    if winner == 'Tie':
                        status_msg = "It's a tie!"
                    else:
                        color_name = "Black" if winner == 'X' else "White"
                        status_msg = f"{color_name} wins!"
                    label = font.render(status_msg, 1, BLACK)
                    screen.blit(label, (10, 5))       
                for x in range(0, WIDTH, CELL_SIZE):
                    pygame.draw.line(screen, BLACK, (x, 40), (x, HEIGHT + 40))
                for y in range(40, HEIGHT + 40, CELL_SIZE):
                    pygame.draw.line(screen, BLACK, (0, y), (WIDTH, y))
                for x in range(GAME_SIZE):
                    for y in range(GAME_SIZE):
                        if game.board[x][y] == 'X':
                            pygame.draw.circle(screen, BLACK, (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2 + 40), CELL_SIZE // 2 - 5)
                        elif game.board[x][y] == 'O':
                            pygame.draw.circle(screen, WHITE, (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2 + 40), CELL_SIZE // 2 - 5)
                        elif game.valid_move(x, y):
                            color = BLACK if game.current_player == 'X' else WHITE
                            pygame.draw.circle(screen, color, (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2 + 40), 5)
                pygame.display.flip()

            # Check if the game is over
            if game.is_over():
                if display:
                    pygame.time.wait(100)
                running = False
                continue
            
            # Get the next move from the player or computer
            if game.current_player == 'O':
                if game.has_valid_moves():
                    # move = rand.choose_move(game.board, game.valid_move)
                    move = mcts_agent.choose_move(game, depth=3000, threshold=14)
                    # eo.click_cell(move[0], move[1])
                    if not game.make_move(*move):
                        print("Invalid computer move")
                        exit()
            else:
                if game.has_valid_moves():
                    move = player.choose_move(game.valid_move, pygame, CELL_SIZE)
                    # move = eothello.choose_move(game, game.current_player, eo)
                    if not game.make_move(*move):
                        print("Invalid player move")
                        exit()

            # Save game info for generating training data
            if not game.is_over():
                states.append(copy.deepcopy(game.board))
                turns.append(1 if game.current_player == 'X' else 0)

        if display:
            pygame.quit()

        num_black = sum(row.count('X') for row in game.board)
        num_white = sum(row.count('O') for row in game.board)
        return game.winner(), states, turns, num_black / (num_black + num_white)

    @staticmethod
    def play_repeated_games(model_path):
        """
        Plays games on repeat, printing the win rate.
        Args:
            model_path (str): path to torch model
        """
        black_wins = 0
        white_wins = 0
        ties = 0
        while True:
            winner, _, _, ratio = OthelloGame.play(model_path, False)
            if winner == 'X':
                black_wins += 1
            elif winner == 'O':
                white_wins += 1
            else:
                ties += 1
            print("Winner:", "Black" if winner == 'X' else "White") 
            print("Black-to-white ratio:", ratio)
            print("Black wins:", black_wins)
            print("White wins:", white_wins)
            print("Ties:", ties)
            print("WR", white_wins / (black_wins + white_wins + ties))

    @staticmethod
    def generate_data(max_games, model_path):
        """
        Generates training data by playing games.
        Args:
            max_games (int): number of games to play
            model_path (str): path to torch model
        """
        all_states = np.zeros((max_games, 5, GAME_SIZE, GAME_SIZE))
        count = 0
        running = True
        start_time = datetime.datetime.now()
        while running:
            game = Othello()
            states = []
            turns = []
            for _ in range(10):
                move = rand.choose_move(game.board, game.valid_move)
                if move is None:
                    break
                game.make_move(*move)
                states.append(copy.deepcopy(game.board))
                turns.append(1 if game.current_player == 'X' else 0)
            winner, states_cntd, turns_cntd, _ = OthelloGame.play(model_path, False, game)
            states.extend(states_cntd)
            turns.extend(turns_cntd)
            winner = {'X': 1.0, 'O': 0.0, 'Tie': 0.5}[winner]
            states = np.array(states)
            for i in range(len(states)):
                state = states[i]
                turn = turns[i]
                if count >= max_games:
                    running = False
                    break
                all_states[count, 0] = (state == 'X').astype(int)
                all_states[count, 1] = (state == 'O').astype(int)
                all_states[count, 2] = (state == ' ').astype(int)
                all_states[count, 3] = turn
                all_states[count, 4] = winner
                count += 1
            if count % 1 == 0:
                print(count, "Estimated time remaining:", (datetime.datetime.now() - start_time) / count * (max_games - count))

        file_id = random.randint(0, 1000000)
        np.save(f'self_games_{file_id}.npy', all_states)

if __name__ == '__main__':
    winner, _, _, ratio = OthelloGame.play("final_model_c.pth", True)
    if winner == 'X':
        print("Black wins!")
    elif winner == 'O':
        print("White wins!")
    else:
        print("Tie!")
    print("Black-to-white ratio:", ratio)
    # OthelloGame.play_repeated_games("final_model_c.pth")
    # OthelloGame.generate_data(100000, "final_model_c.pth")