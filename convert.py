"""
A script to convert the Kaggle pro games dataset into a format usable by the model.
"""

import pandas as pd
import numpy as np
import main as game
from board_utils import BoardUtils

df = pd.read_csv('othello_dataset.csv')
let_to_num = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h':7}

num_moves = 0
for i in range(len(df)):
    num_moves += len(df.loc[i, 'game_moves']) // 2
states = np.zeros((num_moves, 6, 8, 8))
count = 0
for i in range(len(df)):
    if i % 1000 == 0:
        print(i)
    game_moves = df.loc[i, 'game_moves']
    game_moves = [game_moves[i:i+2] for i in range(0, len(game_moves), 2)]
    winner = df.loc[i, 'winner']
    winner = {1: 1.0, -1: 0.0, 0: 0.5}[winner]
    # Get final ratio
    board = game.Othello()
    for move in game_moves:
        x = int(move[1]) - 1
        y = let_to_num[move[0]]
        if not board.make_move(x, y):
            print(x, y)
            print("Invalid move")
            exit()
    b = np.array(board.board)
    ratio = np.sum(b == 'X') / (np.sum(b == 'O') + np.sum(b == 'X'))
    # Get states
    board = game.Othello()
    for move in game_moves:
        x = int(move[1]) - 1
        y = let_to_num[move[0]]
        board.make_move(x, y)
        b = np.array(board.board)
        states[count, 0] = (b == 'X').astype(int)
        states[count, 1] = (b == 'O').astype(int)
        states[count, 2] = (b == ' ').astype(int)
        states[count, 3] = 1 if board.current_player == 'X' else 0
        states[count, 4] = BoardUtils.weights_heuristic(board.board)
        states[count, 5] = winner
        count += 1
np.save('formatted_datasets/states_win_heu.npy', states)
    