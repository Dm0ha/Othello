H_WEIGHTS = [
    [4, -3, 2, 2, 2, 2, -3, 4],
    [-3, -4, -1, -1, -1, -1, -4, -3],
    [2, -1, 1, 0, 0, 1, -1, 2],
    [2, -1, 0, 1, 1, 0, -1, 2],
    [2, -1, 0, 1, 1, 0, -1, 2],
    [2, -1, 1, 0, 0, 1, -1, 2],
    [-3, -4, -1, -1, -1, -1, -4, -3],
    [4, -3, 2, 2, 2, 2, -3, 4]
]
    
class BoardUtils:
    """
    A class for various board utilities.
    """
    @staticmethod
    def valid_move_from_board(board, x, y, turn):
        """
        Check if a move is valid.
        Args:
            board (list[list[str]]): board
            x (int): x cell
            y (int): y cell
            turn (int): current turn (1 or 0)
        Returns:
            bool: whether the move is valid
        """
        game_size = len(board)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        current_player = 'X' if turn == 1 else 'O'
        opponent = 'O' if turn == 1 else 'X'
        if board[x][y] != ' ':
            return False
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < game_size and 0 <= ny < game_size and board[nx][ny] == opponent:
                while 0 <= nx < game_size and 0 <= ny < game_size and board[nx][ny] != ' ':
                    nx += dx
                    ny += dy
                    if 0 <= nx < game_size and 0 <= ny < game_size and board[nx][ny] == current_player:
                        return True
        
        return False
        
    @staticmethod
    def moves_from_board(board, turn):
        """
        Get all valid moves.
        Args:
            board (list[list[str]]): board
            turn (int): current turn (1 or 0)
        Returns:
            list[tuple]: list of moves
        """
        game_size = len(board)
        moves = []
        for x in range(game_size):
            for y in range(game_size):
                if BoardUtils.valid_move_from_board(board, x, y, turn):
                    moves.append((x, y))
        return moves
    
    @staticmethod
    def is_over_from_board(board):
        """
        Check if the game is over.
        Args:
            board (list[list[str]]): board
        Returns:
            bool: whether the game is over
        """
        return len(BoardUtils.moves_from_board(board, 0)) == 0 and len(BoardUtils.moves_from_board(board, 1)) == 0
    
    @staticmethod
    def winner_from_board(board):
        """
        Get the winner of the game.
        Args:
            board (list[list[str]]): board
        Returns:
            int: winner (1 or 0)
        """
        game_size = len(board)
        x_count = 0
        o_count = 0
        for x in range(game_size):
            for y in range(game_size):
                if board[x][y] == 'X':
                    x_count += 1
                elif board[x][y] == 'O':
                    o_count += 1
        if x_count > o_count:
            return 1
        elif o_count > x_count:
            return 0
        else:
            return 0.5
        
    @staticmethod
    def piece_ratio_from_board(board):
        """
        Get the piece ratio of the game.
        Args:
            board (list[list[str]]): board
        Returns:
            float: piece ratio
        """
        game_size = len(board)
        x_count = 0
        o_count = 0
        for x in range(game_size):
            for y in range(game_size):
                if board[x][y] == 'X':
                    x_count += 1
                elif board[x][y] == 'O':
                    o_count += 1
        return x_count / (x_count + o_count)
    
    @staticmethod
    def moves_remaining_from_board(board):
        """
        Get the number of moves remaining.
        Args:
            board (list[list[str]]): board
        Returns:
            int: number of moves remaining
        """
        moves = 0
        for x in range(len(board)):
            for y in range(len(board)):
                if board[x][y] == ' ':
                    moves += 1
        return moves
    
    @staticmethod
    def weights_heuristic(board):
        """
        Get the static weights table heuristic value of the game.
        Args:
            board (list[list[str]]): board
        Returns:
            float: weights heuristic
        """
        game_size = len(board)
        x_count = 0
        o_count = 0
        for x in range(game_size):
            for y in range(game_size):
                if board[x][y] == 'X':
                    x_count += H_WEIGHTS[x][y]
                elif board[x][y] == 'O':
                    o_count += H_WEIGHTS[x][y]
        return x_count - o_count