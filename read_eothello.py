import pyautogui as pag

pag.PAUSE = 2.5

class eOthelloScraper:
    """
    Class for getting board states and administering moves on the eOthello website.
    """
    def __init__(self):
        self.cell_size = None
        self.top_left = None

    def find_board(self):
        """
        Find the board on the screen.
        Returns:
            list[list[tuple]]: list of pixels
        """
        im = pag.screenshot()
        px = im.load()
        size = im.size
        width_middle = im.size[0] / 2
        height_middle = im.size[1] / 2
        miny = None
        maxy = None
        minx = None
        maxx = None
        hit_gray = False

        for i in range(size[1]):
            if not hit_gray and px[width_middle, i] != (34, 34, 34, 255):
                continue
            elif not hit_gray and px[width_middle, i] == (34, 34, 34, 255):
                hit_gray = True
            elif hit_gray and not miny == None and px[width_middle, i] == (34, 34, 34, 255):
                break
            if px[width_middle, i] == (0, 0, 0, 255):
                if miny == None:
                    miny = i
                maxy = i
        hit_gray = False

        for i in range(size[1]):
            if not hit_gray and px[i, height_middle] != (34, 34, 34, 255):
                continue
            elif not hit_gray and px[i, height_middle] == (34, 34, 34, 255):
                hit_gray = True
            if px[i, height_middle] == (0, 0, 0, 255):
                minx = i
                break
        if minx == None or miny == None or maxy == None:
            print("Could not find full board")
            exit(0)
        maxx = minx + (maxy - miny)

        # subtract 15 to not count the borders around board and between cells
        board_width = maxx - minx + 1
        self.cell_size = board_width / 8
        self.top_left = (minx, miny)
        return px

    def populate_board(self, px):
        """
        Populate the board with the current state.
        Args:
            px (list[list[tuple]]): list of pixels
        Returns:
            list[list[str]]: board
            tuple: last played move
            str: last played player (X or O)
        """
        board = [[' '] * 8, [' '] * 8, [' '] * 8, [' '] * 8, [' '] * 8, [' '] * 8, [' '] * 8, [' '] * 8]
        last_played_pos = None
        for i in range(8):
            for j in range(8):
                cell_pos = (self.top_left[0] + j * self.cell_size + self.cell_size / 2, self.top_left[1] + i * self.cell_size + self.cell_size / 2)

                if px[cell_pos[0], cell_pos[1]] == (246, 253, 250, 255):
                    board[i][j] = 'O'
                elif px[cell_pos[0], cell_pos[1]] == (20, 26, 24, 255):
                    board[i][j] = 'X'
                elif px[cell_pos[0], cell_pos[1]] == (234, 51, 35, 255):
                    last_played_pos = (i, j)
                    if px[cell_pos[0] + self.cell_size / 4, cell_pos[1] + self.cell_size / 4] == (246, 253, 250, 255):
                        board[i][j] = 'O'
                    elif px[cell_pos[0] + self.cell_size / 4, cell_pos[1] + self.cell_size / 4] == (20, 26, 24, 255):
                        board[i][j] = 'X'
                else:
                    board[i][j] = ' '
        return board, last_played_pos, board[last_played_pos[0]][last_played_pos[1]]

    def calculate_board_state(self):
        """
        Calculate the current board state from the eOthello website.
        Returns:
            list[list[str]]: board
            tuple: last played move
            str: last played player (X or O)
        """
        px = self.find_board()
        return self.populate_board(px)

    def click_cell(self, row, column):
        """
        Click a cell on the eOthello website.
        Args:
            row (int): row
            column (int): column
        """
        pag.click(x=(self.top_left[0] + column * self.cell_size + self.cell_size / 2) / 2, y=(self.top_left[1] + row * self.cell_size + self.cell_size / 2) / 2, clicks=2, _pause=False)
        pag.moveTo(x=(self.top_left[0] - self.cell_size) / 2, y=(self.top_left[1] - self.cell_size) / 2, _pause=False)