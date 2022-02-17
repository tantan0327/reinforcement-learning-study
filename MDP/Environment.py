from enum import Enum


class State():

    def __init__(self, row = -1, column = -1):
        self.row = row
        self.column = column

    def __repr__(self):
        return "<State: [{}, {}]>".format(self.row, self.column)

    def clone(self):
        return State(self.row, self.column)

    def __hash__(self):
        return hash((self.row, self.column))

    def __eq__(self, other):
        return self.row == other.row and self.column == other.column


class Action(Enum):
    UP = 1
    DOWN = -1
    LEFT = 2
    RIGHT = -2

class Environment():

    def __init__(self, grid, move_prob = 0.8):
        # grid is 2-D array
        # 0: ordinary cell
        # -1: damage cell (game end)
        # 1: reward cell (game end)
        # 9: block cell
        self.grid = grid
        self.agent_state = State()

        # reward is minus, so agent has to reach the goal fast
        self.default_reward = -0.04

        # agent can move to a selected direction in move_prob.
        # move different direction in (1 - move_prob)
        self.move_prob = move_prob
        self.reset()

    @property
    def row_length(self):
        return len(self.grid)

    @property
    def column_length(self):
        return len(self.grid[0])

    @property
    def actions(self):
        return [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

    @property
    def state(self):
        states = []
        for row in range(self.row_length):
            for column in range(self.column_length):
                # Block cells are not included
                    if self.grid[row][column] != 9:
                        states.append(State(row, column))

        return states






    
