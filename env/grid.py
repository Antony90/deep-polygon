from enum import IntEnum
import numpy as np


class GridValue(IntEnum):
    WALL = -1 # can't be 1 since occupied by player tiles
    EMPTY = 0

    @staticmethod
    def unused_value(player_id: int):
        return -player_id - 1

class Grid(np.ndarray):
    def __new__(cls, map_size: int, *args, **kwargs):
        return super().__new__(cls, (map_size, map_size), dtype=np.int8)

    def __init__(self, map_size):
        self.fill(GridValue.EMPTY)
        self[:, 0] = self[:, -1] = self[0, :] = self[-1, :] = GridValue.WALL

    def get_block(self, pos):
        return self[pos[1], pos[0]]

    def set_area(self, owner, start_x, start_y, end_x, end_y):
        '''Fills area, inclusive'''
        self[start_y:end_y+1, start_x:end_x+1] = owner

    def wipe_player(self, owner):
        self[self == owner] = GridValue.EMPTY

    #
    # def eq(self, player: Player):
    #     (min_x, min_y, max_x, max_y) = self.area_bounds
    #     self.game.grid[min_y:max_y, min_x:max_x] == self.id
    #