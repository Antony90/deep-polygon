from enum import IntEnum
from typing import Optional
import numpy as np
from PIL import Image

from constants import STATE_SHAPE

from env.player.base import Player


class Block(IntEnum):
    EMPTY = 0  # TODO: update splix.io train client to have EMPTY = 0, WALL = 1
    WALL = 1
    SELF_BLOCK = 2
    _ENEMY_BLOCK_BASE = 3

    @classmethod
    def enemy_block_for(cls, n: int):
        """Enemy block number from an integer starting at 0, a bijective mapping.
        Can represent some priority/ordering on enemies."""
        return cls._ENEMY_BLOCK_BASE + n


class Overlay:
    """Channel in state for player heads, trails.
    
    The `n`th player starts at `(n * 2)` and has 3 slots: `TRAIL`, `HEAD`.
    
    ```py
    # Example for SELF channel; SELF always takes 0th slot
    state[pos][SELF] = head_slot(0)

    
    # Example for ENEMY channel; can hold an infinite number of enemies
    # This is the kth enemy
    state[pos][ENEMY] = head_slot(k)
    ```
    """
    EMPTY = 0
    # 2 slots per player
    BASE_TRAIL_SLOT = 1
    BASE_HEAD_SLOT  = 2

    @classmethod
    def trail_slot(cls, player_num: int):
        return cls.BASE_TRAIL_SLOT * 2 + player_num
    @classmethod
    def head_slot(cls, player_num: int):
        return cls.BASE_HEAD_SLOT * 2 + player_num


class State:
    # channel indices
    BLOCKS = 0 # wall, player blocks, empty

    # overlay channels contain player head, trail
    SELF   = 1 
    ENEMY  = 2 # can hold infinite enemies
    
    @staticmethod
    def template():
        state = np.zeros(STATE_SHAPE)
        return state

    @staticmethod
    def abs_to_rel_yx(pos: tuple[int], centre_pos: tuple[int]) -> tuple[int]:
        '''Assumes `pos` and `centre_pos` has y then x coordinate'''
        return (pos[0] - centre_pos[0] + (STATE_SHAPE[0]//2), pos[1] - centre_pos[1] + (STATE_SHAPE[1]//2))
    

    @staticmethod
    def abs_to_rel_xy(pos: tuple[int], centre_pos: tuple[int]) -> tuple[int]:
        '''Assumes `pos` and `centre_pos` has x then y coordinate'''
        return (pos[0] - centre_pos[0] + (STATE_SHAPE[1]//2), pos[1] - centre_pos[1] + (STATE_SHAPE[0]//2))
    


    @staticmethod
    def is_outside(pos: tuple[int]) -> bool:
        '''`pos` is (y, x)'''
        return any(pos[i] >= STATE_SHAPE[i] or pos[i] < 0 for i in range(2))
    

    @staticmethod
    def center_yx():
        return (STATE_SHAPE[0]//2, STATE_SHAPE[1]//2)
    
    @staticmethod
    def center_xy():
        return State.center_xy()[::-1]

    @classmethod
    def draw_trail(cls, state: np.ndarray, player: Player, center_xy: tuple[int], enemy_num: Optional[int] = None):
        """Draw a player's trail on to state, relative to the centre pos.

        Args:
            center_xy (tuple[int]): the center pos of the state
            channel (int): overlay channel for trail, either `SELF` or `ENEMY`
            player_num (int, optional): For drawing enemy trails, specify based on an ordering or arbitrarily. Defaults to 0.
        """
        if len(player.trail) == 0:
            return
        center_xy
        i = 0
        while i < len(player.trail):
            x1, y1 = cls.abs_to_rel_xy(tuple(player.trail[i]), center_xy)
            if i == len(player.trail)-1:
                # for last segment pos, link to head pos always at centre
                x2, y2 = cls.abs_to_rel_xy(player.pos, center_xy)
            else:
                x2, y2 = cls.abs_to_rel_xy(tuple(player.trail[i+1]), center_xy)
            i += 1

            xmin = min(x1, x2)
            xmax = max(x1, x2)
            xmin = max(xmin, 0)
            xmax = min(xmax, STATE_SHAPE[1]-1) # 1 extra unit since fill is inclusive

            ymin = min(y1, y2)
            ymax = max(y1, y2)
            ymin = max(ymin, 0)
            ymax = min(ymax, STATE_SHAPE[0]-1)

            if min(xmin, xmax, ymin, ymax) < 0:
                continue

            if enemy_num is None:
                channel = cls.SELF
                player_num = 0 # self always has a single player
            else:
                channel = cls.ENEMY
                player_num = enemy_num


            if xmin == xmax: # same x
                state[ymin:ymax+1, xmin, channel] = Overlay.trail_slot(player_num)
            elif ymin == ymax: # same y
                state[ymin, xmin:xmax+1, channel] = Overlay.trail_slot(player_num)
            
    @classmethod
    def _get_valid_draw_area(cls, center_pos_yx: tuple[int], draw_mask: np.ndarray):
        # TODO: track max/min coords of area to filter faster
        draw_y, draw_x = np.where(draw_mask)

        # transform from absolute to relative position
        ys, xs = cls.abs_to_rel_yx((draw_y, draw_x), center_pos_yx)

        # position indices which are valid
        valid_y = np.where((ys < STATE_SHAPE[0]) & (ys >= 0))[0]
        valid_x = np.where((xs < STATE_SHAPE[1]) & (xs >= 0))[0]
        # take intersection i.e. if x and y are valid the co-ordinate [y, x] is valid
        valid = list(set(valid_y).intersection(valid_x))

        # filter for valid positions
        ys = ys[valid]
        xs = xs[valid]

        return ys, xs

    @classmethod
    def draw_blocks(cls, state: np.ndarray, center_pos_yx: tuple[int], block_draw_mask: np.ndarray, enemy_num: Optional[int] = None):
        '''Draw `value` onto `state` using `centre_pos` as the relative centre to index the `draw_mask`'''
        ys, xs = cls._get_valid_draw_area(center_pos_yx, block_draw_mask)

        if enemy_num is None:
            block = Block.SELF_BLOCK
        else:
            block = Block.enemy_block_for(enemy_num) 

        state[ys, xs, State.BLOCKS] = block

    @classmethod
    def draw_walls(cls, state: np.ndarray, center_pos_yx: tuple[int], wall_draw_mask: np.ndarray):
        ys, xs = cls._get_valid_draw_area(center_pos_yx, wall_draw_mask)
        state[ys, xs, State.BLOCKS] = Block.WALL


    @classmethod
    def draw_head(cls, state: np.ndarray, relative_pos_yx: tuple[int], enemy_num: Optional[int] = None):
        if enemy_num is None:
            channel = cls.SELF
            player_num = 0
        else:
            channel = cls.ENEMY
            player_num = enemy_num
            
        state[relative_pos_yx][channel] = Overlay.head_slot(player_num)

    @classmethod
    def to_img(cls, state: np.ndarray, size=(512, 512)):
        state = state.transpose((1, 2, 0))
        coloured_array = np.zeros_like(state, dtype=np.uint8)

        # TODO: swap wall and empty back for splix.io training
        coloured_array[state[:, :, cls.BLOCKS] == Block.WALL              ] = [255, 255, 255]
        coloured_array[state[:, :, cls.BLOCKS] == Block.EMPTY             ] = [ 16,  20,  27]
        coloured_array[state[:, :, cls.BLOCKS] == Block.SELF_BLOCK        ] = [ 63, 185,  80]
        coloured_array[state[:, :, cls.BLOCKS] == Block.enemy_block_for(0)] = [255,   0,   0]
        coloured_array[state[:, :, cls.BLOCKS] == Block.enemy_block_for(1)] = [255, 205,   0]
        
        # Self overlay colour
        coloured_array[state[:, :, cls.SELF] == Overlay.head_slot(0)      ] = [ 35,  69,  40]
        coloured_array[state[:, :, cls.SELF] == Overlay.trail_slot(0)     ] = [ 51, 123,  61]
        
        # Enemy 1 and 2 overlay colour
        coloured_array[state[:, :, cls.ENEMY] == Overlay.trail_slot(0)    ] = [151,   0,   0]
        coloured_array[state[:, :, cls.ENEMY] == Overlay.head_slot(0)     ] = [ 80,   0,   3]
        coloured_array[state[:, :, cls.ENEMY] == Overlay.trail_slot(1)    ] = [162, 130,   0]
        coloured_array[state[:, :, cls.ENEMY] == Overlay.head_slot(1)     ] = [ 84,  68,   0]

        return Image.fromarray(coloured_array, mode="RGB") \
            .resize(size=size, resample=Image.Resampling.NEAREST)
