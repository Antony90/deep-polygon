import numpy as np

from constants import STATE_WIDTH, Direction
from env.state import State
from env.grid import Grid, GridValue
from env.player.base import Player


class Builder(Player):
    """Emphasise building or killing"""
    max_time_since_kill = 40 # max actions without killing while an enemy is on screen and player is in kill zone
    max_pause = 5            # max consecutive pauses before death
    max_trail_len = 60       # max trail length before death
    max_dist_from_enemy = 5  # max manhattan distance between closest enemy (if any enemy is in fov)], will take penalty when exceeded
    def __init__(self, game, player_id: int, spawn_pos: list[int], spawn_dir: Direction):
        super().__init__(game, player_id, spawn_pos, spawn_dir)
        self.pause_cntr = 0 # consecutive pause actions
        # self.since_kill = 0 # steps since kill
        self.closest_2_enemies: list[Player] = []
        self.closest_enemy_dist = -1 # unset / no enemies in fov
        self.prev_closest_enemy_dist = -1
        self.trail_length = 0 # not the number of segments. see `self.calc_trail_length()`
        self.enemy_in_fov = False
        self.in_land_cntr = 0 # time in land
        self.trail_start_dist = -1


    def draw_enemy(self, enemy: Player, state: np.ndarray, grid: Grid, enemy_num: int):
        State.draw_blocks(state, self.pos[::-1], grid == enemy.id, enemy_num)
        State.draw_trail(state, enemy, self.pos, enemy_num)

        pos_x, pos_y = State.abs_to_rel_xy(enemy.pos, self.pos)
        enemy_pos_rel_yx = (pos_y, pos_x)
        if not State.is_outside(enemy_pos_rel_yx):
            State.draw_head(state, enemy_pos_rel_yx, enemy_num)

    def is_pos_in_fov(self, pos):
        return abs(self.pos[0] - pos[0]) <= STATE_WIDTH//2 and \
               abs(self.pos[1] - pos[1]) <= STATE_WIDTH//2

    def is_enemy_in_fov(self):
        """For each of the closest 2 enemies, check if their head or any part of trail is in the player's FOV."""
        if len(self.closest_2_enemies) == 0:
            return False
    
        for enemy in self.closest_2_enemies:
            # first check heads
            if self.is_pos_in_fov(enemy.pos):
                return True
            
            
            # # TODO: broken, also not ideal for js client side
            # if len(enemy.trail) == 0:
            #     continue
            
            # i = 1
            # while i < len(enemy.trail):
            #     prev_x, prev_y = enemy.trail[i-1]
            #     next_x, next_y = enemy.trail[i]

            #     if prev_x == next_x:
            #         start = min(prev_y, next_y)
            #         end = max(prev_y, next_y)
            #         for var_pos_y in range(start, end + 1):
            #             if self.is_pos_in_fov((prev_x, var_pos_y)):
            #                 return True
            #     else: # prev_y == next_y
            #         start = min(prev_x, next_x)
            #         end = max(prev_x, next_x)
            #         for var_pos_x in range(start, end + 1):
            #             if self.is_pos_in_fov((var_pos_x, prev_y)):
            #                 return True
                        
            #     i += 1

        return False # enemies are in detect range (see self.find_closest_k_enemies) but trail and head aren't in fov
                         
    def before_state_and_reward(self, game, action: Direction, killed_enemy: bool):
        # to be passed to variables state and reward function
        self.prev_closest_enemy_dist = self.closest_enemy_dist
        self.closest_enemy_dist, self.closest_2_enemies = self.find_closest_k_enemies(game, k=2) # closest first

        trail_start = self.trail_start()
        if trail_start:
            self.prev_trail_start_dist = self.trail_start_dist
            self.trail_start_dist = game.manhattan_distance(trail_start, self.pos)
        
        self.trail_length = self.calc_trail_length()
        # self.enemy_in_fov = self.is_enemy_in_fov()
        
        if len(self.trail) == 0:
            self.in_land_cntr += 1
        else:
            self.in_land_cntr = 0

        if action == Direction.PAUSE:
            self.pause_cntr += 1
        else:
            self.pause_cntr = 0

        # if killed_enemy or not self.enemy_in_fov or self.closest_enemy_dist > self.max_dist_from_enemy:
        #     self.since_kill = 0
        # else: # exist enemies or no kill or inside kill range but not killing 
        #     self.since_kill += 1

        # self.since_kill > self.max_time_since_kill
        if (self.pause_cntr == self.max_pause or \
            self.trail_length == self.max_trail_len
        ):
            self.die()
            

    def generate_grid_state(self, game, closest_2_enemies: list[Player]) -> np.ndarray:
        state = State.template()
        state_center_yx = State.center_yx() # player head is in center of state array
        center_pos_yx = self.pos[::-1]

        # self trail, blocks, head
        State.draw_blocks(state, center_pos_yx, game.grid == self.id)
        State.draw_trail(state, self, self.pos)
        State.draw_head(state, state_center_yx)
        
        # enemy trail, blocks, head
        
        # draw lowest priority enemy first so it is overwritten by higher priority in state, if overlapping Block or Overlay
        for i in range(len(closest_2_enemies)-1, -1, -1):
            self.draw_enemy(closest_2_enemies[i], state, game.grid, enemy_num=i)

        State.draw_walls(state, center_pos_yx, game.grid == GridValue.WALL)

        # transpose to have channels first for pytorch CNN
        return state.transpose((2, 0, 1)).astype(np.float32)


    def generate_scalar_state(self, action, game, enemies: list[Player]):
        # unset
        # enemy1_dir = -1
        # enemy2_dir = -1
        # if len(enemies) >= 1 and self.is_pos_in_fov(enemies[0].pos): # assumes enemies ordered by distance
        #     enemy1_dir = enemies[0].last_dir
        #     if len(enemies) == 2 and self.is_pos_in_fov(enemies[1].pos):
        #         enemy2_dir = enemies[1].last_dir
        
        return np.array([
            self.pause_cntr, 
            int(len(self.trail) == 0), # in land 
            self.in_land_cntr,
            action,
            self.trail_length,
            # int(self.enemy_in_fov), # clearer when the kill timer is decreasing
            # self.closest_enemy_dist, # clearer when distance to enemy is too big
            # self.since_kill, 
            # enemy1_dir, 
            # enemy2_dir
        ])


    def generate_state(self, action, game):
        grid = self.generate_grid_state(game, self.closest_2_enemies)
        scalar = self.generate_scalar_state(action, game, self.closest_2_enemies)
        return grid, scalar

    @staticmethod
    def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def get_reward(self, num_killed: int, area_change: float, reversed_direction: bool, action: Direction):
        if self.dead:
            return  -1.0
        # PROBLEMATIC: Violates the Markov property
        # elif reversed_direction: # Trying to reverse while having a trail does nothing, punish
        #     return -0.5
        elif num_killed > 0: # Can kill multiple enemies in 1 step
            return 0.2
        elif area_change != 0:
            # Normalise area of land captured
            # Scales linearly from 0 to 1 for 0 to 80 tiles
            # 1 for > 80 tiles
            return min(area_change/100, 2)
        elif self.has_trail():
            return -0.0050

        # no kill, no land capture, must be in land
        # is punished more than when it has a trail
        # promotes exploring outside land more than staying inside
        else:
            return -0.0100