from random import SystemRandom
from typing import Optional

from constants import DEFAULT_MAP_SIZE, Direction

from env.player.base import Player
from env.player.builder import Builder
from env.grid import Grid, GridValue
from env.player.extra import Killer

systemrandom = SystemRandom()

# TODO: methods to create a group of collaborating players. See Royalist/Protector

class Game:
    def __init__(self, max_players=None, map_size=None) -> None:
        if map_size:
            self.map_size = map_size
        else:
            if max_players is not None:
                self.map_size = self.estimate_map_size(max_players)
            else:
                self.map_size = DEFAULT_MAP_SIZE
                
        self.grid = Grid(self.map_size)
        self.players: dict[Player] = {}
        self.running = True

    @staticmethod
    def estimate_map_size(max_players: int) -> int:
        return max_players * 8

    @staticmethod
    def manhattan_distance(pos1: tuple[int], pos2: tuple[int]):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def step(self, player: Player, action: Direction):
        player_has_trail = len(player.trail) > 0

        took_reverse_dir = False
        # prevent reversing
        if action == Direction.opposite(player.last_dir) or (player_has_trail and player.last_dir == Direction.PAUSE and action == player.get_opposite_trail_dir()):
            action = player.last_dir
            took_reverse_dir = True

        # add "bend" in trail before changing dir
        if action != player.last_dir and player_has_trail:
            player.trail.append([*player.pos])

        match action:
            case Direction.RIGHT:
                player.pos[0] += 1
            case Direction.DOWN:
                player.pos[1] += 1
            case Direction.LEFT:
                player.pos[0] -= 1
            case Direction.UP:
                player.pos[1] -= 1

        area_change = 0

        # check kill conditions
        killed = {}
        killed_enemy = False # any enemy
        if action != Direction.PAUSE:
            if player_has_trail and self.grid.get_block(player.pos) == player.id:
                # entering own land, capture area
                area_before = player.calculate_area()
                player.finish_trail()
                player_has_trail = False
                area_after = player.calculate_area()
                area_change = area_after - area_before

            elif not player_has_trail and not self.grid.get_block(player.pos) == player.id:
                # exiting own land, create trail start
                player.trail.append([*player.pos])
                player.trail_start_dir = action
                player_has_trail = True

            for enemy in self.players.values():
                if enemy.dead:
                    continue

                enemy_has_trail = len(enemy.trail) > 0
                heads_collision = player.pos[0] == enemy.pos[0] \
                    and player.pos[1] == enemy.pos[1] \
                    and (enemy.last_dir == Direction.PAUSE or enemy_has_trail) \
                    and enemy != player

                if heads_collision or enemy.trail_has_block(player.pos, ignore_head=enemy == player):
                    # to introduce immortal bug: and (self.grid.get_block(enemy.pos) != enemy.id or not enemy_has_trail)
                    if enemy.id not in killed:
                        # player kills enemy
                        killed[enemy.id] = player.id
                        player.kills += 1
                        killed_enemy = True
                        enemy.die()
                    if heads_collision and enemy_has_trail:
                        # enemy kills player
                        killed[player.id] = enemy.id
                        enemy.kills += 1

        if self.grid.get_block(player.pos) == GridValue.WALL:
            killed[player.id] = GridValue.WALL


        # # player.trail_start_rel_prev = player.trail_start_rel
        # player.trail_start_rel = player.rel_trail_start_yx()

        # head collision, too far from land
        if player.id in killed :
            player.die()

        player.before_state_and_reward(self, action, killed_enemy)

        num_killed = len([k for k in killed.keys() if k != player.id])
        new_state = player.generate_state(action, self)

        reward = player.get_reward(num_killed, area_change, took_reverse_dir, action)
        done = player.dead

        player.last_dir = action
        return new_state, reward, done
    
    
    def is_valid_spawn(self, spawn_pos: list[int], min_player_dist=15) -> bool:
        # Cannot spawn head on enemy land
        if self.grid.get_block(spawn_pos) != GridValue.EMPTY:
            return False
            
        # Cannot be too close to players
        # Note: this can cause spawning to be impossible
        # In small maps with too many players
        for player in self.players.values():
            dist = self.manhattan_distance(player.pos, spawn_pos)
            if dist <= min_player_dist:
                return False
            
        return True
            

    def spawn_player(self, player_cls: Player, replace_player_id: Optional[int] = None):
        '''Specify `replace_player_id` to replace a specific player instance'''
        player_id = replace_player_id or len(self.players) + 1
            
        player = player_cls.spawn(self, player_id)
                    
        # Store player and generate initial state
        self.players[player_id] = player
        return player, player.generate_state(Direction.PAUSE, self)

    def close(self):
        self.running = False

    def is_running(self):
        return self.running
