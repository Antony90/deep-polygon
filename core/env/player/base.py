from collections import deque
from queue import PriorityQueue

from abc import abstractmethod, ABC
import random
from typing import Literal, TypeVar

import numpy as np

from constants import STATE_SHAPE, STATE_WIDTH, VECTORS_YX, Direction
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from env.game import Game


T = TypeVar("T", bound="Player")

class Player(ABC):
    def __init__(self, game: "Game", player_id: int, spawn_pos: list[int], spawn_dir: Direction):
        self.game = game
        self.id = player_id
        self.pos = spawn_pos
        self.last_dir = Direction.PAUSE # the action applied to be in current state
        self.trail_start_dir = Direction.PAUSE
        self.trail = []
        self.dead = False
        self.kills = 0
        self.area = 0
        # A rectangle representing an optimistic area covered by the player
        # This is updated to possibly become larger when the player captures land
        self.area_bounds = (spawn_pos[0], spawn_pos[0], spawn_pos[1], spawn_pos[1])

        self.time_since_land = 0
        self.trail_start_rel = (0, 0) # relative pos to trail start 

    @classmethod
    def spawn(cls: type[T], env: "Game", player_id: int) -> T:
        # Generate valid spawn pos and starting direction
        generate_spawn_pos = lambda: [
            random.randint(5, env.map_size - 6) for _ in range(2)
        ]

        spawn_pos = generate_spawn_pos()
        max_attempts = 5
        for _ in range(max_attempts):
            if env.is_valid_spawn(spawn_pos):
                break
            spawn_pos = generate_spawn_pos()

        spawn_dir = random.randint(0, 3)
        player = cls(env, player_id, spawn_pos, spawn_dir)

        # Fill starting 5x5 area with player head in center
        player.area_bounds = (
            spawn_pos[0] - 2,
            spawn_pos[1] - 2,
            spawn_pos[0] + 2,
            spawn_pos[1] + 2
        )
        env.grid.set_area(player.id, *player.area_bounds)
        player.calculate_area()

        return player


    def calculate_area(self):
        (minX, minY, maxX, maxY) = self.area_bounds
        self.area = np.count_nonzero(self.game.grid[minY:maxY+1, minX:maxX+1] == self.id)
        return self.area

    def finish_trail(self) -> int:
        grid = self.game.grid

        # 1. Update area bounds with trail
        (min_x, min_y, max_x, max_y) = self.area_bounds
        # Head pos is not part of trail list
        # trail_with_head = self.trail + [self.pos]

        for i in range(len(self.trail)):
            point1 = self.trail[i]
            point2 = self.trail[i + 1] if i < len(self.trail) - 1 else self.pos

            if point1[0] > point2[0] or point1[1] > point2[1]:
                point1, point2 = point2, point1
            self.game.grid.set_area(self.id, *point1, *point2)

            if (point1[0] < min_x):
                min_x = point1[0]
            if (point2[0] > max_x):
                max_x = point2[0]
            if (point1[1] < min_y):
                min_y = point1[1]
            if (point2[1] > max_y):
                max_y = point2[1]

        self.trail.clear()
        self.area_bounds = (min_x, min_y, max_x, max_y)

        # 2. Prepare a mask for the area to fill
        # Within bounds and not owned by the player
        area = grid[min_y:max_y+1, min_x:max_x+1]
        fill_mask = (area != self.id)

        # 3. Flood fill from the border, treating owned cells (including the trail) as barriers
        h, w = fill_mask.shape
        visited = np.zeros_like(fill_mask, dtype=bool)
        q = deque()

        # Enqueue all border cells that are not owned by the player
        for x in range(w):
            if fill_mask[0, x]:
                q.append((0, x))
                visited[0, x] = True
            if fill_mask[h-1, x]:
                q.append((h-1, x))
                visited[h-1, x] = True

        for y in range(h):
            if fill_mask[y, 0]:
                q.append((y, 0))  
                visited[y, 0] = True
            if fill_mask[y, w-1]:
                q.append((y, w-1))
                visited[y, w-1] = True

        # BFS to flood fill all cells starting from the border
        # The "loop" the player's trail created will not be filled
        while q:
            y, x = q.popleft()
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = y+dy, x+dx
                if 0 <= ny < h and 0 <= nx < w and fill_mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    q.append((ny, nx))

        # 4. All unvisited cells in fill_mask are therefore capturable
        to_fill = fill_mask & (~visited)
        captured = np.count_nonzero(to_fill)
        area[to_fill] = self.id

        return captured


    def finish_trail_old(self) -> None:
        (minX, minY, maxX, maxY) = self.area_bounds
        parts = len(self.trail)
        for i in range(parts):
            point1 = self.trail[i]
            point2 = self.trail[i + 1] if i < parts - 1 else self.pos

            if point1[0] > point2[0] or point1[1] > point2[1]:
                point1, point2 = point2, point1
            self.game.grid.set_area(self.id, *point1, *point2)

            if (point1[0] < minX):
                minX = point1[0]
            if (point2[0] > maxX):
                maxX = point2[0]
            if (point1[1] < minY):
                minY = point1[1]
            if (point2[1] > maxY):
                maxY = point2[1]
        self.trail.clear()
        self.area_bounds = (minX, minY, maxX, maxY)

        queue = deque()
        fill_grid = np.zeros_like(self.game.grid) # 0 = not ours, 1 = ours, 2 = prevented

        # prevent filling outside of claimed area
        fill_grid[:, minX] = fill_grid[:, maxX] = fill_grid[minY, :] = fill_grid[maxY, :] = 2

        # heads and trail starts of enemies prevent fill
        for enemy in self.game.get_players():
            # TODO: cull far away players
            if enemy.dead or enemy == self:
                continue
            (vy, vx) = enemy.pos
            for y, x in VECTORS_YX:
                neighbour_pos = (vy + y, vx + x)
                if neighbour_pos[0] < minY or neighbour_pos[0] > maxY or neighbour_pos[1] < minX or neighbour_pos[1] > maxX:
                    continue
                fill_grid[neighbour_pos] = 2
            if len(enemy.trail) > 0:
                (vy, vx) = enemy.trail[0]
                for y, x in VECTORS_YX:
                    neighbour_pos = (vy + y, vx + x)
                    if neighbour_pos[0] < minY or neighbour_pos[0] > maxY or neighbour_pos[1] < minX or neighbour_pos[1] > maxX:
                        continue
                    fill_grid[neighbour_pos] = 2

        fill_grid[self.game.grid == self.id] = 1

        # flood fill where to prevent the land filling in
        queue.extend(zip(*np.where(fill_grid == 2)))
        while len(queue) > 0:
            (vy, vx) = queue.pop()
            for y, x in VECTORS_YX:
                neighbour_pos = (vy + y, vx + x)
                if neighbour_pos[0] < minY or neighbour_pos[0] > maxY or neighbour_pos[1] < minX or neighbour_pos[1] > maxX:
                    continue
                if fill_grid[neighbour_pos] == 0:
                    fill_grid[neighbour_pos] = 2
                    queue.append(neighbour_pos)

        # flood fill land
        queue.extend(zip(*np.where(fill_grid == 1)))
        while len(queue) > 0:
            (vy, vx) = queue.pop()
            for y, x in VECTORS_YX:
                neighbour_pos = (vy + y, vx + x)
                if neighbour_pos[0] < minY or neighbour_pos[0] > maxY or neighbour_pos[1] < minX or neighbour_pos[1] > maxX:
                    continue
                if fill_grid[neighbour_pos] == 0:
                    fill_grid[neighbour_pos] = 1
                    queue.append(neighbour_pos)

        if not self.dead:
            self.game.grid[fill_grid == 1] = self.id

    def trail_has_block(self, pos, ignore_head):
        parts = len(self.trail)
        for i in range(parts - 1 if ignore_head else parts):
            point1 = self.trail[i]
            point2 = self.trail[i+1] if i < parts - 1 else self.pos

            axis = 1 if point1[0] == point2[0] else 0
            if pos[1 - axis] != point1[1 - axis]:
                continue

            start = point1[axis]
            end = point2[axis]
            val = pos[axis]

            if start <= val <= end or start >= val >= end:
                return True
        return False

    def get_opposite_trail_dir(self):
        points = len(self.trail)
        if points < 2:
            return None
        point1 = self.trail[-1]
        point2 = self.trail[-2]

        if point1[0] > point2[0]:
            return Direction.LEFT
        elif point1[0] < point2[0]:
            return Direction.RIGHT
        elif point1[1] > point2[1]:
            return Direction.UP
        elif point1[1] < point2[1]:
            return Direction.DOWN
        else:
            return (self.trail_start_dir + 2) % 4 if points == 2 else None

    def die(self):
        self.dead = True
        self.game.grid.wipe_player(self.id)

    def calc_trail_length(self) -> int:
        if len(self.trail) == 0:
            return 0
        length = 0
        segments = self.trail + [self.pos]
        i = 1
        while i < len(segments):
            prev_x, prev_y = segments[i-1]
            next_x, next_y = segments[i]

            if prev_x == next_x:
                start = min(prev_y, next_y)
                end = max(prev_y, next_y)
            else: # prev_y == next_y
                start = min(prev_x, next_x)
                end = max(prev_x, next_x)

            length += end - start
            i += 1
        return length + 1 # add 1 since last segment pos not counted
    
    def __eq__(self, player):
        # errors if other obj not Player
        return self.id == player.id
    
    def __lt__(self, player):
        return self.id < player.id
    
    def already_paused(self):
        return self.last_dir == Direction.PAUSE

    def find_closest_k_enemies(self, game: "Game", k: int, max_dist=STATE_WIDTH):
        """Closest k (2) enemies, in ascending manhattan distance from position.
        
        If >0 enemies found, returns (closest_enemy_dist, enemies)
        Else return (-1, [])"""
        enemy_queue = PriorityQueue(maxsize=game.get_num_players() - 1)
        found_any = False
        
        for enemy in game.get_players():
            if enemy == self:
                continue
            
            player_dist = game.manhattan_distance(enemy.pos, self.pos)
            if player_dist > max_dist:
                continue

            found_any = True
            enemy_queue.put_nowait((player_dist, enemy))

        if not found_any:
            return -1, []
        
        # at least one enemy in distance range, non-empty queue
        closest_enemies = []
        i = 0 # no. enemies removed

        # get first k enemies from queue (k > 0)
        while enemy_queue.qsize() > 0 and i < k:
            dist_temp, closest = enemy_queue.get_nowait()
            if i == 0:
                closest_dist = dist_temp

            closest_enemies.append(closest)
            i += 1
        
        return closest_dist, closest_enemies
    
    
    def has_trail(self):
        return len(self.trail) > 0


    def trail_start(self) -> list[int, int] | Literal[False]:
        return self.has_trail() and self.trail[0]

    def rel_trail_start_yx(self) -> tuple[int, int]:
        """relative distance to trail start pos or (0, 0) if no trail."""
        trail_start = self.trail_start()
        if not trail_start:
            return (0, 0)
        x, y = trail_start
        px, py = self.pos

        max_dist = STATE_SHAPE[0]//2 # trail start at edge of state
        return (
            np.clip((y - py), -max_dist, max_dist) / max_dist,
            np.clip((x - px), -max_dist, max_dist) / max_dist
        )

    # override the following for specialized AI
    @abstractmethod
    def before_state_and_reward(self, game, action: Direction, killed_enemy: bool):
        """Callback to allow updating private attributes which are used in generating the state"""
        pass

    @abstractmethod
    def generate_state(self, action, game):
        pass
        
    @abstractmethod
    def get_reward(
        self,
        num_killed: int,
        area_change: float,
        reversed: bool,
        action: Direction
    ) -> float:
        pass