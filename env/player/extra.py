"""Extra player classes with non traditional play styles"""

from constants import Direction
from env.player.base import Player


class Killer(Player):
    def get_reward(self, state, new_state):
        pass


class Assassin(Player):
    '''
    Group of Players that can access a target's player pos.
    Only has access to their pos. 

    + reward for killing target 
    - reward for dying
    - reward for killing your target's other assassins
    '''

    class AssasinTarget:
        def __init__(self, player: Player) -> None:
            self.player = player
            self.assassins = []

        def get_pos(self):
            return self.player.pos

        # FIXME: IDE doesn't like this type hint
        def add_assassin(self, assassin):
            self.assassins.append(assassin)

    def __init__(self, game, player_id: int, spawn_pos: tuple[int], spawn_dir: Direction, target: AssasinTarget):
        super().__init__(game, player_id, spawn_pos, spawn_dir)
        self.target = target


class Royalist(Player):
    '''
    Protector ensures Royalist does not die
    Royalist is a normal player with references to its Protectors

    + reward for kill
    - reward for killing any of its Protectors
    = no reward for building
    '''

    def __init__(self, game, player_id: int, spawn_pos: tuple[int], spawn_dir: Direction):
        super().__init__(game, player_id, spawn_pos, spawn_dir)
        self.protectors: Protector = []

    def add_protector(self):
        # FIXME: chicken and egg situation here, idk which is instantiated first
        pass


class Protector(Player):
    '''
    Protector can access other protectors through Royalist

    - reward if Royalist dies
    - reward for killing mutual Protectors
    + reward for killing enemies
    - reward for moving away from Royalist
    '''

    def __init__(self, game, player_id: int, spawn_pos: tuple[int], spawn_dir: Direction, royalist: Royalist):
        super().__init__(game, player_id, spawn_pos, spawn_dir)
        self.royalist = royalist

    # distinguishes other Protector
    def generate_state(self, game):
        pass

    def get_reward(self, state, new_state):
        pass
