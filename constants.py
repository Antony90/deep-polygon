from enum import Enum, IntEnum

DEFAULT_MAP_SIZE = 600
GLOBAL_SPEED = 0.006

STATE_WIDTH = 31
STATE_SHAPE = (STATE_WIDTH, STATE_WIDTH, 3)
STATE_SHAPE_CHANNELS_FIRST = (3, STATE_WIDTH, STATE_WIDTH)

STATE_N_VARS = 5

# RIGHT, DOWN, LEFT, UP
VECTORS_XY = [(1, 0), (0, 1), (-1, 0), (0, -1)]
VECTORS_YX = [(0, 1), (1, 0), (0, -1), (-1, 0)]

class Direction(IntEnum):
	RIGHT = 0
	DOWN = 1
	LEFT = 2
	UP = 3
	PAUSE = 4


	@staticmethod
	def opposite(dir: int) -> int:
		if dir == Direction.PAUSE:
			return None
		return (dir + 2) % 4