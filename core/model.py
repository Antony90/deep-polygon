import torch as T
import torch.nn as nn
import torch.nn.functional as F

from constants import STATE_N_VARS, STATE_SHAPE_CHANNELS_FIRST, Direction

class GridNet(nn.Module):    
    def __init__(self, num_channels: int, num_scalars: int, num_actions: int):
        super(GridNet, self).__init__()
        self.num_scalars = num_scalars
        self.num_actions = num_actions
        # Test 1
        # self.conv1 = nn.Conv2d(grid_shape[0], 32, kernel_size=5, stride=2)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        # self.fc1 = nn.Linear(64 * 12 * 12, 128)
        # self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, num_actions)

        # Test 2
        # self.conv1 = nn.Conv2d(grid_shape[0], 32, kernel_size=5, stride=2)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        # self.fc1 = nn.Linear(64 * 10 * 10, 128)
        # self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, num_actions)

        # Test 3

        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        
        fc_scalar_out = 64
        self.fcs1 = nn.Linear(self.num_scalars, 128)
        self.fcs2 = nn.Linear(128, fc_scalar_out)
        
        flattened_grid_size = 64 * 7 * 7
        self.fc1 = nn.Linear(flattened_grid_size + fc_scalar_out, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_actions)

        # Current model
        # self.conv1 = nn.Conv2d(grid_shape[0], 32, kernel_size=3, stride=2)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=2)
        # self.fc1 = nn.Linear(64 * 14 * 14, 128)
        # self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, num_actions)

    def forward(self, state) -> T.Tensor:
        grid_input, scalar_input = state
        x = F.relu(self.conv1(grid_input))
        x = F.relu(self.conv2(x))
        x = T.flatten(x, 1)

        y = F.relu(self.fcs1(scalar_input))
        y = F.relu(self.fcs2(y))
        
        x = T.cat((x, y), dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    
    # https://stackoverflow.com/a/62508086
    from prettytable import PrettyTable
    def count_parameters(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, f"{params:,}"])
            total_params+=params
        print(table)
        print(f"Total Trainable Params: {total_params:,}")
        return total_params
    
    num_channels = STATE_SHAPE_CHANNELS_FIRST[0]
    model = GridNet(num_channels=num_channels, num_actions=len(Direction), num_scalars=STATE_N_VARS)
    count_parameters(model)
    batch_size = 2
    # test fc1 layer size is correct
    grid = T.zeros((batch_size, *STATE_SHAPE_CHANNELS_FIRST))
    scalar = T.zeros((batch_size, model.num_scalars))
    
    model.forward((grid, scalar))
    