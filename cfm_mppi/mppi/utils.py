import torch

def singleintegrator_dynamics(
    state: torch.Tensor, action: torch.Tensor, delta_t: float = 0.1
) -> torch.Tensor:
    """
    Update robot state based on differential drive dynamics.
    Args:
        state (torch.Tensor): state batch tensor, shape (batch_size, 2) [x, y]
        action (torch.Tensor): control batch tensor, shape (batch_size, 2) [vx, vy]
        delta_t (float): time step interval [s]
    Returns:
        torch.Tensor: shape (batch_size, 2) [x, y]
    """

    # Perform calculations as before
    state = state + action * delta_t
    return state

def doubleintegrator_dynamics(
    state: torch.Tensor, action: torch.Tensor, delta_t: float = 0.1
) -> torch.Tensor:
    """
    Update robot state based on double integrator dynamics.
    Args:
        state (torch.Tensor): state batch tensor, shape (batch_size, 4) [x, y, vx, vy]
        action (torch.Tensor): control batch tensor, shape (batch_size, 2) [ax, ay]
        delta_t (float): time step interval [s]
    Returns:
        torch.Tensor: shape (batch_size, 4) [x, y, vx, vy]
    """
    new_state = torch.zeros_like(state)
    new_state[:, 0] = state[:, 0] + state[:, 2] * delta_t
    new_state[:, 1] = state[:, 1] + state[:, 3] * delta_t
    new_state[:, 2] = state[:, 2] + action[:, 0] * delta_t
    new_state[:, 3] = state[:, 3] + action[:, 1] * delta_t
    return new_state

def unicycle_dynamics(
    state: torch.Tensor, action: torch.Tensor, delta_t: float = 0.1
) -> torch.Tensor:
    """
    Update robot state based on unicycle dynamics (optimized for computational efficiency).
    Args:
        state (torch.Tensor): state batch tensor, shape (batch_size, 3) [x, y, theta]
        action (torch.Tensor): control batch tensor, shape (batch_size, 2) [v, omega]
        delta_t (float): time step interval [s]
    Returns:
        torch.Tensor: shape (batch_size, 3) [x, y, theta]
    """
    # Vectorized computation of state updates
    cos_theta = torch.cos(state[:, 2])
    sin_theta = torch.sin(state[:, 2])
    
    # Compute state derivatives (using broadcasting)
    derivatives = torch.zeros_like(state)
    derivatives[:, 0] = action[:, 0] * cos_theta
    derivatives[:, 1] = action[:, 0] * sin_theta
    derivatives[:, 2] = action[:, 1]
    
    # Update state with a single operation
    new_state = state + derivatives * delta_t
    
    # Efficient theta normalization (only normalize theta, not the whole tensor)
    new_state[:, 2] = torch.atan2(torch.sin(new_state[:, 2]), torch.cos(new_state[:, 2]))
    
    return new_state

def stage_cost(state, action, goal, obstacle_state, radius, time, prev_action=None, alpha=20.0) -> torch.Tensor:
    """
    Calculate stage cost.
    Args:
        state (torch.Tensor): state batch tensor, shape (batch_size, 2 or greater) [x, y, ...]
        action (torch.Tensor): control batch tensor, shape (batch_size, 2) [vx, vy]
        goal (torch.Tensor): goal tensor, shape (2,) [x, y]
        obstacle_state (torch.Tensor): obstacle state tensor, shape (num_obstacles, 2) [x, y]
        radius (float): collision radius
        time (int): current time step
    Returns:
        torch.Tensor: shape (batch_size,)
    """
    pos = state[:, :2]  # Ensure we only consider the first two dimensions [x, y]
    goal_cost = torch.norm(pos - goal, dim=1) # (batch_size,)

    diff = pos.unsqueeze(1) - obstacle_state[:,:2].unsqueeze(0)
    distances = torch.norm(diff, dim=2) # (batch_size, num_obstacles)

    collision_cost = torch.exp(-alpha * (distances - radius))
    collision_cost = torch.clamp(collision_cost, max=1.0)
    collision_cost = collision_cost.sum(dim=1)

    if prev_action is not None:
        smooth_cost = torch.norm(action - prev_action, dim=1) # (batch_size,)
        cost = 0.1*goal_cost + 100*(1+0.99**time)*collision_cost + 0.1*smooth_cost
    else:
        cost = 0.1*goal_cost + 100*(1+0.99**time)*collision_cost

    return cost

def terminal_cost(state, goal) -> torch.Tensor:
    """
    Calculate terminal cost.
    Args:
        state (torch.Tensor): state batch tensor, shape (batch_size, 2 or greater) [x, y, ...]
        goal (torch.Tensor): goal tensor, shape (2,) [x, y]
    Returns:
        torch.Tensor: shape (batch_size,)
    """
    state = state[:, :2]  # Ensure we only consider the first two dimensions [x, y]
    goal_cost = torch.norm(state - goal, dim=1) # (batch_size,)
    cost = 0.1*goal_cost
    return cost


