from pathlib import Path
import json
from cfm_mppi.models.transformer import TransformerModel
import torch
import numpy as np
import pickle
import time


from cfm_mppi.mppi.flowmppi import FlowMPPI
from cfm_mppi.mppi.utils import stage_cost, terminal_cost, unicycle_dynamics
from cfm_mppi.utils import AgentHistory, evaluate, HumanAgent
from cfm_mppi.evaluation.eval_utils import synthesize_control, CFMConfig
import sys

if len(sys.argv) > 1:
    dataset = sys.argv[1]

torch.manual_seed(0)
np.random.seed(0)

## hyperparameters
SAFE_MARGIN = 0.5
HORIZON = 80
# CFM
SAFE_COEF = [0.1, 0.3, 0.5, 0.7, 0.9]
GOAL_COEF = 0.1

ODE_TIMES = [0.5, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
ODE_TIMES2 = [0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
NOISE_LEVEL = torch.tensor([0.8]).cuda()

# MPPI
MPPI_SIGMA = torch.tensor([0.3, 0.6])
MPPI_LAMBDA = 0.1
U_MIN = torch.tensor([-2.0, -2.0])  # Minimum control inputs
U_MAX = torch.tensor([2.0, 2.0])  # Maximum control inputs
D=0.1

device = 'cuda'
SCALE = 10
dt = 0.1
n_sample = 200
horizon = HORIZON
noise_level = torch.tensor([0.8], device=device)

checkpoint_path = Path("./output_dir/cfm_transformer/checkpoint.pth")
args_filepath = checkpoint_path.parent / 'args.json'
with open(args_filepath, 'r') as f:
    args_dict = json.load(f)

model = TransformerModel()
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["model"])
model.train(False)

model.to(device=device)
batch_size = args_dict['batch_size']

# dataset
if dataset == "ucy" or dataset == "sdd":
    batch_ego = torch.load(f'./dataset/eval80_ego_{dataset}.pt')
    with open(f'./dataset/eval80_obs_{dataset}.pkl', 'rb') as f:
        batch_obs = pickle.load(f)
elif dataset == "sfm":
    batch_ego = torch.zeros(300)



all_average_times = []
all_collisions = []
all_distances = []

state_trajectories = torch.zeros([batch_ego.shape[0], 3, horizon+1], dtype=torch.float32)
control_trajectories = torch.zeros([batch_ego.shape[0], 2, horizon], dtype=torch.float32)

for idx in range(batch_ego.shape[0]):
    if dataset == 'ucy' or dataset == 'sdd':
        state_obs = batch_obs[idx]
        nan_mask = torch.isnan(state_obs).any(dim=(0,2,3))
        state_obs = state_obs[:, ~nan_mask]
        pos_obs = state_obs[:,:,:2,:].to(device)
        vel_obs = state_obs[:,:,2:4,:].to(device)

        start = torch.zeros(1,2).to(device)
        goal = batch_ego[idx,:2,-1].to(device)
    elif dataset == 'sfm':
        start = torch.zeros(1,2).to(device)
        goal = torch.tensor(
            [[6.0, 6.0]], dtype=torch.float32, device=device  # Example goal position
        )
        rng = np.random.RandomState(idx)
        n_hum=20
        humans = [None]*n_hum
        for i in range(n_hum):
            humans[i] = HumanAgent(goal.squeeze().cpu().numpy(), random_generator=rng)

        pos_obs = torch.zeros([1, n_hum, 2, horizon], dtype=torch.float32, device=device)
        vel_obs = torch.zeros([1, n_hum, 2, horizon], dtype=torch.float32, device=device)
        for i in range(n_hum):
            pos_obs[0,i,:,0] = torch.tensor(humans[i].state, device=device)
            vel_obs[0,i,:,0] = torch.tensor(humans[i].control, device=device)



    flowmppi_solver = FlowMPPI(
            num_samples=n_sample,
            dim_state=3,
            dim_control=2,
            dynamics=unicycle_dynamics,
            stage_cost=stage_cost,
            terminal_cost=terminal_cost,
            u_min=U_MIN,
            u_max=U_MAX,
            sigmas=torch.tensor(MPPI_SIGMA),
            lambda_=MPPI_LAMBDA,
            goal=goal.squeeze(),
            horizon=horizon,
            dt=dt,
            dynamics_type="unicycle"
        )


    state = torch.zeros(1,3).to(device)
    x_0 = torch.randn([n_sample, 2, 80], dtype=torch.float32, device=device)
    state_hist = torch.zeros([3, horizon+1], dtype=torch.float32)
    state_hist[:,0] = state.cpu().detach()
    control_hist = torch.zeros([2, horizon], dtype=torch.float32)

    histories = {
        'ego_state': AgentHistory(max_length=10),
        'ego_control_sin': AgentHistory(max_length=10),
        'obs_state': AgentHistory(max_length=10),
        'obs_control': AgentHistory(max_length=10)
    }

    total_time = 0
    noise = None
    for t in range(horizon):
        if dataset == 'sfm':
            if t!=0:
                for i in range(n_hum):
                    others_states = torch.cat([pos_obs[0,np.r_[0:i, i+1:n_hum],:,t-1].cpu(), state[:,:2].cpu()], dim=0)
                    control_history_sin = histories['ego_control_sin'].get()
                    others_controls = torch.cat([vel_obs[0,np.r_[0:i, i+1:n_hum],:,t-1].cpu(), control_history_sin[:,:,-1].cpu()], dim=0)
                    humans[i].social_force_step(others_states.numpy(), others_controls.numpy())
                    pos_obs[0,i,:,t] = torch.tensor(humans[i].state, device=device)
                    vel_obs[0,i,:,t] = torch.tensor(humans[i].control, device=device)
        time_start = time.time()
        pos_obs_1 = pos_obs[:,:,:,t]
        vel_obs_1 = vel_obs[:,:,:,t]

        if t == 0:
            x_t = x_0
            t_curr = torch.tensor([0], device=device)
            ode_times = ODE_TIMES
            current_planning_horizon = horizon
        else:
            t_curr = NOISE_LEVEL
            ode_times = ODE_TIMES2
            current_planning_horizon = x_t.shape[-1]

        config = CFMConfig(
            ode_times=ode_times,
            dt=dt,
            agent_radius=SAFE_MARGIN,
            space_scale=SCALE,
            safe_margin_coefs=SAFE_COEF,
            goal_margin_coef=GOAL_COEF,
            device=device
        )

        controls_dyn, controls_sin = synthesize_control(
            model, flowmppi_solver, config, state, goal, x_t, t_curr, 
            pos_obs_1, vel_obs_1, current_planning_horizon, histories=histories, d=D
        )

        control_dyn = controls_dyn[:,:,0]

        state = unicycle_dynamics(state, control_dyn, dt)
        state_hist[:,t+1] = state.cpu().detach()
        control_hist[:,t] = control_dyn.cpu().detach()

        if t==horizon-1:
            break

        noise = torch.randn(n_sample, controls_sin.shape[1], controls_sin.shape[2], device=device)


        control_history_len = len(histories['ego_control_sin'])
        x_t = noise_level*controls_sin/10 + (1-noise_level)*noise
        x_t = x_t[:,:,(control_history_len+1):]
        
        histories['ego_control_sin'].update(controls_sin[:,:,control_history_len])
        histories['ego_state'].update(state)
        histories['obs_state'].update(pos_obs_1)
        histories['obs_control'].update(vel_obs_1)
        
        control_history_sin = histories['ego_control_sin'].get()
        x_t = torch.cat([control_history_sin.expand(n_sample, -1, -1)/10, x_t], dim=-1)

        time_end = time.time()
        total_time += time_end - time_start

    average_time = total_time / horizon
    collision, distance = evaluate(state_hist[:,1:], control_hist, pos_obs.squeeze(0).detach().cpu(), goal.squeeze(0).detach().cpu(), SAFE_MARGIN)

    all_average_times.append(average_time)
    all_collisions.append(collision)
    all_distances.append(distance)

    state_trajectories[idx] = state_hist
    control_trajectories[idx] = control_hist

    print(idx, flush=True)


all_average_times = torch.tensor(all_average_times)
all_collisions = torch.tensor(all_collisions).float()
all_distances = torch.tensor(all_distances)

mean_time = torch.mean(all_average_times)
var_time = torch.var(all_average_times)

collision_rate = torch.mean(all_collisions) * 100


mean_distance = torch.mean(all_distances)
var_distance = torch.var(all_distances)


directory_path = Path(f'./results/{dataset}_uni')
directory_path.mkdir(parents=True, exist_ok=True)
filename = directory_path / 'cfm_mppi.txt'

with open(filename, 'w') as f:
    f.write("===== SIMULATION RESULTS =====\n\n")
    f.write(f"Date and Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    f.write("===== HYPERPARAMETERS =====\n")
    f.write(f"SAFE_MARGIN: {SAFE_MARGIN}\n")
    f.write(f"SAFE_COEF: {SAFE_COEF}\n")
    f.write(f"GOAL_COEF: {GOAL_COEF}\n")
    f.write(f"MPPI_SIGMA: {MPPI_SIGMA}\n")
    f.write(f"MPPI_LAMBDA: {MPPI_LAMBDA}\n")
    f.write(f"ODE_TIMES: {ODE_TIMES}\n")
    f.write(f"ODE_TIMES2: {ODE_TIMES2}\n")

    f.write("===== SUMMARY STATISTICS =====\n")
    f.write(f"Average Time:\n  Mean: {mean_time:.4f}\n  Variance: {var_time:.6f}\n\n")
    f.write(f"Collision Rate:\n : {collision_rate:.4f}\n\n")
    f.write(f"Distance:\n  Mean: {mean_distance:.4f}\n  Variance: {var_distance:.6f}\n\n")
    f.write("===== DETAILED RESULTS =====\n")
    for i in range(all_collisions.shape[0]):
        f.write(f"{i+1}\t{all_collisions[i]:.4f}\n")

    
