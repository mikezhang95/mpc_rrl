
# jax
import jax
from jax import jit, jacfwd, jacrev, hessian, lax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
jnp.set_printoptions(precision=3)
# from jax.config import config
# config.update("jax_enable_x64", True)
# from jax.experimental.host_callback import call

import os, sys
import numpy as np
import pickle
import time
import copy

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CUR_DIR)
from constants import *

# load jax parameters
MODEL_NAME = "bicycle_model_100ms_20000_v4_jax"
model_path=f"{CUR_DIR}/net_{MODEL_NAME}.model"
NN_W1, NN_W2, NN_W3, NN_LR_MEAN = pickle.load(open(model_path, mode="rb"))
NN_B3 = np.zeros((2))
print(f'\nLoading System Parameters...\n- NN_LR_MEAN: {NN_LR_MEAN}\n- NN_W1 Shape: {NN_W1.shape}\n- NN_W2 Shape: {NN_W2.shape}\n- NN_W3 Shape: {NN_W3.shape}')

class MPCAgent():
    """ MPC algorithm. """

    def __init__(self):
        self.init_lr = copy.deepcopy(NN_LR_MEAN)
        self.init_w3 = copy.deepcopy(NN_W3)
        # pre compile
        state = np.random.randn(N_X) * 1e-8
        u_trj = np.random.randn(MPC_HORIZON, N_U) * 1e-8
        waypoints = np.random.randn(FUTURE_WAYPOINTS_AS_STATE, 2) * 1e-8
        run_ilqr_main(state, u_trj, waypoints) 
        return

    def act(self, state, info={}, sample=False):
        # state[2] += 0.01
        state = jnp.array(state)
        u_trj = np.random.randn(MPC_HORIZON, N_U)*1e-8
        u_trj[:,2] -= jnp.pi/2.5
        u_trj = jnp.array(u_trj)
        
        waypoints = info["waypoints"]
        waypoints = jnp.array(waypoints)
        x_trj, u_trj, cost_trace = run_ilqr_main(state, u_trj, waypoints)
        
        # MPC only executes first control
        steering = np.sin(u_trj[0,0]) # [-1,1]
        throttle = np.sin(u_trj[0,1]) # [-1,1]
        brake = np.sin(u_trj[0,2]) # [-1,1]
        return np.array([steering, throttle, brake]), x_trj, cost_trace

    def set_parameters(self, action):
        w3 = self.init_w3 * ( 1 + 1.0 * np.array(action).reshape(2, 32) )
        # w3 = self.init_w3 * (1 + 2 * np.array(action).reshape(2, 1))

        # set global variable...
        global NN_W3, NN_B3
        NN_W3 = w3
        # NN_B3 = action
        return


@jit
def NN3(x):
    x = jnp.tanh(NN_W1@x)
    x = jnp.tanh(NN_W2@x)
    x = NN_W3@x + NN_B3
    return x

@jit
def continuous_dynamics(state, u):
    # state = [x, y, v, phi, beta, u]

    x = state[0]
    y = state[1]
    v = state[2]
    v_sqrt = jnp.sqrt(v)
    phi = state[3]
    beta = state[4]
    # normalize u
    steering = jnp.sin(u[0])
    throttle_brake = jnp.sin(u[1:])*0.5 + 0.5

    deriv_x = v*jnp.cos(phi+beta)
    deriv_y = v*jnp.sin(phi+beta)
    deriv_phi = v*jnp.sin(beta)/NN_LR_MEAN

    x1 = jnp.hstack((
                v_sqrt,
                jnp.cos(beta), 
                jnp.sin(beta),
                steering,
                throttle_brake
            ))

    x2 = jnp.hstack((
                v_sqrt,
                jnp.cos(beta), 
                -jnp.sin(beta),
                -steering,
                throttle_brake
            ))

    x1 = NN3(x1)
    x2 = NN3(x2)

    deriv_v = ( x1[0]*(2*v_sqrt+x1[0]) + x2[0]*(2*v_sqrt+x2[0]) )/2 # x1[0]+x2[0]
    deriv_beta = ( x1[1] - x2[1] )/2
    derivative = jnp.hstack((deriv_x, deriv_y, deriv_v/DT, deriv_phi, deriv_beta/DT))

    return derivative

@jit
def discrete_dynamics(state, u):
    return state + continuous_dynamics(state, u)*DT

@jit
def rollout(x0, u_trj):
    x_final, x_trj = jax.lax.scan(rollout_looper, x0, u_trj)
    return jnp.vstack((x0, x_trj))
    
@jit
def rollout_looper(x_i, u_i):
    x_ip1 = discrete_dynamics(x_i, u_i)
    return x_ip1, x_ip1

# TODO: remove length if weighing func is not needed
# TODO: LSE function different from claimed
@jit
def distance_func(x, route):
    x, ret = lax.scan(distance_func_looper, x, route)
    return -logsumexp(ret)

@jit
def distance_func_looper(ijnput_, p):
    dp = 1.0
    delta_x = ijnput_[0]-p[0]
    delta_y = ijnput_[1]-p[1]
    return ijnput_, -(delta_x**2.0 + delta_y**2.0)/(1.0*dp**2.0)

# TODO: LSE to calculate each point to all trajectories?
@jit
def cost_1step(x, u, route): # x.shape:(5), u.shape(2)
    global TIME_STEPS_RATIO
    steering = jnp.sin(u[0])
    throttle = jnp.sin(u[1])*0.5 + 0.5
    brake = jnp.sin(u[2])*0.5 + 0.5
    
    c_position = distance_func(x, route)
    c_speed = (x[2]-TARGET_SPEED)**2 
    c_control = (steering**2 + throttle**2 + brake**2 + throttle*brake)
    return (0.04*c_position + 0.002*c_speed + 0.0005*c_control)/TIME_STEPS_RATIO

@jit
def cost_final(x, route): # x.shape:(5), u.shape(2)
    global TARGET_RATIO
    c_position = (x[0]-route[-1,0])**2 + (x[1]-route[-1,1])**2
    c_speed = (x[2] - TARGET_SPEED)**2
    return (c_position/(TARGET_RATIO**2) + 0.0*c_speed)*1

@jit
def cost_trj(x_trj, u_trj, route):
    total = 0.
    total, x_trj, u_trj, route = jax.lax.fori_loop(0, MPC_HORIZON, cost_trj_looper, [total, x_trj, u_trj, route])
    total += cost_final(x_trj[-1], route)
    
    return total

# XXX: check if the cost_1step needs `target`
@jit
def cost_trj_looper(i, ijnput_):
    total, x_trj, u_trj, route = ijnput_
    total += cost_1step(x_trj[i], u_trj[i], route)
    
    return [total, x_trj, u_trj, route]

@jit
def derivative_init():
    jac_l = jit(jacfwd(cost_1step, argnums=[0,1]))
    hes_l = jit(hessian(cost_1step, argnums=[0,1]))
    jac_l_final = jit(jacfwd(cost_final))
    hes_l_final = jit(hessian(cost_final))
    jac_f = jit(jacfwd(discrete_dynamics, argnums=[0,1]))
    
    return jac_l, hes_l, jac_l_final, hes_l_final, jac_f

jac_l, hes_l, jac_l_final, hes_l_final, jac_f = derivative_init()

@jit
def derivative_stage(x, u, route): # x.shape:(5), u.shape(3)
    global jac_l, hes_l, jac_f
    l_x, l_u = jac_l(x, u, route)
    (l_xx, l_xu), (l_ux, l_uu) = hes_l(x, u, route)
    f_x, f_u = jac_f(x, u)

    return l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u

@jit
def derivative_final(x, target):
    global jac_l_final, hes_l_final
    l_final_x = jac_l_final(x, target)
    l_final_xx = hes_l_final(x, target)

    return l_final_x, l_final_xx

@jit
def Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx):
    Q_x = l_x + f_x.T@V_x
    Q_u = l_u + f_u.T@V_x

    Q_xx = l_xx + f_x.T@V_xx@f_x
    Q_ux = l_ux + f_u.T@V_xx@f_x
    Q_uu = l_uu + f_u.T@V_xx@f_u

    return Q_x, Q_u, Q_xx, Q_ux, Q_uu

@jit
def gains(Q_uu, Q_u, Q_ux):
    Q_uu_inv = jnp.linalg.inv(Q_uu)
    k = - Q_uu_inv@Q_u
    K = - Q_uu_inv@Q_ux

    return k, K

@jit
def V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k):
    V_x = Q_x + K.T@Q_u + Q_ux.T@k + K.T@Q_uu@k
    V_xx = Q_xx + 2*K.T@Q_ux + K.T@Q_uu@K

    return V_x, V_xx

@jit
def expected_cost_reduction(Q_u, Q_uu, k):
    return -Q_u.T@k - 0.5 * k.T@Q_uu@k

@jit
def forward_pass(x_trj, u_trj, k_trj, K_trj):
    u_trj = jnp.arcsin(jnp.sin(u_trj))
    
    x_trj_new = jnp.empty_like(x_trj)
    x_trj_new = jax.ops.index_update(x_trj_new, jax.ops.index[0], x_trj[0])
    u_trj_new = jnp.empty_like(u_trj)
    
    x_trj, u_trj, k_trj, K_trj, x_trj_new, u_trj_new = lax.fori_loop(
        0, MPC_HORIZON, forward_pass_looper, [x_trj, u_trj, k_trj, K_trj, x_trj_new, u_trj_new]
    )

    return x_trj_new, u_trj_new

@jit
def forward_pass_looper(i, ijnput_):
    x_trj, u_trj, k_trj, K_trj, x_trj_new, u_trj_new = ijnput_
    
    u_next = u_trj[i] + k_trj[i] + K_trj[i]@(x_trj_new[i] - x_trj[i])
    u_trj_new = jax.ops.index_update(u_trj_new, jax.ops.index[i], u_next)

    x_next = discrete_dynamics(x_trj_new[i], u_trj_new[i])
    x_trj_new = jax.ops.index_update(x_trj_new, jax.ops.index[i+1], x_next)
    
    return [x_trj, u_trj, k_trj, K_trj, x_trj_new, u_trj_new]

@jit
def backward_pass(x_trj, u_trj, regu, target):
    k_trj = jnp.empty_like(u_trj)
    K_trj = jnp.empty((MPC_HORIZON, N_U, N_X))
    expected_cost_redu = 0.
    V_x, V_xx = derivative_final(x_trj[-1], target)
     
    V_x, V_xx, k_trj, K_trj, x_trj, u_trj, expected_cost_redu, regu, target = lax.fori_loop(
        0, MPC_HORIZON, backward_pass_looper, [V_x, V_xx, k_trj, K_trj, x_trj, u_trj, expected_cost_redu, regu, target]
    )
        
    return k_trj, K_trj, expected_cost_redu


@jit
def backward_pass_looper(i, ijnput_):
    V_x, V_xx, k_trj, K_trj, x_trj, u_trj, expected_cost_redu, regu, target = ijnput_
    n = MPC_HORIZON-1-i
    
    l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u = derivative_stage(x_trj[n], u_trj[n], target)
    Q_x, Q_u, Q_xx, Q_ux, Q_uu = Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx)
    Q_uu_regu = Q_uu + jnp.eye(N_U)*regu
    k, K = gains(Q_uu_regu, Q_u, Q_ux)
    k_trj = jax.ops.index_update(k_trj, jax.ops.index[n], k)
    K_trj = jax.ops.index_update(K_trj, jax.ops.index[n], K)
    V_x, V_xx = V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k)
    expected_cost_redu += expected_cost_reduction(Q_u, Q_uu, k)
    
    return [V_x, V_xx, k_trj, K_trj, x_trj, u_trj, expected_cost_redu, regu, target]


@jit
def run_ilqr_main(x0, u_trj, target):
    global jac_l, hes_l, jac_l_final, hes_l_final, jac_f
    
    max_iter = 50 # 300 
    regu = jnp.array(100.)
    
    x_trj = rollout(x0, u_trj)
    cost_trace = jax.ops.index_update(
        jnp.zeros((max_iter+1)), jax.ops.index[0], cost_trj(x_trj, u_trj, target)
    )

    x_trj, u_trj, cost_trace, regu, target = lax.fori_loop(
        1, max_iter+1, run_ilqr_looper, [x_trj, u_trj, cost_trace, regu, target]
    )
    
    return x_trj, u_trj, cost_trace

@jit
def run_ilqr_looper(i, ijnput_):
    x_trj, u_trj, cost_trace, regu, target = ijnput_
    k_trj, K_trj, expected_cost_redu = backward_pass(x_trj, u_trj, regu, target)
    x_trj_new, u_trj_new = forward_pass(x_trj, u_trj, k_trj, K_trj)
    
    total_cost = cost_trj(x_trj_new, u_trj_new, target)
    
    x_trj, u_trj, cost_trace, regu = lax.cond(
        pred = (cost_trace[i-1] > total_cost),
        true_operand = [i, cost_trace, total_cost, x_trj, u_trj, x_trj_new, u_trj_new, regu],
        true_fun = run_ilqr_true_func,
        false_operand = [i, cost_trace, x_trj, u_trj, regu],
        false_fun = run_ilqr_false_func,
    )
    
    max_regu = 10000.0
    min_regu = 0.01
    
    regu += jax.nn.relu(min_regu - regu)
    regu -= jax.nn.relu(regu - max_regu)

    return [x_trj, u_trj, cost_trace, regu, target]

@jit
def run_ilqr_true_func(ijnput_):
    i, cost_trace, total_cost, x_trj, u_trj, x_trj_new, u_trj_new, regu = ijnput_
    
    cost_trace = jax.ops.index_update(
        cost_trace, jax.ops.index[i], total_cost 
    )
    x_trj = x_trj_new
    u_trj = u_trj_new
    regu *= 0.7
    
    return [x_trj, u_trj, cost_trace, regu]

@jit
def run_ilqr_false_func(ijnput_):
    i, cost_trace, x_trj, u_trj, regu = ijnput_
    
    cost_trace = jax.ops.index_update(
        cost_trace, jax.ops.index[i], cost_trace[i-1] 
    )
    regu *= 2.0
    return [x_trj, u_trj, cost_trace, regu]


if __name__ == "__main__":

    from tqdm import tqdm
    import imageio

    # mpc agent init
    agent = MPCAgent()

    # carla init
    is_render = False 
    perturb_spec = {"car_type": "model3"}
    from carla_env import CarlaEnv
    env = CarlaEnv(is_render=is_render, random_seed=2022, task_name="straight", perturb_spec=perturb_spec, port=20000)

    state, info = env.reset()

    # total_time = 0
    frames = []
    mpc_times, env_times = [], []
    for i in tqdm(range(100)):
        t0 = time.time()
        action, x_trj, cost = agent.act(state, info)
        t1 = time.time()
        next_state, reward, done, info = env.step(action, {"x_trj": x_trj})

        agent.set_parameters([0.00]*2)

        t2 = time.time()
        if is_render:
            frames.extend(env.render())

        print(f"=== Step {i} ===")
        print(f"state: {state} action: {action} reward: {reward}") 
        print(f"route error: {info['route_error']} goal error: {info['goal_error']}")
        state = next_state
        mpc_times.append(t1-t0)
        env_times.append(t2-t1)
    
    mpc_time = np.mean(mpc_times)
    env_time = np.mean(env_times)
    print(f"MPC_AVG_TIMES: {mpc_time} ENV_AVG_TIMES: {env_time}\nAVG_TIMES: {mpc_time+env_time} FPS: {1.0/(mpc_time+env_time)}\n")
    env.close()

    if is_render:
        path = "mpc_agent.mp4"
        imageio.mimsave(path, frames, fps=FPS)


