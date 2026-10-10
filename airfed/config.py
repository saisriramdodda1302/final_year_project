"""All settings in one place.

Values marked "paper" come straight from Table II / Section V of the AirFed paper.
Values marked "ours" fill gaps the paper leaves open; the README explains each one.
"""

from dataclasses import dataclass, field, replace


@dataclass
class EnvConfig:
    # --- network and area (paper) ---
    area: float = 1000.0                  # side of the square service area, metres
    n_uavs: int = 6                       # K
    n_devices: int = 40                   # M
    alt_range: tuple = (80.0, 150.0)      # each UAV flies at a fixed altitude in this range
    cpu_range: tuple = (1e9, 3e9)         # UAV CPU frequency f_k, Hz
    v_max: float = 20.0                   # m/s
    accel: float = 5.0                    # m/s^2
    battery: float = 500e3                # J
    r_comm: float = 400.0                 # max UAV-to-UAV link distance, m
    episode_seconds: float = 300.0

    # --- radio (paper) ---
    rssi_min_dbm: float = -90.0           # a device is covered above this
    g0_db: float = -30.0                  # air-to-ground channel gain at 1 m
    g_inter_db: float = -20.0             # air-to-air channel gain at 1 m
    bw: float = 10e6                      # device <-> UAV bandwidth, Hz
    bw_inter: float = 20e6                # UAV <-> UAV bandwidth, Hz
    noise_dbm: float = -114.0
    p_uav_tx: float = 0.5                 # W
    p_uav_rx: float = 0.1                 # W
    p_dev_tx: float = 0.1                 # W

    # --- flight and compute energy (paper) ---
    p_hover: float = 80.0                 # W
    rho_air: float = 1.225                # kg/m^3
    drag_area: float = 0.1                # m^2
    drag_coef: float = 0.3

    # --- ours: gaps or fixes, see README "Where the paper was unclear" ---
    dt: float = 10.0                      # step length; Eq. 38 needs dt >= 2*v_max/a = 8 s
    ground_exponent: float = 3.5          # air-to-ground path-loss exponent (paper's 2 covers 22 km)
    kappa: float = 1e-28                  # switched capacitance (paper's 1e-18 gives ~500 MJ per task)
    p_idle: float = 2.0                   # W while a task waits in the queue
    t_decision: float = 0.005             # s spent deciding at each UAV on the path
    load_max_steps: float = 2.0           # a UAV refuses work once its backlog exceeds this many steps

    # --- tasks (paper) ---
    rate_range: tuple = (0.3, 0.8)        # Poisson arrival rate per device, tasks/s
    cycles_range: tuple = (50e6, 200e6)   # w_l
    input_mb: tuple = (1.0, 3.0)          # s_l
    output_mb: tuple = (0.1, 0.5)         # d_l
    deadline_range: tuple = (5.0, 20.0)   # c_l, seconds

    # --- reward (paper, Eqs. 31 and 67-70) ---
    alpha: float = 0.5                    # weight on time
    beta: float = 0.5                     # weight on energy
    deadline_penalty: float = 10.0        # lambda
    coverage_reward: float = 0.1          # eta

    coverage_credit: str = "covered"      # "covered" = Eq. 70 as written; "served" = only the serving UAV

    # --- ours: reward normalisation bounds ---
    time_ref: float = 20.0                # longest deadline; completion times are divided by this

    @property
    def steps(self) -> int:
        return int(round(self.episode_seconds / self.dt))


@dataclass
class ModelConfig:
    # paper, Table II
    gat_hidden: tuple = (128, 64)
    heads: int = 4
    gru_hidden: int = 128
    shared: int = 128
    actor_hidden: tuple = (128, 128)
    critic_hidden: tuple = (128, 64)
    gamma_urgency: float = 0.5            # gamma_urg in Eq. 54
    alpha_speed: float = 0.1              # graph refresh speed-up in Algorithm 1

    # ours
    graph_dt_base: float = 20.0           # base graph refresh interval, s
    init_log_std: float = -2.0            # starting exploration noise of the velocity actor (std ~0.14)


@dataclass
class TrainConfig:
    # paper
    lr_vel: float = 3e-4
    lr_off: float = 3e-4
    lr_critic: float = 5e-4
    gamma: float = 0.95
    entropy: float = 0.01

    # ours
    lr_encoder: float = 3e-4              # GAT + GRU + shared layer
    update_every: int = 1                 # update after every step, as in the paper's one-step advantage
    grad_clip: float = 1.0
    gae_lambda: float = 0.0               # 0 = the paper's one-step advantage; >0 = GAE inside the window
    keep_best: bool = True                # hand back the snapshot with the best 5-episode training reward


@dataclass
class FedConfig:
    # paper, Section IV-C and Table II
    rssi_fl_dbm: float = -85.0
    b_min: int = 4
    b_max: int = 16
    f_base: float = 0.03                  # Hz
    alpha_mobility: float = 0.05
    alpha_succ: float = 0.6
    alpha_stab: float = 0.4
    rho: float = 0.75                     # reputation smoothing

    # ours: chance an exchange is lost grows with distance
    drop_base: float = 0.02
    drop_far: float = 0.2


@dataclass
class Variant:
    """Switches for AirFed and its ablations (paper Fig. 5)."""
    name: str = "airfed"
    use_gat: bool = True
    use_fed: bool = True
    use_reputation: bool = True
    use_quant: bool = True


VARIANTS = {
    "airfed": Variant("airfed"),
    "no_gat": Variant("no_gat", use_gat=False),
    "no_reputation": Variant("no_reputation", use_reputation=False),
    "no_fed": Variant("no_fed", use_fed=False),
    "no_quant": Variant("no_quant", use_quant=False),
    # a plain independent learner: MLP encoder, no federation
    "independent": Variant("independent", use_gat=False, use_fed=False),
}


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    fed: FedConfig = field(default_factory=FedConfig)

    def with_env(self, **kw) -> "Config":
        return replace(self, env=replace(self.env, **kw))
