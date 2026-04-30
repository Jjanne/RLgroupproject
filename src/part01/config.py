"""Configuration for the Part 01 MountainCar experiments."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_DIR / "results" / "part01"
MODELS_DIR = RESULTS_DIR / "models"
LOGS_DIR = RESULTS_DIR / "logs"
PLOTS_DIR = RESULTS_DIR / "plots"
TABLES_DIR = RESULTS_DIR / "tables"
TENSORBOARD_DIR = RESULTS_DIR / "tensorboard"


@dataclass(frozen=True)
class ExperimentConfig:
    """Description of one train/evaluate experiment."""

    slug: str
    title: str
    env_id: str
    algorithm: str
    # NOTE: None for neural-net agents (DQN, PPO) that operate on raw observations.
    state_bins: Optional[Tuple[int, int]]
    episodes: int
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.997
    q_init_low: float = 0.0
    q_init_high: float = 0.0
    action_values: Optional[Tuple[float, ...]] = None
    training_wrapper: str = "none"
    wrapper_kwargs: Dict[str, float] = field(default_factory=dict)
    seed: int = 42
    eval_episodes: int = 150
    log_every: int = 250
    description: str = ""
    is_sb3_agent: bool = False  
    total_timesteps: int = 100_000

    @property
    def uses_discrete_actions(self) -> bool:
        return self.action_values is None

    @property
    def has_engineered_reward(self) -> bool:
        return self.training_wrapper != "none"

    @property
    def native_objective_name(self) -> str:
        if self.env_id == "MountainCarContinuous-v0":
            return "fuel-aware reward"
        return "minimum steps"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


PART01_EXPERIMENTS: List[ExperimentConfig] = [
    # ------------------------------------------------------------------
    # Tabular baselines (unchanged)
    # ------------------------------------------------------------------
    ExperimentConfig(
        slug="discrete_q_learning",
        title="Discrete MountainCar - Q-learning baseline",
        env_id="MountainCar-v0",
        algorithm="q_learning",
        state_bins=(40, 40),
        episodes=3500,
        epsilon_decay=0.997,
        description=(
            "Baseline discrete-control solution. Objective: minimise the number "
            "of steps required to reach the flag."
        ),
    ),
    ExperimentConfig(
        slug="discrete_sarsa",
        title="Discrete MountainCar - SARSA baseline",
        env_id="MountainCar-v0",
        algorithm="sarsa",
        state_bins=(40, 40),
        episodes=3500,
        epsilon_decay=0.997,
        q_init_low=-2.0,
        q_init_high=0.0,
        description=(
            "On-policy baseline used to compare how a more conservative update "
            "rule alters stability and policy structure."
        ),
    ),
    ExperimentConfig(
        slug="discrete_directional_cost",
        title="Discrete MountainCar - directional action cost",
        env_id="MountainCar-v0",
        algorithm="q_learning",
        state_bins=(40, 40),
        episodes=4000,
        epsilon_decay=0.9975,
        training_wrapper="directional_cost",
        wrapper_kwargs={"left_cost": 0.18, "right_cost": 0.10},
        description=(
            "Adapted discrete environment where left and right thrust have "
            "different costs, approximating directional actuation effort."
        ),
    ),
    ExperimentConfig(
        slug="discrete_non_null_cost",
        title="Discrete MountainCar - non-null action cost",
        env_id="MountainCar-v0",
        algorithm="q_learning",
        state_bins=(40, 40),
        episodes=4000,
        epsilon_decay=0.9975,
        training_wrapper="uniform_cost",
        wrapper_kwargs={"action_cost": 0.12},
        description=(
            "Adapted discrete environment with a linear penalty on every non-null "
            "action, pushing the policy toward more economical control."
        ),
    ),
    ExperimentConfig(
        slug="continuous_q_learning",
        title="Continuous MountainCar - discretised-action Q-learning",
        env_id="MountainCarContinuous-v0",
        algorithm="q_learning",
        state_bins=(36, 28),
        episodes=5000,
        epsilon_decay=0.998,
        q_init_low=0.0,
        q_init_high=3.0,
        action_values=(-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0),
        training_wrapper="continuous_energy",
        wrapper_kwargs={"energy_scale": 180.0, "progress_scale": 8.0},
        description=(
            "Continuous-action variant solved with a tabular controller over a "
            "discretised thrust set. Training uses energy/progress shaping, while "
            "evaluation is always reported on the native fuel-aware reward."
        ),
    ),
    # ------------------------------------------------------------------
    # Neural-net agents (SB3)
    # ------------------------------------------------------------------
    ExperimentConfig(
        slug="discrete_dqn",
        title="Discrete MountainCar - DQN",
        env_id="MountainCar-v0",        # same discrete env as Q-learning / SARSA
        algorithm="dqn",
        state_bins=None,  
        is_sb3_agent=True,             # raw (position, velocity) fed to the network
        episodes=200_000,               # used as total_timesteps by SB3
        training_wrapper="none",        # no shaping — fair comparison with tabular Q
        description=(
            "Deep Q-Network on the native discrete action space. Direct structural "
            "comparison against tabular Q-learning: same environment and action set, "
            "neural function approximation instead of a Q-table."
        ),
    ),
    ExperimentConfig(
        slug="continuous_ppo",
        title="Continuous MountainCar - PPO",
        env_id="MountainCarContinuous-v0",
        algorithm="ppo",
        state_bins=None,                 
        is_sb3_agent=True,             # raw (position, velocity) fed to the network
        episodes=300_000,               
        training_wrapper="continuous_energy",
        wrapper_kwargs={"energy_scale": 180.0, "progress_scale": 8.0},
        description=(
            "Proximal Policy Optimisation on the continuous action space. "
            "Uses the same energy/progress reward shaping as the tabular continuous "
            "baseline, enabling a direct comparison of tabular vs. policy-gradient "
            "approaches on the same shaped objective."
        ),
    ),
]


def ensure_result_directories() -> None:
    """Create the directories used by the Part 01 pipeline."""

    for path in (RESULTS_DIR, MODELS_DIR, LOGS_DIR, PLOTS_DIR, TABLES_DIR, TENSORBOARD_DIR):
        path.mkdir(parents=True, exist_ok=True)