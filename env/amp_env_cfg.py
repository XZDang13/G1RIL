import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

from .G1_23_DOF_CFG import G1_CFG

@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.0),
            "dynamic_friction_range": (0.2, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_torso_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-2.5, 4.0),
            "operation": "add",
        },
    )

    add_pelvis_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

@configclass
class G1AMPEnvCfg(DirectRLEnvCfg):
    expert_motion_file = None
    episode_length_s = 10.0

    decimation = 2

    observation_space = 75
    privilege_observation_space = 98
    motion_space = 98
    motion_buffer_size = 2
    motion_observation_space = motion_space * motion_buffer_size
    action_space = 23
    state_space = 0

    reference_body = "pelvis"

    expert_motion_file = None

    early_termination = True
    termination_height = 0.5

    training = True
    add_noise = True

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene:InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True
    )

    robot:ArticulationCfg = G1_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    #events: EventCfg = EventCfg()

@configclass
class G1WalkEnvCfg(G1AMPEnvCfg):
    expert_motion_file = "env/motion_data/walk.npz"
    observation_space = 75
    privilege_observation_space = 98
    motion_space = 98
    motion_buffer_size = 2
    motion_observation_space = motion_space * motion_buffer_size
    
@configclass
class G1DanceEnvCfg(G1AMPEnvCfg):
    expert_motion_file = "env/motion_data/chacha.npz"
    observation_space = 75
    privilege_observation_space = 98
    motion_space = 98
    motion_buffer_size = 2
    motion_observation_space = motion_space * motion_buffer_size
    episode_length_s = 10