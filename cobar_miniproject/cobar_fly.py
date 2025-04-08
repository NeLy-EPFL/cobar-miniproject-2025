from flygym import Fly, Simulation
import numpy as np
import gymnasium as gym
from flygym.arena import BaseArena

class CobarFly(Fly):
    def __init__(
        self, debug=False, enable_vision=True, render_raw_vision=False, **kwargs
    ):
        """Specific Fly instance for use in the COBAR project.
        The physics have been to improve simulation speed.

        Args:
            debug (bool, optional): If this is true, get the raw observations from the fly. Don't use this when running your controller! Defaults to False.
            enable_vision (bool, optional): Whether the fly-perspective vision is rendered. This can be disabled if not needed to increase simulation speed. Defaults to True.
            render_raw_vision (bool, optional): Whether the raw fly-perspective vision is rendered. This can be disabled if not needed to increase simulation speed. Defaults to False.
        """
        super().__init__(
            contact_sensor_placements=[
                f"{side}{pos}{segment}"
                for side in "LR"
                for pos in "FMH"
                for segment in ["Tarsus5"]
            ],
            self_collisions="none",
            floor_collisions=[
                f"{side}{pos}{dof}"
                for side in "LR"
                for pos in "FMH"
                for dof in ["Tarsus5"]
            ],
            xml_variant="seqik_simple",
            head_stabilization_model="thorax",
            neck_kp=1000,
            enable_adhesion=True,
            enable_olfaction=True,
            enable_vision=enable_vision,
            render_raw_vision=render_raw_vision,
            **kwargs,
        )
        self.debug = debug

        # add these joints to enable using head stabilisation models
        fly_head_body = self.model.worldbody.find("body", "Head")  # type: ignore
        fly_head_body.add(
            "joint",
            **{
                "name": "joint_Head_yaw",
                "class": "nmf",
                "type": "hinge",
                "pos": "0 0 0",
                "axis": "1 0 0",
                "stiffness": str(self.neck_stiffness),
                "springref": "0.0",
                "damping": str(self.non_actuated_joint_damping),
                "frictionloss": "0.0",
            },
        )
        fly_head_body.add(
            "joint",
            **{
                "name": "joint_Head",
                "class": "nmf",
                "type": "hinge",
                "pos": "0 0 0",
                "axis": "0 1 0",
                "stiffness": str(self.neck_stiffness),
                "springref": "0.0",
                "damping": str(self.non_actuated_joint_damping),
                "frictionloss": "0.0",
            },
        )
        fly_head_body.add(
            "joint",
            **{
                "name": "joint_Head_roll",
                "class": "nmf",
                "type": "hinge",
                "pos": "0 0 0",
                "axis": "0 0 1",
                "stiffness": str(self.neck_stiffness),
                "springref": "0.0",
                "damping": str(self.non_actuated_joint_damping),
                "frictionloss": "0.0",
            },
        )
        self.model.visual.map.zfar = 10  # type: ignore

        self._geoms_to_hide = [
            "A1A2",
            "A3",
            "A4",
            "A5",
            "A6",
            "Haustellum",
            "Head",
            "LArista",
            "LEye",
            "LFCoxa",
            "LFFemur",
            "LFTarsus1",
            "LFTarsus2",
            "LFTarsus3",
            "LFTarsus4",
            "LFTarsus5",
            "LFTibia",
            "LFuniculus",
            "LHCoxa",
            "LHFemur",
            "LHTarsus1",
            "LHTarsus2",
            "LHTarsus3",
            "LHTarsus4",
            "LHTarsus5",
            "LHTibia",
            "LHaltere",
            "LMCoxa",
            "LMFemur",
            "LMTarsus1",
            "LMTarsus2",
            "LMTarsus3",
            "LMTarsus4",
            "LMTarsus5",
            "LMTibia",
            "LPedicel",
            "LWing",
            "RArista",
            "REye",
            "RFCoxa",
            "RFFemur",
            "RFTarsus1",
            "RFTarsus2",
            "RFTarsus3",
            "RFTarsus4",
            "RFTarsus5",
            "RFTibia",
            "RFuniculus",
            "RHCoxa",
            "RHFemur",
            "RHTarsus1",
            "RHTarsus2",
            "RHTarsus3",
            "RHTarsus4",
            "RHTarsus5",
            "RHTibia",
            "RHaltere",
            "RMCoxa",
            "RMFemur",
            "RMTarsus1",
            "RMTarsus2",
            "RMTarsus3",
            "RMTarsus4",
            "RMTarsus5",
            "RMTibia",
            "RPedicel",
            "RWing",
            "Rostrum",
            "Thorax",
        ]

    def get_observation(self, sim: Simulation):
        observation = super().get_observation(sim)

        # if we're running the fly in debug mode (for development) it will return all raw observations
        # otherwise, it will return only a reduced observation space with egocentric observations
        if self.debug:
            return observation

        end_effector_positions_relative = CobarFly.absolute_to_relative_pos(
            observation["end_effectors"][:, :2],
            observation["fly"][0, :2],
            observation["fly_orientation"],
        )
        velocities_relative = CobarFly.absolute_to_relative_pos(
            observation["fly"][1, :2],
            observation["fly"][0, :2],
            observation["fly_orientation"],
        )

        observation_to_return = {
            "joints": observation["joints"],
            "end_effectors": end_effector_positions_relative,
            "contact_forces": observation["contact_forces"],
            "heading": np.array(np.arctan2(
                observation["fly_orientation"][1], observation["fly_orientation"][0]
            )).reshape(1),
            "velocity": velocities_relative,
        }

        if self.enable_olfaction:
            observation_to_return["odor_intensity"] = observation["odor_intensity"]

        if self.enable_vision:
            observation_to_return["vision"] = observation["vision"]
            if self.render_raw_vision:
                observation_to_return["raw_vision"] = self.get_info()["raw_vision"]

        return observation_to_return

    def post_step(self, sim: Simulation):
        obs = self.get_observation(sim)
        reward = self.get_reward()
        terminated = self.is_terminated()
        truncated = self.is_truncated()
        info = self.get_info()

        if self.enable_vision:
            vision_updated_this_step = sim.curr_time == self._last_vision_update_time
            self._vision_update_mask.append(vision_updated_this_step)
            info["vision_updated"] = vision_updated_this_step

        # Fly has flipped if the z component of the "up" cardinal vector is negative
        cardinal_vector_z = sim.physics.bind(self._body_sensors[6]).sensordata.copy()
        info["flip"] = cardinal_vector_z[2] < 0

        if self.head_stabilization_model is not None:
            # this is tracked to decide neck actuation for the next step
            self._last_observation = obs
            info["neck_actuation"] = self._last_neck_actuation

        return obs, reward, terminated, truncated, info

    @staticmethod
    def absolute_to_relative_pos(
        pos: np.ndarray, base_pos: np.ndarray, heading: np.ndarray
    ) -> np.ndarray:
        """
        This function converts an absolute position to a relative position
        with respect to the base position and heading of the fly.

        It is used to obtain the fly-centric end effector (leg tip) positions.

        """
        rel_pos = pos - base_pos.reshape(1, -1)
        heading = heading / np.linalg.norm(heading)
        angle = np.arctan2(heading[1], heading[0])
        rot_matrix = np.array(
            [[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]]
        )
        pos_rotated = np.dot(rel_pos, rot_matrix.T)
        return pos_rotated

    def _define_observation_space(self, arena: BaseArena):
        _observation_space = {
            "joints": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(3, len(self.actuated_joints))
            ),
            "end_effectors": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6, 2)),
            "contact_forces": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(len(self.contact_sensor_placements), 3)
            ),
            "heading": gym.spaces.Box(
                low=-np.pi, high=np.pi, shape=(1,)
            ),
            "velocity": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1, 2)
            ),
        }
        if self.enable_olfaction:
            _observation_space["odor_intensity"] = gym.spaces.Box(
                low=0,
                high=np.inf,
                shape=(2, len(self._antennae_sensors)),
            )
        if self.enable_vision:
            _observation_space["vision"] = gym.spaces.Box(
                low=0,
                high=255,
                shape=(2, self.config["vision"]["num_ommatidia_per_eye"], 2),
            )
            if self.render_raw_vision:
                _observation_space["raw_vision"] = gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(2, self.config["vision"]["raw_img_height_px"], self.config["vision"]["raw_img_width_px"], 3),
                )
        return gym.spaces.Dict(_observation_space)
