"""Usage:

(robodiff)$ python demo_real_robot.py -o <demo_save_dir> --robot_ip <ip_of_ur5>

Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start recording.
Press "S" to stop recording.
Press "Q" to exit program.
Press "Backspace" to delete the previously recorded episode.
"""

import time
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
from typing import Any, Dict, List, Tuple

import click
import cv2
import numpy as np
from yixuan_utilities.kinematics_helper import KinHelper

from interactive_world_sim.real_world.aloha_bimanual_master import AlohaBimanualMaster
from interactive_world_sim.real_world.real_aloha_env import RealAlohaEnv
from interactive_world_sim.utils.action_utils import joint_pos_to_action_primitive
from interactive_world_sim.utils.data_sampler import DataSampler
from interactive_world_sim.utils.keystroke_counter import Key, KeyCode, KeystrokeCounter


# The click options with multiple=True usually pass tuples.
# Therefore, we annotate them as Tuple[str, ...] where appropriate.
@click.command()
@click.option(
    "--output_dir", "-o", default=".", help="Directory to save demonstration dataset."
)
@click.option(
    "--robot_sides",
    "-r",
    default=["right", "left"],
    multiple=True,
    help="Which robot to control.",
)
@click.option(
    "--vis_camera_idx", default=0, type=int, help="Which RealSense camera to visualize."
)
@click.option(
    "--frequency", "-f", default=10, type=float, help="Control frequency in Hz."
)
@click.option("--ctrl_mode", "-cm", default="joint", help="Control Mode.")
@click.option(
    "--total_steps", "-ts", default=200, type=int, help="Total steps per episode."
)
def main(
    output_dir: str,
    robot_sides: Tuple[str],
    vis_camera_idx: int,
    frequency: float,
    ctrl_mode: str,
    total_steps: int,
) -> None:
    dt: float = 1 / frequency

    # Create the specified output directories.
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    kin_helper: KinHelper = KinHelper(robot_name="trossen_vx300s")
    data_sampler: DataSampler = DataSampler(
        img_h=480, img_w=640, T_length=120, task=ctrl_mode
    )

    shm_manager = SharedMemoryManager()
    shm_manager.start()
    key_counter = KeystrokeCounter()
    key_counter.start()
    env = RealAlohaEnv(
        output_dir=output_dir,
        robot_sides=list(robot_sides),
        ctrl_mode=ctrl_mode,
        # recording resolution
        obs_image_resolution=(640, 480),
        frequency=frequency,
        enable_multi_cam_vis=True,
        record_raw_video=True,
        video_capture_fps=30,
        video_capture_resolution=(640, 480),
        # number of threads per camera view for video recording (H.264)
        thread_per_video=3,
        # video recording quality, lower is better (but slower).
        video_crf=21,
        shm_manager=shm_manager,
    )
    env.start()
    master_bot = AlohaBimanualMaster(
        shm_manager=shm_manager, robot_sides=list(robot_sides)
    )
    master_bot.start()

    cv2.setNumThreads(1)
    # Optional: adjust RealSense exposure and white balance if needed.
    # env.realsense.set_exposure(exposure=120, gain=0)
    # env.realsense.set_white_balance(white_balance=5900)

    time.sleep(1.0)
    print("Ready!")
    state: Dict[str, Any] = env.get_robot_state()
    iter_idx: int = 0
    stop: bool = False
    is_recording: bool = False
    init_img = data_sampler.sample(idx=env.episode_id)  # warm up
    env.multi_cam_vis.set_goal_img(init_img)
    step_count: int = 0  # Track steps in current episode

    while not stop:
        # Calculate timing values.
        t_start: float = time.monotonic()
        t_cycle_end: float = time.monotonic() + dt

        # Update observation from the environment.
        obs_start_time: float = time.monotonic()
        obs: Dict[str, Any] = env.get_obs_async()
        obs_end_time: float = time.monotonic()

        # Handle key presses from the keystroke counter.
        press_events: List[Any] = key_counter.get_press_events()
        for key_stroke in press_events:
            if key_stroke == KeyCode(char="q"):
                # Exit program.
                stop = True
            elif key_stroke == KeyCode(char="c"):
                # Start recording.
                env.start_episode(
                    t_start - time.monotonic() + time.time(),
                    curr_outdir=output_dir,
                )
                key_counter.clear()
                is_recording = True
                step_count = 0  # Reset step counter for new episode
                print("Recording!")
            elif key_stroke == KeyCode(char="s"):
                # Stop recording.
                env.end_episode(curr_outdir=output_dir, incr_epi=True)
                init_img = data_sampler.sample(idx=env.episode_id)  # warm up
                env.multi_cam_vis.set_goal_img(init_img)
                key_counter.clear()
                is_recording = False
                step_count = 0  # Reset step counter
                print("Stopped.")
            elif key_stroke == KeyCode(char="r"):
                # Re-randomize the T position goal image.
                init_img = data_sampler.sample(idx=env.episode_id)
                env.multi_cam_vis.set_goal_img(init_img)
                key_counter.clear()
                print("Goal image re-randomized!")
            elif key_stroke == Key.backspace:
                # Delete the most recent recorded episode.
                if click.confirm("Are you sure to drop an episode?"):
                    env.drop_episode()
                    key_counter.clear()
                    is_recording = False

        # Auto-save episode after total_steps steps
        if is_recording and step_count >= total_steps:
            env.end_episode(curr_outdir=output_dir, incr_epi=True)
            init_img = data_sampler.sample(idx=env.episode_id)  # warm up
            env.multi_cam_vis.set_goal_img(init_img)
            key_counter.clear()
            is_recording = False
            step_count = 0
            print(f"Episode auto-saved after {total_steps} steps!")

        # Visualization.
        vis_start_time: float = time.monotonic()
        vis_img: np.ndarray = obs[f"camera_{vis_camera_idx}_color"][
            -1, :, :, ::-1
        ].copy()
        episode_id: int = env.episode_id
        text: str = f"Episode: {episode_id}"
        if is_recording:
            text += f", Recording! Step: {step_count}/{total_steps}"
            step_count += 1
        cv2.putText(
            vis_img,
            text,
            (10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            thickness=2,
            color=(255, 255, 255),
        )
        cv2.imshow("default", vis_img)
        cv2.pollKey()
        vis_end_time: float = time.monotonic()

        state_start_time: float = time.monotonic()
        state = master_bot.get_motion_state()
        state_end_time: float = time.monotonic()
        target_state: np.ndarray = state["joint_pos"].copy()

        cvt_start_time: float = time.monotonic()
        actions = joint_pos_to_action_primitive(
            joint_pos=target_state,
            kin_helper=kin_helper,
            ctrl_mode=ctrl_mode,
            base_pose_in_world=env.puppet_bot.base_pose_in_world,
        )
        cvt_end_time: float = time.monotonic()

        # Execute teleoperation command.
        send_action_start_time: float = time.monotonic()
        env.exec_actions(
            joint_actions=target_state[None],
            actions=actions,
            timestamps=np.array([t_cycle_end - time.monotonic() + time.time() + 0.1]),
        )
        send_action_end_time: float = time.monotonic()
        time.sleep(max(0, dt - (time.monotonic() - t_start)))
        real_freq = 1 / (time.monotonic() - t_start)
        if real_freq < frequency - 1:
            print("Warning: Real frequency is too low!")
            print(
                f"Real frequency: {real_freq:.2f} Hz, "
                f"target frequency: {frequency:.2f} Hz"
            )
            print("obs fetch time: ", obs_end_time - obs_start_time)
            print("vis time: ", vis_end_time - vis_start_time)
            print("state fetch time: ", state_end_time - state_start_time)
            print("cvt time: ", cvt_end_time - cvt_start_time)
            print("send action time: ", send_action_end_time - send_action_start_time)
            print("Total time: ", time.monotonic() - t_start)
        iter_idx += 1


if __name__ == "__main__":
    main()
