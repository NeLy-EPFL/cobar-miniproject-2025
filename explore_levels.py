import argparse
import cv2
import tqdm
from cobar_miniproject import levels
from cobar_miniproject.keyboard_controller import KeyBoardController
from cobar_miniproject.cobar_fly import CobarFly
from cobar_miniproject.vision import (
    get_fly_vision,
    get_fly_vision_raw,
    render_image_with_vision,
)
from flygym import YawOnlyCamera, SingleFlySimulation
from flygym.arena import FlatTerrain

import pygame
import sys

# Set a background color (R, G, B)
background_color = (50, 150, 200)

# OPTIONS
# what to display as the simulation is running
ONLY_CAMERA = 0
WITH_FLY_VISION = 1
WITH_RAW_VISION = 2

VISUALISATION_MODE = WITH_FLY_VISION

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the fly simulation.")
    parser.add_argument(
        "--level", type=int, default=4, help="Level to load (default: -1)"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for the simulation (default: 0)"
    )
    args = parser.parse_args()

    level = args.level
    seed = args.seed
    timestep = 1e-4

    # you can pass in parameters to enable different senses here
    fly = CobarFly(debug=False, enable_vision=True, render_raw_vision=False)

    if level <= -1:
        level_arena = FlatTerrain()
    elif level <= 1:
        # levels 0 and 1 don't need the timestep
        level_arena = levels[level](fly=fly, seed=seed)
    else:
        # levels 2-4 need the timestep
        level_arena = levels[level](fly=fly, timestep=timestep, seed=seed)

    cam = YawOnlyCamera(
        attachment_point=fly.model.worldbody,
        camera_name="camera_back_track_game",
        targeted_fly_names=[fly.name],
        play_speed=0.2,
    )

    # Initialize Pygame
    pygame.init()

    # Set up the display
    screen_width = 640
    screen_height = 752
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Simple Pygame Display")
    screen.fill(background_color)

    sim = SingleFlySimulation(
        fly=fly,
        cameras=[cam],
        timestep=timestep,
        arena=level_arena,
    )

    controller = KeyBoardController(timestep=timestep, seed=seed)

    # run cpg simulation
    obs, info = sim.reset()
    obs_hist = []
    info_hist = []

    # create window
    # cv2.namedWindow("Simulation", cv2.WINDOW_NORMAL)

    with tqdm.tqdm(desc="running simulation") as progress_bar:
        while True:
            # Get observations
            obs, reward, terminated, truncated, info = sim.step(
                controller.get_actions(obs)
            )
            if controller.done_level(obs):
                # finish the path integration level
                break

            obs_hist.append(obs)
            info_hist.append(info)

            rendered_img = sim.render()[0]
            if rendered_img is not None:
                if VISUALISATION_MODE == WITH_FLY_VISION:
                    rendered_img = render_image_with_vision(
                        rendered_img, get_fly_vision(fly), obs["odor_intensity"],
                    )
                elif VISUALISATION_MODE == WITH_RAW_VISION:
                    rendered_img = render_image_with_vision(
                        rendered_img, get_fly_vision_raw(fly), obs["odor_intensity"],
                    )
                
                rendered_img = cv2.cvtColor(rendered_img, cv2.COLOR_BGR2RGB)
                # Fill the screen with the background color
                pygame.surfarray.blit_array(screen, rendered_img.swapaxes(0, 1))
                # cv2.imshow("Simulation", rendered_img)
                # cv2.waitKey(1)

            if controller.quit:
                print("Simulation terminated by user.")
                break
            if hasattr(level_arena, "quit") and level_arena.quit:
                print("Target reached. Simulation terminated.")
                break

            progress_bar.update()
            # Update the display
            pygame.display.flip()

    pygame.quit()
    print("Simulation finished")

    # Save video
    cam.save_video("./outputs/hybrid_controller.mp4", 0)
