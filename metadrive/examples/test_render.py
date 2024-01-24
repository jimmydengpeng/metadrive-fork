from metadrive import MetaDriveEnv
from metadrive.examples import expert
import time

def test_single_env(save_gif=False):
    # Set the envrionment config
    config = {"start_seed": 1000, 
            "environment_num":1,
            "traffic_density":0.1,
            "render_mode": "top_down", 
            }

    env = MetaDriveEnv(config)

    print("Starting the environment ...\n")

    ep_reward = 0.0
    obs, info = env.reset()
    frames = []
    for i in range(1000):
        obs, reward, terminated, truncated, info = env.step(expert(env.vehicle))
        ep_reward += reward
        frame = env.render(film_size=(800, 800), track_target_vehicle=True, screen_size=(500, 500))
        frames.append(frame)
        if terminated or truncated:
            print("Arriving Destination: {}".format(info["arrive_dest"]))
            print("\nEpisode reward: ", ep_reward)
            break

    print("\nThe last returned information: {}".format(info))

    env.close()
    print("\nMetaDrive successfully run!")

    if save_gif:
        # render image
        print("\nGenerate gif...")
        import pygame
        import numpy as np
        from PIL import Image

        imgs = [pygame.surfarray.array3d(frame) for frame in frames]
        imgs = [Image.fromarray(img) for img in imgs]
        imgs[0].save("demo.gif", save_all=True, append_images=imgs[1:], duration=50, loop=0)
        print("\nOpen gif...")
        from IPython.display import Image
        Image(open("demo.gif", 'rb').read())



def test_multi_agent_env(save_gif=False):
    from metadrive import MultiAgentRoundaboutEnv, MultiAgentBottleneckEnv, MultiAgentIntersectionEnv, MultiAgentParkingLotEnv, MultiAgentTollgateEnv

    env_classes = [
        MultiAgentIntersectionEnv, 
        MultiAgentRoundaboutEnv, 
        MultiAgentBottleneckEnv, 
        MultiAgentParkingLotEnv, 
        MultiAgentTollgateEnv]

    frames = []
    for env_class in env_classes:
        t0 = time.time()
        env = env_class({
            "preload_models": False, 
            "render_mode": "top_down",
        })
        print("Starting the environment {}\n".format(env))
        env.reset()
        t1 = time.time()
        print('>>> reset() time:', t1-t0)
        # exit()
        tm={"__all__":False}
        for i in range(100):
            if tm["__all__"]:
                # frames.append(frame)
                continue
            action = env.action_space.sample()
            for a in action.values(): 
                a[-1] = 1.0
            o,r,tm,tc,i = env.step(action)
            # film_size: 镜头大小(远近), 越大镜头越近，越小镜头越远
            # screen_size: pygame窗口大小
            frame = env.render(film_size=(800, 800), track_target_vehicle=False, screen_size=(800, 800))
            frames.append(frame)
        env.close()
        # break

    if save_gif:
        # render image
        print("\nGenerate gif...")
        import pygame
        import numpy as np
        from PIL import Image

        imgs = [pygame.surfarray.array3d(frame) for frame in frames]
        imgs = [Image.fromarray(img) for img in imgs]
        imgs[0].save("demo.gif", save_all=True, append_images=imgs[1:], duration=50, loop=0)
        print("\nOpen gif...")
        from IPython.display import Image
        # Image(open("demo.gif", 'rb').read())


if __name__ == '__main__':
    # test_single_env()
    test_multi_agent_env()