
import carla

client = carla.Client("localhost", 20000)
world = client.load_world("Town01")

world.apply_settings(carla.WorldSettings(
            no_rendering_mode=True,
            synchronous_mode=True,
            fixed_delta_seconds=0.01)
        )
mmap = world.get_map()

print("get_world succeeds")
print(world.get_weather())
print(mmap)
