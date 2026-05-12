import numpy as np
from obj_classes import *
from ray_tracer import benchmark, render_scene, render_scene_over_time
from saved_scenes import benchmark_scene, showcase_scene, empty_scene


#benchmark(benchmark_scene)
render_scene_over_time(showcase_scene)
