import numpy as np
from obj_classes import *

"""Initialize scene"""
benchmark_scene = Scene(
    cw = 450,
    ch = 300,
    vw = 1.5,
    vh = 1,
    d = 1,
    O = np.array([0, 0, 0], dtype=np.float64),
    max_rec_depth=4,
    rays_per_pixel=10
)

benchmark_scene.add_objects(
    Sphere(
            center=np.array([-8, -4, 20]),
            radius=1,
            color=np.array([255, 0, 0]), # Red
           ),
    Sphere(
            center=np.array([10, -1, 20]),
            radius=4,
            color=np.array([255, 255, 255]), # White
            smoothness=1 # reflective
           ),
    Sphere(
            center=np.array([0, -2, 16]),
            radius=2,
            color=np.array([0, 255, 0]), # Green
            smoothness=0
           ),
    Sphere(
            center=np.array([3, -54, 20]),
            radius=50,
            color=np.array([255, 255, 0]), # Yellow
        ),
    Sphere(
        center=np.array([-70, 40, 100]),
        radius=80,
        color=np.array([0, 0, 0]),
        emitted_color=np.array([255, 255, 255]),
        emission_strength=2 # Light source
        )
)