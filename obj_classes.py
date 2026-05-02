import numpy as np
from object_type_flags import *


class Scene:
    def __init__(self, cw, ch, vw, vh, d, O, max_rec_depth=3, rays_per_pixel=1):
        self.objects = []
        self.lights = []
        self.max_rec_depth = max_rec_depth
        self.rays_per_pixel = rays_per_pixel

        """Controls image resolution"""
        self.cw: int = cw
        self.ch: int = ch
        self.img = np.ones((ch, cw, 3), dtype=np.float64)

        """Controls FOV"""
        self.vw = vw
        self.vh = vh
        self.d = d

        """Camera position"""
        self.O = O

        """Array for faster runtime, will be set using compile"""
        self.obj_types = None  # Stores the type of each Object
        self.colors = None
        self.emitted_colors = None
        self.emission_strengths = None
        self.smoothnesses = None
        self.is_glass = None
        self.ref_idxs = None

        self.sphere_centers = None
        self.sphere_radii = None


    def add_objects(self, *args):
        for obj in args:
            self.objects.append(obj)


    def add_lights(self, *args):
        for light in args:
            self.lights.append(light)


    def compile(self):
        """Turns the object data into lists for performance improvement"""
        """General object data"""
        obj_types = np.zeros(len(self.objects), dtype=np.int32)
        colors = np.zeros((len(self.objects), 3), dtype=np.float64)
        emitted_colors = np.zeros((len(self.objects), 3), dtype=np.float64)
        emission_strengths = np.zeros(len(self.objects), dtype=np.float64)
        smoothnesses = np.zeros(len(self.objects), dtype=np.float64)
        is_glass = np.zeros(len(self.objects), dtype=bool)
        ref_idxs = np.ones(len(self.objects), dtype=np.float64)

        """Sphere data"""
        sphere_centers = np.zeros((len(self.objects), 3))
        sphere_radii = np.zeros(len(self.objects))

        for i, obj in enumerate(self.objects):

            colors[i] = obj.color
            emitted_colors[i] = obj.emitted_color
            emission_strengths[i] = obj.emission_strength
            smoothnesses[i] = obj.smoothness
            is_glass[i] = obj.is_glass
            ref_idxs[i] = obj.ref_idx


            if type(obj) == Sphere:
                obj_types[i] = SPHERE
                sphere_radii[i] = obj.r
                sphere_centers[i] = obj.C

        self.obj_types = obj_types
        self.colors = colors
        self.emitted_colors = emitted_colors
        self.emission_strengths = emission_strengths
        self.smoothnesses = smoothnesses
        self.is_glass = is_glass
        self.ref_idxs = ref_idxs

        self.sphere_centers = sphere_centers
        self.sphere_radii = sphere_radii


class Sphere:
    def __init__(self, center: np.ndarray, radius: float, color=np.array([0, 0, 0]), emitted_color=np.array([0, 0, 0]), emission_strength=0, smoothness=0, is_glass=False, ref_idx=1):
        self.C = center
        self.r = radius
        self.color = color / 255
        self.emitted_color = emitted_color / 255
        self.emission_strength = emission_strength
        self.smoothness = smoothness
        self.is_glass = is_glass
        self.ref_idx = ref_idx
