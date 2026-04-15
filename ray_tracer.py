import numpy as np
import matplotlib.pyplot as plt
import cProfile
import math
from numba import njit
from object_type_flags import *
from light_type_flags import *


class Scene:
    def __init__(self, cw, ch, vw, vh, d, O, bgcol=np.array([0, 0, 0], dtype=np.float64), max_rec_depth=3):
        self.objects = []
        self.lights = []
        self.bgcol = bgcol
        self.bgcol /= 255
        self.max_rec_depth = max_rec_depth

        """Controls image resolution"""
        self.cw: int = cw
        self.ch: int = ch
        self.img = np.full((ch, cw, 3), 255, dtype=np.float64)

        """Controls FOV"""
        self.vw = vw
        self.vh = vh
        self.d = d

        """Camera position"""
        self.O = O

        """Array for faster runtime, will be set using compile"""
        self.obj_types = None  # Stores the type of each Object
        self.colors = None
        self.speculars = None
        self.reflectives = None

        self.sphere_centers = None
        self.sphere_radii = None

        self.light_types = None
        self.light_intensities = None
        self.light_directions = None
        self.light_positions = None


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
        speculars = np.zeros(len(self.objects), dtype=np.float64)
        reflectives = np.zeros(len(self.objects), dtype=np.float64)

        """Sphere data"""
        sphere_centers = np.zeros((len(self.objects), 3))
        sphere_radii = np.zeros(len(self.objects))

        for i, obj in enumerate(self.objects):

            colors[i] = obj.color / 255
            speculars[i] = obj.specular
            reflectives[i] = obj.reflective

            if type(obj) == Sphere:
                obj_types[i] = SPHERE
                sphere_radii[i] = obj.r
                sphere_centers[i] = obj.C

        self.types = obj_types
        self.colors = colors
        self.speculars = speculars
        self.reflectives = reflectives

        self.sphere_centers = sphere_centers
        self.sphere_radii = sphere_radii


        """Light data"""
        light_types = np.zeros(len(self.lights), dtype=np.int32)
        light_intensities = np.zeros(len(self.lights), dtype=np.float64)
        light_directions = np.zeros((len(self.lights), 3), dtype=np.float64)
        light_positions = np.zeros((len(self.lights), 3), dtype=np.float64)

        for i, light in enumerate(self.lights):
            light_intensities[i] = light.intensity

            if light.type == "ambient":
                light_types[i] = AMBIENT

            elif light.type == "point":
                light_types[i] = POINT
                light_positions[i] = light.position

            elif light.type == "directional":
                light_types[i] = DIRECTIONAL
                light_directions[i] = light.direction

            else:
                raise TypeError("Invalid light type encountered during compilation")

        self.light_types = light_types
        self.light_intensities = light_intensities
        self.light_directions = light_directions
        self.light_positions = light_positions


class Sphere:
    def __init__(self, center: np.ndarray, radius: float, color=np.array([0, 0, 0]), specular=-1, reflective=0):
        self.C = center
        self.r = radius
        self.color = color
        self.specular = specular
        self.reflective = reflective


class Light:
    def __init__(self, type: str, intensity, position=None, direction=None):
        self.type = type
        self.intensity = intensity
        self.position = position
        self.direction = direction


def canvas_to_screen(cx: int, cy: int, scene) -> tuple[int]:
    """Converts canvas coordinates to screen coordinates"""
    cx += scene.cw // 2
    cy *= -1
    cy += scene.ch // 2
    return (cx, cy)


def put_pixel(cx: int, cy: int, scene: Scene, col: np.ndarray[3]) -> None:
    """Sets the color of a single pixel"""
    cx, cy = canvas_to_screen(cx, cy, scene)
    scene.img[cy, cx] = col
    return


def canvas_to_viewport(cx: int, cy: int, scene: Scene) -> np.ndarray:
    """Converts canvas coordinates to viewport coordinates"""
    if (scene.d != 1):
        raise ValueError("This function assumes d = 1")

    vx = cx * scene.vw / scene.cw
    vy = cy * scene.vh / scene.ch
    vz = scene.d
    return np.array([vx, vy, vz])


@njit
def get_normal_vector_sphere(C: np.ndarray[3], P: np.ndarray[3]) -> np.ndarray[3]:
    N = P - C
    N /= math.sqrt(N[0] * N[0] + N[1] * N[1] + N[2] * N[2])
    return N


@njit
def find_intersections_sphere(C: np.ndarray[3], r: float, O, D) -> np.ndarray:
    """Finds the possible scalars for the ray. Invalid solutions return np.nan"""
    CO = O - C
    a = np.dot(D, D)
    b = 2 * np.dot(CO, D)
    c = np.dot(CO, CO) - r ** 2

    l1 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    l2 = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)

    return np.array([l1, l2])


@njit
def closest_intersection(O: np.ndarray[3], D: np.ndarray[3], obj_types, sphere_centers, sphere_radii, t_min, t_max) -> tuple:
    """
    Finds the closest intersection between a ray and any object
    :return: The intersected object and the t used to intersect
    """
    closest_obj = -1
    closest_t = np.inf

    for obj in range(len(obj_types)):
        if obj_types[obj] == SPHERE:
            intersections = find_intersections_sphere(sphere_centers[obj], sphere_radii[obj], O, D)

        for t in intersections:
            if np.isnan(t) or t < t_min or t > t_max:
                continue

            if t < closest_t:
                closest_obj = obj
                closest_t = t

    return closest_obj, closest_t


@njit
def exists_intersection(O: np.ndarray[3], D: np.ndarray[3], obj_types, sphere_centers, sphere_radii, t_min, t_max) -> tuple:
    """
    Finds the closest intersection between a ray and any object
    :return: The intersected object and the t used to intersect
    """
    for obj in range(len(obj_types)):
        if obj_types[obj] == SPHERE:
            intersections = find_intersections_sphere(sphere_centers[obj], sphere_radii[obj], O, D)

        for t in intersections:
            if np.isnan(t) or t < t_min or t > t_max:
                continue
                continue

            return True

    return False


@njit
def trace_ray(
        O,
        D,
        bgcol,
        max_rec_depth,
        obj_types,
        colors,
        speculars,
        reflectives,
        sphere_centers,
        sphere_radii,
        light_types,
        light_intensities,
        light_directions,
        light_positions,
        t_min=0,
        t_max=np.inf,
        recursion_depth=0
) -> np.ndarray[3]:

    """Traces the rays path and returns the closest intersected Object"""
    closest_obj, closest_t = closest_intersection(O, D, obj_types, sphere_centers, sphere_radii, t_min, t_max)

    if closest_obj == -1:
        return bgcol

    P = O + D * closest_t

    if obj_types[closest_obj] == SPHERE:
        N = get_normal_vector_sphere(sphere_centers[closest_obj], P)

    V = -D
    local_color = colors[closest_obj] * compute_lighting(P, N, V, speculars[closest_obj], obj_types, sphere_centers, sphere_radii, light_types, light_intensities, light_directions, light_positions)

    # If object is not reflecctive, or we hit recursion limit, stop
    r = reflectives[closest_obj]
    if r <= 0 or recursion_depth >= max_rec_depth:
        return local_color

    R = reflect_ray(V, N)
    reflected_color = trace_ray(
                                P,
                                R,
                                bgcol,
                                max_rec_depth,
                                obj_types,
                                colors,
                                speculars,
                                reflectives,
                                sphere_centers,
                                sphere_radii,
                                light_types,
                                light_intensities,
                                light_directions,
                                light_positions,
                                t_min=0.001,
                                t_max=np.inf,
                                recursion_depth=recursion_depth + 1
    )

    return local_color * (1 - r) + reflected_color * r


@njit
def compute_lighting(P,
                     N,
                     V,
                     s,
                     obj_types,
                     sphere_centers,
                     sphere_radii,
                     light_types,
                     light_intensities,
                     light_directions,
                     light_positions
) -> float:
    """
    Takes in a point, the surface normal at that point and of course the scene
    and computes the intensity of the reflected light
    """
    i: float = 0
    for light in range(len(light_types)):
        if light_types[light] == AMBIENT:
            i += light_intensities[light]

        else:
            if light_types[light] == POINT:
                L = light_positions[light] - P
                t_max = 1.001

            elif light_types[light] == DIRECTIONAL:
                L = light_directions[light]
                t_max = np.inf

            # Shadow check
            if exists_intersection(P, L, obj_types, sphere_centers, sphere_radii, t_min=0.001, t_max=t_max):
                continue

            # Diffuse
            cos_a = np.dot(N, L)
            norm_L = math.sqrt(L[0] * L[0] + L[1] * L[1] + L[2] * L[2])
            if cos_a > 0:
                i += light_intensities[light] * cos_a / norm_L

            # Specular
            if s != -1:
                R = reflect_ray(L, N)
                r_dot_v = np.dot(R, V)
                norm_R = math.sqrt(R[0] * R[0] + R[1] * R[1] + R[2] * R[2])
                norm_V = math.sqrt(V[0] * V[0] + V[1] * V[1] + V[2] * V[2])
                if r_dot_v > 0:
                    i += light_intensities[light] * pow(r_dot_v / (norm_R * norm_V), s)

    return i


@njit
def reflect_ray(R, N):
    return 2 * N * np.dot(N, R) - R


@njit
def norm(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def benchmark(scene, runs=10, warmup=2):
    import time

    # warmup
    for _ in range(warmup):
        scene.img.fill(255)
        render_scene(scene)

    times = []
    for _ in range(runs):
        scene.img.fill(255)
        start = time.perf_counter()
        render_scene(scene)
        end = time.perf_counter()
        times.append(end - start)

    print("Times:", times)
    print("Avg:", sum(times)/len(times))
    print("Min:", min(times))
    print("Max:", max(times))


def render_scene(scene) -> None:
    """Renders a scene from the given scene object"""
    scene.compile()

    """Unpack all scene arguments, to avoid object access inside loops for performance reasons"""

    bgcol = scene.bgcol
    max_rec_depth = scene.max_rec_depth

    cw = scene.cw
    ch = scene.ch
    img = scene.img

    vw = scene.vw
    vh = scene.vh
    d = scene.d

    """Camera position"""
    O = scene.O

    """Array for faster runtime, will be set using compile"""
    obj_types = scene.types  # Stores the type of each Object
    colors = scene.colors
    speculars = scene.speculars
    reflectives = scene.reflectives

    sphere_centers = scene.sphere_centers
    sphere_radii = scene.sphere_radii

    light_types = scene.light_types
    light_intensities = scene.light_intensities
    light_directions = scene.light_directions
    light_positions = scene.light_positions


    for cx in range(-cw // 2, cw // 2):
        for cy in range(-ch // 2 + 1, ch // 2 + 1):
            V = canvas_to_viewport(cx, cy, scene)
            D = V - O
            color = trace_ray(
                                O,
                                D,
                                bgcol,
                                max_rec_depth,
                                obj_types,
                                colors,
                                speculars,
                                reflectives,
                                sphere_centers,
                                sphere_radii,
                                light_types,
                                light_intensities,
                                light_directions,
                                light_positions,
            )
            put_pixel(cx, cy, scene, col=color)

        #(f"{round((cx + scene.cw // 2) / scene.cw * 100, 2)}%")




"""Initialize scene"""
scene = Scene(
    cw = 500,
    ch = 500,
    vw = 1,
    vh = 1,
    d = 1,
    O = np.array([0, 0, 0])
)

scene.add_objects(
    Sphere(
            center=np.array([0, -1, 3]),
            radius=1,
            color=np.array([255, 0, 0]), # Red
            specular = 500,
            reflective = 0.2
           ),
    Sphere(
            center=np.array([2, 0, 4]),
            radius=1,
            color=np.array([0, 0, 255]), # Blue
            specular = 500,
            reflective = 0.3
           ),
    Sphere(
            center=np.array([-2, 0, 4]),
            radius=1,
            color=np.array([0, 255, 0]), # Green
            specular = 10,
            reflective = 0.4
           ),
    Sphere(
            center=np.array([0, -5001, 0]),
            radius=5000,
            color=np.array([255, 255, 0]), # Yellow
            specular = 1000,
            reflective = 0.5
    )
)

scene.add_lights(
    Light(type="ambient", intensity=0.3), # 0.2
    Light(type="point", position=np.array([2, 1, 0]), intensity=0.6), # 0.6
    Light(type="directional", intensity=0.1, direction=np.array([1, 4, 4])) # 0.2
)
scene.img.fill(255)

benchmark(scene)
#render_scene(scene)

plt.imshow(scene.img)
plt.show()