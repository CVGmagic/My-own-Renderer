import numpy as np
import matplotlib.pyplot as plt
import cProfile
import math


class Scene:
    def __init__(self, cw, ch, vw, vh, d, O, bgcol=np.array([0, 0, 0]), max_rec_depth=3):
        self.objects = []
        self.lights = []
        self.bgcol = bgcol
        self.max_rec_depth = max_rec_depth

        """Controls image resolution"""
        self.cw: int = cw
        self.ch: int = ch
        self.img = np.full((ch, cw, 3), 255, dtype=int)

        """Controls FOV"""
        self.vw = vw
        self.vh = vh
        self.d = d

        """Camera position"""
        self.O = O


    def add_objects(self, *args):
        for obj in args:
            self.objects.append(obj)


    def add_lights(self, *args):
        for light in args:
            self.lights.append(light)


class Ray:
    def __init__(self, O, D):
        self.O = O
        self.D = D
        self.length = np.linalg.norm(self.D)


class Sphere:
    def __init__(self, center: np.ndarray, radius: float, color=np.array([0, 0, 0]), specular=-1, reflective=0):
        self.C = center
        self.r = radius
        self.color = color
        self.specular = specular
        self.reflective = reflective


    def find_intersections(self, ray: Ray) -> np.ndarray:
        """Finds the possible scalars for the ray. Invalid solutions return np.nan"""
        CO = ray.O - self.C
        a = np.dot(ray.D, ray.D)
        b = 2 * np.dot(CO, ray.D)
        c = np.dot(CO, CO) - self.r ** 2

        l1 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        l2 = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)

        return np.array([l1, l2])


    def get_normal_vector(self, P: np.ndarray) -> np.ndarray:
        N = P - self.C
        N /= math.sqrt(N[0] * N[0] + N[1] * N[1] + N[2] * N[2])
        return N


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


def put_pixel(cx: int, cy: int, scene: Scene, col: np.ndarray) -> None:
    """Sets the color of a single pixel"""
    cx, cy = canvas_to_screen(cx, cy, scene)
    scene.img[cy, cx] = col
    return


def canvas_to_viewport(cx: int, cy: int, scene) -> np.ndarray:
    """Converts canvas coordinates to viewport coordinates"""
    if (scene.d != 1):
        raise ValueError("This function assumes d = 1")

    vx = cx * scene.vw / scene.cw
    vy = cy * scene.vh / scene.ch
    vz = scene.d
    return np.array([vx, vy, vz])


def closest_intersection(ray, scene, t_min, t_max) -> tuple:
    """
    Finds the closest intersection between a ray and any object
    :return: The intersected object and the t used to intersect
    """
    closest_obj = None
    closest_t = np.inf

    for obj in scene.objects:
        intersections = obj.find_intersections(ray)

        for t in intersections:
            if np.isnan(t) or t < t_min or t > t_max:
                continue

            if t < closest_t:
                closest_obj = obj
                closest_t = t

    return closest_obj, closest_t


def exists_intersection(ray, scene, t_min, t_max) -> tuple:
    """
        Finds any intersection between a ray and any object between t_min and t_max
        :return: Bool, telling us if anything was intersected at all
        """
    for obj in scene.objects:
        intersections = obj.find_intersections(ray)

        for t in intersections:
            if np.isnan(t) or t < t_min or t > t_max:
                continue

            return True

    return False


def trace_ray(ray: Ray, scene, t_min=0, t_max=np.inf, recursion_depth=0) -> np.ndarray:
    """Traces the rays path and returns the closest intersected Object"""
    closest_obj, closest_t = closest_intersection(ray, scene, t_min, t_max)

    if not closest_obj:
        return scene.bgcol

    P = ray.O + ray.D * closest_t
    N = closest_obj.get_normal_vector(P)
    V = -ray.D
    local_color = closest_obj.color * compute_lighting(P, N, V, closest_obj.specular, scene)

    # If object is not reflecctive, or we hit recursion limit, stop
    r = closest_obj.reflective
    if r <= 0 or recursion_depth >= scene.max_rec_depth:
        return local_color

    R = reflect_ray(V, N)
    reflected_color = trace_ray(Ray(P, R), scene, t_min=0.001, t_max=np.inf, recursion_depth=recursion_depth + 1)

    return local_color * (1 - r) + reflected_color * r


def render_scene(scene) -> None:
    for cx in range(-scene.cw // 2, scene.cw // 2):
        for cy in range(-scene.ch // 2 + 1, scene.ch // 2 + 1):
            V = canvas_to_viewport(cx, cy, scene)
            D = V - scene.O
            ray = Ray(scene.O, D)
            color = trace_ray(ray, scene)
            put_pixel(cx, cy, scene, col=color)

        #(f"{round((cx + scene.cw // 2) / scene.cw * 100, 2)}%")


def compute_lighting(P, N, V, s, scene) -> float:
    """
    Takes in a point, the surface normal at that point and of course the scene
    and computes the intensity of the reflected light
    """
    i: float = 0
    for light in scene.lights:
        if light.type == "ambient":
            i += light.intensity

        else:
            if light.type == "point":
                L = light.position - P
                t_max = 1.001

            elif light.type == "directional":
                L = light.direction
                t_max = np.inf

            # Shadow check
            shadow_sphere, shadow_t = closest_intersection(Ray(P, L), scene, t_min=0.001, t_max=t_max)
            if shadow_sphere:
                continue

            # Diffuse
            cos_a = np.dot(N, L)
            norm_L = math.sqrt(L[0] * L[0] + L[1] * L[1] + L[2] * L[2])
            if cos_a > 0:
                i += light.intensity * cos_a / norm_L

            # Specular
            if s != -1:
                R = reflect_ray(L, N)
                r_dot_v = np.dot(R, V)
                norm_R = math.sqrt(R[0] * R[0] + R[1] * R[1] + R[2] * R[2])
                norm_V = math.sqrt(V[0] * V[0] + V[1] * V[1] + V[2] * V[2])
                if r_dot_v > 0:
                    i += light.intensity * pow(r_dot_v / (norm_R * norm_V), s)

    return i


def reflect_ray(R, N):
    return 2 * N * np.dot(N, R) - R


def norm(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def benchmark(scene, runs=5, warmup=2):
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


"""Initialize scene"""
scene = Scene(
    cw = 300,
    ch = 300,
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
    Light(type="point", intensity=0.6, position=np.array([2,1,0])), # 0.6
    Light(type="directional", intensity=0.1, direction=np.array([1, 4, 4])) # 0.2
)
scene.img.fill(255)


benchmark(scene)

plt.imshow(scene.img)
plt.show()