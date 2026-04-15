import numpy as np
import matplotlib.pyplot as plt


class Scene:
    def __init__(self, cw, ch, vw, vh, d, O, bgcol=np.array([255, 255, 255])):
        self.objects = []
        self.lights = []
        self.bgcol = bgcol

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
        N /= np.linalg.norm(N)
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


def trace_ray(ray: Ray, scene, t_min=0, t_max=np.inf) -> np.ndarray:
    """Traces the rays path and returns the closest intersected Object"""
    closest_obj, closest_t = closest_intersection(ray, scene, t_min, t_max)

    if closest_obj:
        P = ray.O + ray.D * closest_t
        N = closest_obj.get_normal_vector(P)
        V = -ray.D
        return closest_obj.color * compute_lighting(P, N, V, closest_obj.specular, scene)
    else:
        return scene.bgcol


def render_scene(scene) -> None:
    for cx in range(-scene.cw // 2, scene.cw // 2):
        for cy in range(-scene.ch // 2 + 1, scene.ch // 2 + 1):
            V = canvas_to_viewport(cx, cy, scene)
            D = V - scene.O
            ray = Ray(scene.O, D)
            color = trace_ray(ray, scene)
            put_pixel(cx, cy, scene, col=color)


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
            if cos_a > 0:
                i += light.intensity * cos_a / (np.linalg.norm(N) * np.linalg.norm(L))

            # Specular
            if s != -1:
                R = 2 * N * cos_a - L
                r_dot_v = np.dot(R, V)
                if r_dot_v > 0:
                    i += light.intensity * np.pow(r_dot_v / (np.linalg.norm(R) * np.linalg.norm(V)), s)

    return i


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
            specular = 500
           ),
    Sphere(
            center=np.array([2, 0, 4]),
            radius=0.5,
            color=np.array([0, 0, 255]), # Blue
            specular = 500
           ),
    Sphere(
            center=np.array([-2, 0, 4]),
            radius=1,
            color=np.array([0, 255, 0]), # Green
            specular = 10
           ),
    Sphere(
            center=np.array([0, -5001, 0]),
            radius=5000,
            color=np.array([255, 255, 0]), # Yellow
            specular = 1000
    ),
    Sphere(
        center = np.array([0, 0.5, 2]),
        radius = 0.3,
        color = np.array([255, 105, 180]), # Pink
        specular = 10000
    )
)

scene.add_lights(
    Light(type="ambient", intensity=0.3), # 0.2
    Light(type="point", intensity=0.6, position=np.array([2,1,0])), # 0.6
    Light(type="directional", intensity=0.1, direction=np.array([1, 4, 4])) # 0.2
)

render_scene(scene)

plt.imshow(scene.img)
plt.show()