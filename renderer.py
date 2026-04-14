import numpy as np
import matplotlib.pyplot as plt


class Scene:
    def __init__(self, cw, ch, vw, vh, d, O):
        self.objects = []
        self.lights = []

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


    def add_objects(self, lst):
        for elem in lst:
            self.objects.append(elem)


    def add_lights(self, lst):
        for elem in lst:
            self.lights.append(elem)


class Ray:
    def __init__(self, V):
        self.D = V - O
        self.length = np.linalg.norm(self.D)


class Sphere:
    def __init__(self, center: np.ndarray, radius: float, color=np.array([0, 0, 0])):
        self.C = center
        self.r = radius
        self.color = color
        self.CO = O - self.C
        self.c = np.dot(self.CO, self.CO) - self.r**2


    def find_intersections(self, ray: Ray) -> np.ndarray:
        """Finds the possible scalars for the ray. Invalid solutions return np.nan"""
        a = np.dot(ray.D, ray.D)
        b = 2 * np.dot(self.CO, ray.D)
        c = self.c

        l1 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        l2 = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)

        return np.array([l1, l2])


class Light:
    def __init__(self, type: string, intensity, position=None, direction=None):
        self.type = type
        self.intensity = intensity
        self.position = position
        self.direction = direction


def canvas_to_screen(cx: int, cy: int) -> tuple[int]:
    """Converts canvas coordinates to screen coordinates"""
    cx += cw // 2
    cy *= -1
    cy += ch // 2
    return (cx, cy)


def put_pixel(cx: int, cy: int, col: np.ndarray) -> None:
    """Sets the color of a single pixel"""
    cx, cy = canvas_to_screen(cx, cy)
    img[cy, cx] = col
    return


def canvas_to_viewport(cx: int, cy: int) -> np.ndarray:
    """Converts canvas coordinates to viewport coordinates"""
    if (d != 1):
        raise ValueError("This function assumes d = 1")

    vx = cx * vw / cw
    vy = cy * vh / ch
    vz = d
    return np.array([vx, vy, vz])


def trace_ray(ray: Ray):
    """Traces the rays path and returns the closest intersected Object"""
    min_dist: float = np.inf
    closest = None

    for obj in scene.objects:
        intersections = obj.find_intersections(ray)

        for t in intersections:
            if np.isnan(t) or t < 0:
                continue

            dist: float = t * ray.length
            if dist < min_dist:
                min_dist = dist
                closest = obj

    return closest



"""List of Objects"""
scene = Scene(
    cw = 500,
    ch = 500,
    vw = 1,
    vh = 1,
    d = 1,
    O = np.array([0, 0, 0])
)
objects = [
    Sphere(np.array([1, 2, 10]), 1, color=np.array([0, 0, 255])),
    Sphere(np.array([0, 2, 5]), 0.5, color=np.array([255, 0, 0]))
]
scene.add_objects(objects)



for cx in range(-cw // 2, cw // 2):
    for cy in range(-ch // 2 + 1, ch // 2 + 1):
        V = canvas_to_viewport(cx, cy)
        ray = Ray(V)
        obj = trace_ray(ray)
        if obj:
            put_pixel(cx, cy, col=obj.color)

plt.imshow(img)
plt.show()