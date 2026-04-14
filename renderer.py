import numpy as np
import matplotlib.pyplot as plt


"""Controls image resolution"""
cw: int = 100
ch: int = 200
img = np.zeros((h, w, 3), dtype=int)


"""Controls FOV"""
vw = 1
vh = 1
d = 1


"""Camera position"""
O = np.array([0, 0, 0])



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
    return np.ndarray()


class Ray:
    def __init__(self, V):
        self.D = V - O


class Sphere:
    def __init__(self, C: np.ndarray, r: float):
        self.C = C
        self.r = r
        self.CO = O - self.C


    def find_intersections(self, ray: Ray):
        a = np.dot(ray.D, ray.D)
        b = 2 * np.dot(CO, D)
        c = np.dot(Co, CO) - r**2


plt.imshow(img)
plt.show()