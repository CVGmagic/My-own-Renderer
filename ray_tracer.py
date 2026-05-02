import numpy as np
import matplotlib.pyplot as plt
import cProfile
import math
from numba import njit, prange
from object_type_flags import *
from light_type_flags import *
from obj_classes import *
from saved_scenes import benchmark_scene




class Light:
    def __init__(self, type: str, intensity, position=None, direction=None):
        self.type = type
        self.intensity = intensity
        self.position = position
        self.direction = direction


@njit
def canvas_to_screen(cx: int, cy: int, cw: int, ch: int) -> tuple[int]:
    """Converts canvas coordinates to screen coordinates"""
    cx += cw // 2
    cy *= -1
    cy += ch // 2
    return (cx, cy)


@njit
def put_pixel(img, cx: int, cy: int, cw: int, ch: int, col: np.ndarray[3]) -> None:
    """Sets the color of a single pixel"""
    cx, cy = canvas_to_screen(cx, cy, cw, ch)
    img[cy, cx] = col
    return


@njit
def canvas_to_viewport(cx: int, cy: int, cw: int, ch: int, vw: float, vh: float, d: float) -> np.ndarray[3]:
    """Converts canvas coordinates to viewport coordinates"""
    if d != 1:
        raise ValueError("This function assumes d = 1")

    vx = cx * vw / cw
    vy = cy * vh / ch
    vz = d
    return np.array([vx, vy, vz], dtype=np.float64)


@njit
def get_normal_vector_sphere(C: np.ndarray[3], P: np.ndarray[3]) -> np.ndarray[3]:
    """Returns a normaalized surface normal"""
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
def random_hemisphere_direction(N: np.ndarray[3]) -> np.ndarray[3]:
    z = np.random.random()  # [0,1)
    phi = 2 * math.pi * np.random.random()

    r = math.sqrt(1 - z * z)
    x = r * math.cos(phi)
    y = r * math.sin(phi)

    # Use frisvad's algorithm to efficiently find orthonorml basis
    if N[2] < -0.9999999:  # might have to tweak this value
        b1x = 0.0
        b1y = -1.0
        b1z = 0.0

        b2x = -1.0
        b2y = 0.0
        b2z = 0.0
    else:
        a = 1.0 / (1.0 + N[2])
        b = -N[0] * N[1] * a

        b1x = 1 - N[0] * N[0] * a
        b1y = b
        b1z = -N[0]

        b2x = b
        b2y = 1 - N[1] * N[1] * a
        b2z = -N[1]

    return np.array([x * b1x + y * b2x, x * b1y + y * b2y, x * b1z + y * b2z], dtype=np.float64) + z * N


@njit
def random_cos_weighted_hemisphere_direction(N: np.ndarray[3]) -> np.ndarray[3]:
    """Returns a normed vetor pointing in a random direction (cosine weighted) in
    a hemisphere"""

    # Generate two random numbers
    u1 = np.random.random()
    u2 = np.random.random()

    # Generate evenly distributed points accross a disk
    r = math.sqrt(u1) # because area grows with r^2
    phi = 2 * math.pi * u2

    # Map disk onto hemisphere (automatically cosine weighted)
    x = r * math.cos(phi)
    y = r * math.sin(phi)
    z = math.sqrt(1 - u1)

    # Use frisvad's algorithm to efficiently find orthonorml basis
    if N[2] < -0.9999999:  # might have to tweak this value
        b1x = 0.0
        b1y = -1.0
        b1z = 0.0

        b2x = -1.0
        b2y = 0.0
        b2z = 0.0
    else:
        a = 1.0 / (1.0 + N[2])
        b = -N[0] * N[1] * a

        b1x = 1 - N[0] * N[0] * a
        b1y = b
        b1z = -N[0]

        b2x = b
        b2y = 1 - N[1] * N[1] * a
        b2z = -N[1]

    return np.array([x * b1x + y * b2x, x * b1y + y * b2y, x * b1z + y * b2z], dtype=np.float64) + z * N


@njit(fastmath=True)
def trace_ray(
        O,
        D,
        colors,
        emitted_colors,
        emission_strengths,
        smoothnesses,
        bounces_left,
        obj_types,
        sphere_centers,
        sphere_radii
) -> np.ndarray[3]:
    """Traces a ray"""
    incoming_light = np.zeros(3, dtype=np.float64)
    ray_color = np.ones(3, dtype=np.float64)

    for i in range(bounces_left):
        # Multiple ray bounces are now implemented iteratively for small performance gain
        obj, t = closest_intersection(O, D, obj_types, sphere_centers, sphere_radii, t_min=0.001, t_max=np.inf)

        if obj == -1: # No object found
            incoming_light += get_environment_lighting(D) * ray_color
            return np.clip(incoming_light, 0, 1)

        P = O + D * t

        if obj_types[obj] == SPHERE:
            N = get_normal_vector_sphere(sphere_centers[obj], P)
        else:
            raise TypeError("Unknown Object type encountered")

        # Combine diffuse and specular reflection depending on smoothness
        specular_chance = smoothnesses[obj]
        if np.random.random() < specular_chance:
            new_D = reflect_ray(D, N)
        else:
            new_D = random_cos_weighted_hemisphere_direction(N)


        emitted_light = emitted_colors[obj] * emission_strengths[obj]
        incoming_light += emitted_light * ray_color # Objects only reflect their color
        ray_color *= colors[obj] # Ray always gets darker

        O = P
        D = new_D

    return incoming_light


@njit
def reflect_ray(D, N) -> np.ndarray[3]:
    """Reflects an incoming ray on a surface described by the surface normal"""
    return D - 2 * dot(D, N) * N


@njit
def norm(v) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


@njit
def dot(v1, v2) -> float:
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


@njit
def cross(v1, v2) -> np.ndarray[3]:
    x = v1[1] * v2[2] - v1[2] * v2[1]
    y = v1[2] * v2[0] - v1[0] * v2[2]
    z = v1[0] * v2[1] - v1[1] * v2[0]
    return np.array([x, y, z])


@njit
def refract_ray(R, N, n_cur, n_new) -> np.ndarray[3]:
    R_hat = R / norm(R)
    eta = n_cur / n_new
    cos_theta_1 = dot(-R_hat, N)
    cos_theta_2 = math.sqrt(1 - eta**2 * (1 - cos_theta_1**2))
    T = eta * R_hat + (eta * cos_theta_1 - cos_thets_2) * N
    return T


@njit
def compute_reflection(R, N, n_cur, n_new) -> float:
    cos_phi = dot(-R, N) / norm(R)
    sqrt_R_0 = (n_cur - n_new) / (n_cur + n_new)
    R_0 = sqrt_R_0 * sqrt_R_0
    return R_0 + (1 - R_0) * (1 - cos_phi)**5


@njit
def lerp_vector(v1: np.ndarray[3], v2: np.ndarray[3], t: float) -> np.ndarray[3]:
    return v1 + (v2 - v1) * t


@njit
def get_environment_lighting(D):
    """Gets an environment color in case the ray misses everything"""
    if True:
        return np.zeros(3, dtype=np.float64)
    ground_color = np.array([0.6, 0.6, 0.6], dtype=np.float64)
    sky_color = np.array([86, 205, 255], dtype=np.float64) / 255

    # TODO Make a nice sky gradient
    if D[1] < -0.3:
        return ground_color
    else:
        return sky_color


def benchmark(scene, runs=10, warmup=2):
    """Benchmarks a scene from the given scene object"""
    import time

    scene.compile()

    """Unpack all scene arguments, to avoid object access inside loops for performance reasons"""

    max_rec_depth = scene.max_rec_depth
    rays_per_pixel = scene.rays_per_pixel

    cw = scene.cw
    ch = scene.ch
    img = scene.img

    vw = scene.vw
    vh = scene.vh
    d = scene.d

    """Camera position"""
    O = scene.O

    """Array for faster runtime, will be set using compile"""
    obj_types = scene.obj_types  # Stores the type of each Object
    colors = scene.colors
    emitted_colors = scene.emitted_colors
    emission_strengths = scene.emission_strengths

    sphere_centers = scene.sphere_centers
    sphere_radii = scene.sphere_radii

    # warmup
    for _ in range(warmup):
        scene.img.fill(1.0)
        fill_image(
            cw=scene.cw,
            ch=scene.ch,
            img=scene.img,
            vw=scene.vw,
            vh=scene.vh,
            d=scene.d,
            O=O,
            colors=colors,
            emitted_colors=emitted_colors,
            emission_strengths=emission_strengths,
            smoothnesses=scene.smoothnesses,
            recursion_limit=max_rec_depth,
            rays_per_pixel=rays_per_pixel,
            obj_types=obj_types,
            sphere_centers=sphere_centers,
            sphere_radii=sphere_radii
        )

    print("Warmup complete, beginning benchmarking")
    times = []
    for _ in range(runs):
        scene.img.fill(1.0)
        start = time.perf_counter()
        fill_image(
            cw=scene.cw,
            ch=scene.ch,
            img=scene.img,
            vw=scene.vw,
            vh=scene.vh,
            d=scene.d,
            O=O,
            colors=colors,
            emitted_colors=emitted_colors,
            emission_strengths=emission_strengths,
            smoothnesses=scene.smoothnesses,
            recursion_limit=max_rec_depth,
            rays_per_pixel=rays_per_pixel,
            obj_types=obj_types,
            sphere_centers=sphere_centers,
            sphere_radii=sphere_radii
        )
        end = time.perf_counter()
        times.append(end - start)

    print("Times:", times)
    print("Avg:", sum(times)/len(times))
    print("Min:", min(times))
    print("Max:", max(times))
    plt.title(f"Resolution: {scene.cw}x{scene.ch} | Rays per pixel: {scene.rays_per_pixel}")
    plt.plot(times)
    plt.show()


@njit(fastmath=True, parallel=True)
def fill_image(
        cw,
        ch,
        img,
        vw,
        vh,
        d,
        O,
        colors,
        emitted_colors,
        emission_strengths,
        smoothnesses,
        recursion_limit,
        rays_per_pixel,
        obj_types,
        sphere_centers,
        sphere_radii
) -> None:
    """Fills the image in"""
    for i in prange(cw * ch):
        y_idx = i // cw # After every row we move up one column
        x_idx = i % cw

        # Convert back to canvas coordinates
        cx = x_idx - (cw // 2)
        cy = (ch // 2) - y_idx

        # Continue tracing like normal
        V = canvas_to_viewport(cx, cy, cw, ch, vw, vh, d)
        D = V - O

        avg_col = np.array([0, 0, 0], dtype=np.float64)

        for _ in range(rays_per_pixel):
            avg_col += trace_ray(
                O=O,
                D=D,
                colors=colors,
                emitted_colors=emitted_colors,
                emission_strengths=emission_strengths,
                smoothnesses=smoothnesses,
                bounces_left=recursion_limit,
                obj_types=obj_types,
                sphere_centers=sphere_centers,
                sphere_radii=sphere_radii
            )
        avg_col /= rays_per_pixel
        put_pixel(img, cx, cy, cw, ch, col=avg_col)


def render_scene(scene) -> None:
    """Renders a scene from the given scene object"""
    scene.compile()

    """Unpack all scene arguments, to avoid object access inside loops for performance reasons"""

    max_rec_depth = scene.max_rec_depth
    rays_per_pixel = scene.rays_per_pixel

    cw = scene.cw
    ch = scene.ch
    img = scene.img

    vw = scene.vw
    vh = scene.vh
    d = scene.d

    """Camera position"""
    O = scene.O

    """Array for faster runtime, will be set using compile"""
    obj_types = scene.obj_types  # Stores the type of each Object
    colors = scene.colors
    emitted_colors = scene.emitted_colors
    emission_strengths = scene.emission_strengths

    sphere_centers = scene.sphere_centers
    sphere_radii = scene.sphere_radii

    fill_image(
        cw=scene.cw,
        ch=scene.ch,
        img=scene.img,
        vw = scene.vw,
        vh = scene.vh,
        d = scene.d,
        O=O,
        colors=colors,
        emitted_colors=emitted_colors,
        emission_strengths=emission_strengths,
        smoothnesses=scene.smoothnesses,
        recursion_limit=max_rec_depth,
        rays_per_pixel=rays_per_pixel,
        obj_types=obj_types,
        sphere_centers=sphere_centers,
        sphere_radii=sphere_radii
    )
    plt.title(f"Resolution: {scene.cw}x{scene.ch} | Rays per pixel: {scene.rays_per_pixel}")
    plt.imshow(scene.img)
    plt.show()


def render_scene_over_time(scene) -> None:
    """Renders a scene from the given scene object"""
    scene.compile()

    """Unpack all scene arguments, to avoid object access inside loops for performance reasons"""

    max_rec_depth = scene.max_rec_depth
    rays_per_pixel = scene.rays_per_pixel

    cw = scene.cw
    ch = scene.ch
    img = scene.img

    vw = scene.vw
    vh = scene.vh
    d = scene.d

    """Camera position"""
    O = scene.O

    """Array for faster runtime, will be set using compile"""
    obj_types = scene.obj_types  # Stores the type of each Object
    colors = scene.colors
    emitted_colors = scene.emitted_colors
    emission_strengths = scene.emission_strengths

    sphere_centers = scene.sphere_centers
    sphere_radii = scene.sphere_radii

    cur_img = np.zeros_like(scene.img)
    fig, ax = plt.subplots()
    img_display = ax.imshow(scene.img)
    ax.set_title("Starting render...")
    plt.show(block=False)

    lim = scene.rays_per_pixel
    i = 0
    update_frequency = 10
    while i < lim:
        fill_image(
            cw=scene.cw,
            ch=scene.ch,
            img=scene.img,
            vw=scene.vw,
            vh=scene.vh,
            d=scene.d,
            O=O,
            colors=colors,
            emitted_colors=emitted_colors,
            emission_strengths=emission_strengths,
            smoothnesses=scene.smoothnesses,
            recursion_limit=max_rec_depth,
            rays_per_pixel=1,
            obj_types=obj_types,
            sphere_centers=sphere_centers,
            sphere_radii=sphere_radii
        )

        weight = 1 / (i + 1)
        cur_img = cur_img * (1 - weight) + scene.img * weight
        if i % update_frequency == 0:
            img_display.set_data(np.power(cur_img, 0.45)) # 0.45
            ax.set_title(f"Resolution: {scene.cw}x{scene.ch} | Rays per pixel: {i}")
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(1e-32)

        i += 1


"""Initialize scene"""
scene = Scene(
    cw = 450,
    ch = 300,
    vw = 1.5,
    vh = 1,
    d = 1,
    O = np.array([0, 0, 0], dtype=np.float64),
    max_rec_depth=4,
    rays_per_pixel=10
)

scene.add_objects(
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


#scene.img.fill(255)
benchmark(benchmark_scene)

#render_scene_over_time(scene)

