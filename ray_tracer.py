import numpy as np
import matplotlib.pyplot as plt
import math
from numba import njit, prange
from object_type_flags import *
from obj_classes import *
from saved_scenes import benchmark_scene


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


@njit(fastmath=True)
def canvas_to_viewport(cx: int, cy: int, cw: int, ch: int, vw: float, vh: float, d: float) -> np.ndarray[3]:
    """Converts canvas coordinates to viewport coordinates"""
    if d != 1:
        raise ValueError("This function assumes d = 1")

    vx = cx * vw / cw
    vy = cy * vh / ch
    vz = d
    return np.array([vx, vy, vz], dtype=np.float64)


@njit(fastmath=True)
def get_normal_vector_sphere(C: np.ndarray[3], P: np.ndarray[3]) -> np.ndarray[3]:
    """Returns a normaalized surface normal"""
    N = P - C
    N /= math.sqrt(N[0] * N[0] + N[1] * N[1] + N[2] * N[2])
    return N


@njit
def find_intersections_triangle_slow(A, B, C, O, D):
    # TODO Optimize the hell out of this function
    """Finds the intersections with a triangle"""

    AB = B - A
    AC = C - A

    # Coefficient matrix
    M = np.array([[D[0], -AB[0], -AC[0]],
                  [D[1], -AB[1], -AC[1]],
                  [D[2], -AB[2], -AC[2]]])

    det = np.linalg.det(M)

    if abs(det) < 1e-8:
        return np.nan

    # Solution vector
    b = np.array([A[0] - O[0],
                  A[1] - O[1],
                  A[2] - O[2]])

    solution = np.linalg.solve(M, b)

    t, a, b = solution[0], solution[1], solution[2]

    if a >= 0 and b >= 0 and a + b <= 1 :
        return t
    else:
        return np.nan


@njit(fastmath=True)
def find_intersections_triangle(A, B, C, O, D):
    AB = B - A
    AC = C - A

    # pvec = cross(D, AC)
    pvec_x = D[1]*AC[2] - D[2]*AC[1]
    pvec_y = D[2]*AC[0] - D[0]*AC[2]
    pvec_z = D[0]*AC[1] - D[1]*AC[0]

    det = AB[0]*pvec_x + AB[1]*pvec_y + AB[2]*pvec_z

    if abs(det) < 1e-8:
        return np.nan

    inv_det = 1.0 / det

    tvec0 = O[0] - A[0]
    tvec1 = O[1] - A[1]
    tvec2 = O[2] - A[2]

    u = (tvec0*pvec_x + tvec1*pvec_y + tvec2*pvec_z) * inv_det
    if u < 0.0 or u > 1.0:
        return np.nan

    # qvec = cross(tvec, AB)
    qvec_x = tvec1*AB[2] - tvec2*AB[1]
    qvec_y = tvec2*AB[0] - tvec0*AB[2]
    qvec_z = tvec0*AB[1] - tvec1*AB[0]

    v = (D[0]*qvec_x + D[1]*qvec_y + D[2]*qvec_z) * inv_det
    if v < 0.0 or u + v > 1.0:
        return np.nan

    t = (AC[0]*qvec_x + AC[1]*qvec_y + AC[2]*qvec_z) * inv_det

    if t <= 0.0:
        return np.nan

    return t


@njit
def find_intersections_sphere(C: np.ndarray[3], r: float, O, D) -> np.ndarray:
    """Finds the possible scalars for the ray. Invalid solutions return np.nan"""
    CO = O - C
    a = np.dot(D, D)
    b = 2 * np.dot(CO, D)
    c = np.dot(CO, CO) - r ** 2

    disc = b**2 - 4 * a * c

    if disc < 1e-8:
        return np.nan

    sqrt_disc = np.sqrt(disc)

    l2 = (-b - sqrt_disc) / (2 * a)

    if l2 > 0: # If l2 is > 0, it is always the better solution
        return l2
    else:
        l1 = (-b + sqrt_disc) / (2 * a)
        return l1


@njit
def closest_intersection(O: np.ndarray[3], D: np.ndarray[3], obj_types, sphere_centers, sphere_radii, triangles, triangle_normals, t_min, t_max) -> tuple:
    """
    Finds the closest intersection between a ray and any object
    :return: The intersected object and the t used to intersect
    """
    closest_obj = -1
    closest_t = np.inf

    for obj in range(len(obj_types)):
        if obj_types[obj] == SPHERE:
            t = find_intersections_sphere(sphere_centers[obj], sphere_radii[obj], O, D)
        elif obj_types[obj] == TRIANGLE:
            ABC = triangles[obj]
            n = triangle_normals[obj]
            t = find_intersections_triangle(ABC[0], ABC[1], ABC[2], O, D)


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


@njit(fastmath=True)
def random_cos_weighted_hemisphere_direction(N: np.ndarray[3], rand_state) -> np.ndarray[3]:
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
        is_glass,
        ref_idxs,
        absorptions,
        bounces_left,
        obj_types,
        sphere_centers,
        sphere_radii,
        triangles,
        triangle_normals,
        rand_state
) -> np.ndarray[3]:
    """Traces a ray"""
    incoming_light = np.zeros(3, dtype=np.float64)
    ray_color = np.ones(3, dtype=np.float64)

    for i in range(bounces_left):
        # Multiple ray bounces are now implemented iteratively for small performance gain
        obj, t = closest_intersection(O, D, obj_types, sphere_centers, sphere_radii, triangles, triangle_normals, t_min=0.001, t_max=np.inf)

        if obj == -1: # No object found
            incoming_light += get_environment_lighting(D) * ray_color
            break

        P = O + D * t

        if obj_types[obj] == SPHERE:
            N = get_normal_vector_sphere(sphere_centers[obj], P)
        elif obj_types[obj] == TRIANGLE:
            N = triangle_normals[obj]
        else:
            raise TypeError("Unknown Object type encountered")

        if is_glass[obj]:
            if dot(N, D) < 0: # Front face
                n_cur = 1.0
                n_new = ref_idxs[obj]
            else: # Back face
                n_new = 1.0
                n_cur = ref_idxs[obj]
                N = -N # Flip normal vector, because that's what functions expect
                transmittance = np.exp(-colors[obj] * absorptions[obj] * norm(D) * t)
                ray_color *= transmittance


            reflection_chance = compute_reflection_fresnel(D, N, n_cur, n_new)
            if np.random.random() < reflection_chance: # Reflect ray
                new_D = reflect_ray(D, N)
            else: # Refract ray
                new_D = refract_ray(D, N, n_cur, n_new)
                new_D /= norm(new_D)


        else:
            # Combine diffuse and specular reflection depending on smoothness
            specular_chance = smoothnesses[obj]
            if np.random.random() < specular_chance:
                new_D = reflect_ray(D, N)
            else:
                new_D = random_cos_weighted_hemisphere_direction(N, rand_state)


            emitted_light = emitted_colors[obj] * emission_strengths[obj]
            incoming_light += emitted_light * ray_color # Objects only reflect their color
            ray_color *= colors[obj] # Ray always gets darker


        O = P
        D = new_D

    return incoming_light


@njit(fastmath=True)
def reflect_ray(D, N) -> np.ndarray[3]:
    """Reflects an incoming ray on a surface described by the surface normal"""
    return D - 2 * dot(D, N) * N


@njit(fastmath=True)
def norm(v) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


@njit(fastmath=True)
def dot(v1, v2) -> float:
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


@njit(fastmath=True)
def cross(v1, v2) -> np.ndarray[3]:
    x = v1[1] * v2[2] - v1[2] * v2[1]
    y = v1[2] * v2[0] - v1[0] * v2[2]
    z = v1[0] * v2[1] - v1[1] * v2[0]
    return np.array([x, y, z])


@njit(fastmath=True)
def refract_ray(R, N, n_cur, n_new) -> np.ndarray[3]:
    R_hat = R / norm(R)
    eta = n_cur / n_new
    cos_theta_1 = dot(-R_hat, N)
    cos_theta_2 = math.sqrt(1 - eta**2 * (1 - cos_theta_1**2))
    T = eta * R_hat + (eta * cos_theta_1 - cos_theta_2) * N
    return T


@njit(fastmath=True)
def compute_reflection_fresnel(R, N, n_cur, n_new) -> float:
    cos_phi = dot(-R, N) / norm(R)
    sqrt_R_0 = (n_cur - n_new) / (n_cur + n_new)
    R_0 = sqrt_R_0 * sqrt_R_0
    return R_0 + (1 - R_0) * (1 - cos_phi)**5


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


@njit(fastmath=True)
def rand(state) -> float:
    """Generates a random number between 0 and 1 and changes the state"""
    # TODO Random function has visible patterns
    state[0] = (1664525 * state[0] + 1013904223) & 0xFFFFFFFF
    return state[0] / 4294967296.0


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
            is_glass=scene.is_glass,
            ref_idxs=scene.ref_idxs,
            absorptions=scene.absorptions,
            recursion_limit=max_rec_depth,
            rays_per_pixel=rays_per_pixel,
            obj_types=obj_types,
            sphere_centers=sphere_centers,
            sphere_radii=sphere_radii,
            triangles=scene.triangles,
            triangle_normals=scene.triangle_normals
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
            is_glass=scene.is_glass,
            ref_idxs=scene.ref_idxs,
            absorptions=scene.absorptions,
            recursion_limit=max_rec_depth,
            rays_per_pixel=rays_per_pixel,
            obj_types=obj_types,
            sphere_centers=sphere_centers,
            sphere_radii=sphere_radii,
            triangles = scene.triangles,
            triangle_normals = scene.triangle_normals
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
        is_glass,
        ref_idxs,
        absorptions,
        recursion_limit,
        rays_per_pixel,
        obj_types,
        sphere_centers,
        sphere_radii,
        triangles,
        triangle_normals
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

        # Generates a seed for the randomizer for each pixel
        rand_state = np.array([(i * 9781 + 1) & 0xFFFFFFFF], dtype=np.uint32)

        avg_col = np.array([0, 0, 0], dtype=np.float64)

        for _ in range(rays_per_pixel):
            avg_col += trace_ray(
                O=O,
                D=D,
                colors=colors,
                emitted_colors=emitted_colors,
                emission_strengths=emission_strengths,
                smoothnesses=smoothnesses,
                is_glass=is_glass,
                ref_idxs=ref_idxs,
                absorptions=absorptions,
                bounces_left=recursion_limit,
                obj_types=obj_types,
                sphere_centers=sphere_centers,
                sphere_radii=sphere_radii,
                triangles=triangles,
                triangle_normals=triangle_normals,
                rand_state=rand_state,
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
    is_glass = scene.is_glass

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
        is_glass=is_glass,
        ref_idxs=scene.ref_idxs,
        absorptions=scene.absorptions,
        recursion_limit=max_rec_depth,
        rays_per_pixel=rays_per_pixel,
        obj_types=obj_types,
        sphere_centers=sphere_centers,
        sphere_radii=sphere_radii,
        triangles=scene.triangles,
        triangle_normals=scene.triangle_normals
    )
    # Removes axis ticks
    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False
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
    is_glass = scene.is_glass

    sphere_centers = scene.sphere_centers
    sphere_radii = scene.sphere_radii

    cur_img = np.zeros_like(scene.img)
    fig, ax = plt.subplots()
    img_display = ax.imshow(scene.img)
    # Removes axis ticks
    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False
                    )
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
            is_glass=is_glass,
            ref_idxs=scene.ref_idxs,
            absorptions=scene.absorptions,
            recursion_limit=max_rec_depth,
            rays_per_pixel=1,
            obj_types=obj_types,
            sphere_centers=sphere_centers,
            sphere_radii=sphere_radii,
            triangles=scene.triangles,
            triangle_normals=scene.triangle_normals
        )
        weight = 1 / (i + 1)
        cur_img = cur_img * (1 - weight) + scene.img * weight
        i += 1
        if i % update_frequency == 0:
            # Set NaNs to pink
            mask = np.isnan(cur_img).any(axis=2)
            img[mask] = np.array([1.0, 0.0, 1.0])
            img_display.set_data(cur_img)
            #img_display.set_data(np.power(cur_img, 0.45)) # 0.45
            ax.set_title(f"Resolution: {scene.cw}x{scene.ch} | Rays per pixel: {i}")
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(1e-32)





