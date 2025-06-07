import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.stats import special_ortho_group

def plot_free_vec(x, c="b"):
    if len(x) == 2:
        plt.figure()
        plt.plot([0, x[0]], [0, x[1]], c + "-", linewidth=2)
        plt.plot([x[0]], [x[1]], "o", color=c)
        plt.plot([0], [0], "*", color='k')
        plt.xlim(min(0, x[0]) - 1, max(0, x[0]) + 1)
        plt.ylim(min(0, x[1]) - 1, max(0, x[1]) + 1)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("plot_free_vec (2D)")
        plt.grid()
        plt.show()

    elif len(x) == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot([0, x[0]], [0, x[1]], [0, x[2]], c + "-", linewidth=2)
        ax.scatter(x[0], x[1], x[2], color=c, marker="o")
        ax.scatter(0, 0, 0, color='k', marker="*")
        ax.set_xlim([min(0, x[0]) - 1, max(0, x[0]) + 1])
        ax.set_ylim([min(0, x[1]) - 1, max(0, x[1]) + 1])
        ax.set_zlim([min(0, x[2]) - 1, max(0, x[2]) + 1])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("plot_free_vec (3D)")
        plt.show()

    else:
        raise ValueError("Το διάνυσμα πρέπει να είναι είτε 2D είτε 3D.")

# Δοκιμή με 2D διάνυσμα
x2 = np.array([1, 2])
plot_free_vec(x2, "r")

# Δοκιμή με 3D διάνυσμα
x3 = np.array([1, 2, 3])
plot_free_vec(x3, "g")


def plot_vec(a, x, c="b"):
    end = np.array(a) + np.array(x)

    if len(x) == 2:
        plt.figure()
        plt.plot([a[0], end[0]], [a[1], end[1]], c + "-", linewidth=2)
        plt.plot([end[0]], [end[1]], "o", color=c)
        plt.plot([a[0]], [a[1]], "*", color='k')
        plt.xlim(min(a[0], end[0]) - 1, max(a[0], end[0]) + 1)
        plt.ylim(min(a[1], end[1]) - 1, max(a[1], end[1]) + 1)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("plot_vec (2D)")
        plt.grid()
        plt.show()

    elif len(x) == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot([a[0], end[0]], [a[1], end[1]], [a[2], end[2]], c + "-", linewidth=2)
        ax.scatter(end[0], end[1], end[2], color=c, marker="o")
        ax.scatter(a[0], a[1], a[2], color='k', marker="*")
        ax.set_xlim([min(a[0], end[0]) - 1, max(a[0], end[0]) + 1])
        ax.set_ylim([min(a[1], end[1]) - 1, max(a[1], end[1]) + 1])
        ax.set_zlim([min(a[2], end[2]) - 1, max(a[2], end[2]) + 1])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("plot_vec (3D)")
        plt.show()

    else:
        raise ValueError("Το διάνυσμα πρέπει να είναι είτε 2D είτε 3D.")

# Δοκιμές
plot_vec([1, 1], [2, 3], "r")   # 2D
plot_vec([1, 1, 1], [2, 3, 4], "g")  # 3D

def make_unit(x):
    x = np.array(x)
    norm = np.linalg.norm(x)
    if norm == 0:
        raise ValueError("Zero vector cannot be normalized")
    return x / norm

# Paradeigma xrisis gia 2D dianisma
print(make_unit([2, 3]))

# Paradeigma xrisis gia 2D dianisma
print(make_unit([2, 3, 4]))

def project_vec(a, b):
    a = np.array(a)
    b = np.array(b)
    a_unit = make_unit(a)
    return np.dot(b, a_unit) * a_unit

# Paradeigma xrisis gia 2D dianisma
print(project_vec([1, 0], [2, 3]))

# Paradeigma xrisis gia 3D dianisma
print(project_vec([1, 0, 0], [2, 3, 4]))

def cross_demo():
    a = np.random.rand(3)
    b = np.random.rand(3)
    c = np.cross(a, b)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.quiver(0, 0, 0, a[0], a[1], a[2], color='r', label='a')
    ax.quiver(0, 0, 0, b[0], b[1], b[2], color='g', label='b')
    ax.quiver(0, 0, 0, c[0], c[1], c[2], color='b', label='a × b')

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("cross_demo (3D Cross Product)")
    ax.legend()
    plt.show()

# Εκτέλεση
cross_demo()

def plot_Rot(R):
    origin = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    transformed = R @ np.eye(3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.quiver(*origin, transformed[0], transformed[1], transformed[2], color=["r", "g", "b"])

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("plot_Rot – Rotation Matrix Visualization")
    plt.show()

# Δοκιμή με πίνακα στροφής 90° γύρω από τον άξονα z
Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
plot_Rot(Rz)




def generate_rot():
    R = special_ortho_group.rvs(3)  # Τυχαίος πίνακας στροφής
    plot_Rot(R)
    return R

#dokimi
R = generate_rot()
print(R)


#PART B


def rotX(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])

def rotY(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

def rotZ(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

#epalitheusi me xrisi plot_Rot
theta = np.radians(45)
plot_Rot(rotX(theta))
plot_Rot(rotY(theta))
plot_Rot(rotZ(theta))

def plot_hom(G, scale=1.0):
    origin = G[:3, 3]
    x_axis = G[:3, 0] * scale
    y_axis = G[:3, 1] * scale
    z_axis = G[:3, 2] * scale

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(*origin, *x_axis, color='r', label='X')
    ax.quiver(*origin, *y_axis, color='g', label='Y')
    ax.quiver(*origin, *z_axis, color='b', label='Z')

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Plot_hom")
    ax.legend()
    plt.show()

def homogen(R, p):
    G = np.eye(4)
    G[:3, :3] = R
    G[:3, 3] = p
    return G

#paradeigma
R = rotZ(np.pi / 4)
p = np.array([0.5, 0.5, 0])
G = homogen(R, p)
plot_hom(G)
print("omogenis metasximatismos G =\n", G)

def gr(R):
    return homogen(R, np.zeros(3))

def gp(p):
    return homogen(np.eye(3), p)

#dokimi
plot_hom(gr(rotX(np.pi/4)))
plot_hom(gp([1, 0, 0]))


def gRX(theta):
    return gr(rotX(theta))

def gRY(theta):
    return gr(rotY(theta))

def gRZ(theta):
    return gr(rotZ(theta))

#epalitheusi
plot_hom(gRX(np.pi/4))
plot_hom(gRY(np.pi/4))
plot_hom(gRZ(np.pi/4))


def rotAndTranVec(G, vin):
    vin = np.array(vin)
    vout = G[:3, :3] @ vin + G[:3, 3]
    return vout

#dokimi
G = homogen(rotZ(np.pi/2), [1, 0, 0])
vin = np.array([1, 0, 0])
vout = rotAndTranVec(G, vin)
print("rotAndTranVec output:/vout =", vout)


def rotAndTrans_shape(X, Y, Z, G):
    Xout = np.zeros_like(X)
    Yout = np.zeros_like(Y)
    Zout = np.zeros_like(Z)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i, j], Y[i, j], Z[i, j], 1])  #omogenis sintetagmeni (x, y, z, 1)
            transformed = G @ point  #pollaplasiasmos me ton 4x4 pinaka G
            Xout[i, j] = transformed[0]
            Yout[i, j] = transformed[1]
            Zout[i, j] = transformed[2]

    return Xout, Yout, Zout


#dimiourgia kilindrou
theta = np.linspace(0, 2*np.pi, 30)
z = np.linspace(0, 1, 10)
theta, z = np.meshgrid(theta, z)
r = 0.5
X = r * np.cos(theta)
Y = r * np.sin(theta)
Z = z

#metasxim
G = homogen(rotY(np.pi/4), [0, 0, 0.5])
Xout, Yout, Zout = rotAndTrans_shape(X, Y, Z, G)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Xout, Yout, Zout, color='c', alpha=0.7)
ax.set_title("plot_surface: rotAndTrans_shape (cylinder)")
plt.show()


#Part C

def g0e(q1,q2,q3):
    l1,l2= 1.0, 1.0

    T1 = np.array([[1, 0, 0, q1],
                   [0, 1, 0, 0],
                   [0, 0, 1, l1],
                   [0, 0, 0, 1]])
    
    Rz = np.array([[np.cos(q2), -np.sin(q2), 0, 0],
                   [np.sin(q2), np.cos(q2), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    
    Ry = np.array([[np.cos(q3), 0, np.sin(q3), 0],
                   [0, 1, 0, 0],
                   [-np.sin(q3), 0, np.cos(q3), 0],
                   [0, 0, 0, 1]])

    
    G = T1 @ Rz @ Ry
    return G
    

def plot_robot(q1,q2,q3):
    G = g0e(q1, q2, q3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot vasis
    ax.scatter(0, 0, 0, color='k', marker='o', s=50)
    ax.text(0, 0, 0, 'Base')
    
    # Plot prismatikis arthrosis
    ax.plot([0, q1], [0, 0], [1, 1], 'k--')
    ax.scatter(q1, 0, 1, color='r', marker='o', s=50)
    ax.text(q1, 0, 1, 'Joint 1')

    # Plot telikou apotelesmatos
    p = G[:3, 3]
    ax.plot([q1, p[0]], [0, p[1]], [1, p[2]], 'k-')
    ax.scatter(p[0], p[1], p[2], color='b', marker='o', s=50)
    ax.text(p[0], p[1], p[2], 'End Effector')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([0, 2])
    plt.show()



# Paradeigma
plot_robot(1, np.radians(45), np.radians(30))

def workspace():
    q1_range = np.linspace(0.7, 3.0, 10)
    q2_range = np.linspace(-np.radians(170), np.radians(170), 10)
    q3_range = np.linspace(-np.radians(135), np.radians(135), 10)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for q1 in q1_range:
        for q2 in q2_range:
            for q3 in q3_range:
                G = g0e(q1, q2, q3)
                p = G[:3, 3]
                ax.scatter(p[0], p[1], p[2], color='b', s=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([0, 4])
    plt.show()

# Paradeigma 
workspace()

def ikine(p, q1_range=None, q2_range=None, q3_range=None):
    # Default ranges (όπως στο workspace)
    if q1_range is None:
        q1_range = np.linspace(0.7, 3.0, 20)
    if q2_range is None:
        q2_range = np.linspace(-np.radians(170), np.radians(170), 20)
    if q3_range is None:
        q3_range = np.linspace(-np.radians(135), np.radians(135), 20)

    best_q = None
    min_d = np.inf

    for q1 in q1_range:
        for q2 in q2_range:
            for q3 in q3_range:
                G = g0e(q1, q2, q3)
                p_est = G[:3, 3]
                d = np.linalg.norm(p_est - p)
                if d < min_d:
                    min_d = d
                    best_q = (q1, q2, q3)

    return best_q + (min_d,)

#target
target_p = np.array([1.5, 0.5, 1.8])
q1, q2, q3, d = ikine(target_p)
print("Λύση:", q1, q2, q3)
print("Σφάλμα d =", d)

#epalitheusi me g0e
p_check = g0e(q1, q2, q3)[:3, 3]
print("Υπολογισμένο p:", p_check)
print("Απόσταση από στόχο:", np.linalg.norm(p_check - target_p))

def get_trajectory(p0, pf, t, tf, degree=5):
    if degree == 5:
        # Sintelestes gia polionimou 5ou vathmou
        a0 = p0
        a1 = 0
        a2 = 0
        a3 = (10 * (pf - p0)) / (tf ** 3)
        a4 = (-15 * (pf - p0)) / (tf ** 4)
        a5 = (6 * (pf - p0)) / (tf ** 5)
        
        u = a0 + a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5
        u_dot = a1 + 2 * a2 * t + 3 * a3 * t**3 + 4 * a4 * t**4 + 5 * a5 * t**5
    elif degree == 3:
        # Sintelestes gia polionimou 3ou vathmou
        a0 = p0
        a1 = 0
        a2 = (3 * (pf - p0)) / (tf ** 2)
        a3 = (-2 * (pf - p0)) / (tf ** 3)
        
        u = a0 + a1 * t + a2 * t**2 + a3 * t**3
        u_dot = a1 + 2 * a2 * t + 3 * a3 * t**2
    else:
        raise ValueError("Polynomial degree must be 3 or 5")
    
    return u, u_dot

# Paradeigma xrisis kai plot
t_vals = np.linspace(0, 6, 100)
p0, pf = 0, 10
positions_5th = []
velocities_5th = []
positions_3rd = []
velocities_3rd = []

for t in t_vals:
    pos_5, vel_5 = get_trajectory(p0, pf, t, 6, degree=5)
    positions_5th.append(pos_5)
    velocities_5th.append(vel_5)
    
    pos_3, vel_3 = get_trajectory(p0, pf, t, 6, degree=3)
    positions_3rd.append(pos_3)
    velocities_3rd.append(vel_3)

plt.figure()
plt.plot(t_vals, positions_5th, label="5th-degree position")
plt.plot(t_vals, positions_3rd, label="3rd-degree position")
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend()
plt.show()

plt.figure()
plt.plot(t_vals, velocities_5th, label="5th-degree velocity")
plt.plot(t_vals, velocities_3rd, label="3rd-degree velocity")
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.legend()
plt.show()


def trajectory_joint_demo():
    q_start = [np.radians(-90), 0.7, np.radians(-35)]
    q_end = [np.radians(70), 2.0, np.radians(145)]
    tf = 6
    dt = 0.1
    t_vals = np.arange(0, tf + dt, dt)
    
    for t in t_vals:
        q1 = get_trajectory(q_start[0], q_end[0], t, tf, degree=5)[0]
        q2 = get_trajectory(q_start[1], q_end[1], t, tf, degree=5)[0]
        q3 = get_trajectory(q_start[2], q_end[2], t, tf, degree=5)[0]
        plot_robot(np.degrees(q1), q2, np.degrees(q3))
        time.sleep(0.1)

# Paradeigma xrisis
trajectory_joint_demo()

def trajectory_task_demo():
    p_start = [1, 1, 1]
    p_end = [2, 1, 1.5]
    tf = 10
    dt = 0.1
    t_vals = np.arange(0, tf + dt, dt)
    
    for t in t_vals:
        px = get_trajectory(p_start[0], p_end[0], t, tf, degree=5)[0]
        py = get_trajectory(p_start[1], p_end[1], t, tf, degree=5)[0]
        pz = get_trajectory(p_start[2], p_end[2], t, tf, degree=5)[0]
        q1, q2, q3, d = ikine([px, py, pz])
        plot_robot(q1, q2, q3)
        time.sleep(0.1)

# Paradeigma xrisis
trajectory_task_demo()


def dance_demo():
    tf = 10.0  
    dt = 0.1
    ts = np.arange(0, tf + dt, dt)

    
    base = np.array([1.0, 0.0, 1.0])

    freqs = np.linspace(1.0, 3.0, len(ts)) 
    errors = []

    for i, t in enumerate(ts):
        f = freqs[i]
        p = base + np.array([
            0.2 * np.sin(2 * np.pi * f * t),   
            0.2 * np.cos(2 * np.pi * f * t),  
            0.1 * np.sin(4 * np.pi * f * t)    
        ])

        try:
            q1, q2, q3, d = ikine(p)
            errors.append(d)

           
            plot_robot(q1, q2, q3)
            plt.pause(0.01)
            plt.clf()

        except:
            print(f"sfalma ikine gia p = {p}")
            errors.append(np.nan)

    #grafima sfalmatos
    plt.figure()
    plt.plot(ts, errors, label="sfalma d (thesis)")
    plt.xlabel("xronos (s)")
    plt.ylabel("sfalma (m)")
    plt.title("Dance Demo – sfalma metaksi epithimitis kai pragmatikis troxias")
    plt.grid()
    plt.legend()
    plt.show()

dance_demo()
