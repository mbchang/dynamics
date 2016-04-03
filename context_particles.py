import numpy as np
from utils import *

# Hardcoded Global Variables
G_w_width, G_w_height = 384.0, 288.0 #640.0,480.0
G_max_velocity, G_min_velocity = 2*3000.0, -2*3000.0 #2*4500.0, -2*4500.0  # make this double the max initial velocity, there may be some velocities that go over, but those are anomalies (this is normalizing factor)
G_mass_values = [0.33, 1.0, 3.0, 1e30]  # hardcoded  TODO: add 1e30
G_goo_strength_values = [0.0, -5.0, -20.0, -100.0]  # hardcoded the -100 is a dummy
G_goo_strength2color = {0.0: "darkmagenta", -5.0: "brown", -20.0: "yellowgreen", -100.0: "pink"}
G_particle_mass2color = {0.33: "yellow", 1.0: "red", 3.0: "blue", 1e30: "green"}  # TODO: add 1e30

class Context_Goo():
    def __init__(self, goo_vector):
        """
            goo_vector: [cx, cy, width, height, [onehot strength], id]

             Inverse Transformation: normalize --> xywh2ltrb (don't need to crop)

             Reference Forward Transformation
             goos[:,:4] = crop_to_window(goos[:,:4])  # crop so that dimensions are inside window
             goos[:,:4] = ltrb2xywh(goos[:,:4])  # convert [left, top, right, bottom] to [cx, cy, w, h]
             goos[:,:4] = goos[:,:4]/G_w_width  # normalize coordinates
        """
        # unpack
        cx_cy_w_h, one_hot = self.unpack_goo_vector(goo_vector)

        # normalize
        cx_cy_w_h = cx_cy_w_h*G_w_width

        # xywh2ltrb
        [self.left, self.top, self.right, self.bottom] = self.xywh2ltrb_one_box(cx_cy_w_h)

        # onehot
        self.strength = one_hot_to_num(one_hot, G_goo_strength_values) # convert to strength
        self.color = G_goo_strength2color[self.strength]  # convert to color

    def __str__(self):
        return 'Goo: left: %s, top: %s, right: %s, bottom: %s, color: %s' \
                %(self.left, self.top, self.right, self.bottom, self.color)

    def unpack_goo_vector(self, goo_vector):
        """
            goo_vector: [cx, cy, width, height, [onehot strength], id]
        """
        assert goo_vector.shape == (9,)
        cx_cy_w_h = goo_vector[:4]
        one_hot_strength = goo_vector[4:8]
        object_id = goo_vector[8]
        assert object_id == 0

        return cx_cy_w_h, one_hot_strength

    def xywh2ltrb_one_box(self, box):
        """
            box: [cx, cy, width, height]
            out: [left, top, right, bottom]

            decoupled from normalization
        """
        cx, cy, width, height = box[:]
        left = cx - width/2
        right = cx + width/2
        assert right > left
        top = cy - height/2
        bottom = cy + height/2
        assert bottom > top
        return [left, top, right, bottom]

    def format(self):
        return [[self.left, self.top], [self.right, self.bottom],
                self.strength, self.color]

class Context_Particle():
    def __init__(self, particle_path):
        """
            particle_path: np array (winsize/2, 8)
        """
        self.particle_path = particle_path

        # Get path
        self.path = np.copy(particle_path[:, :4])  # (winsize, [px, py, vx, vy])
        self.path[:,:2] = self.path[:,:2]*G_w_width  # unnormalize position
        self.path[:,2:] = self.path[:,2:]*G_max_velocity  # unnormalize velocity

    def __str__(self):
        return "Particle: color: %s" %(self.color)

    def to_dict(self, accel):
        start,end = (6,10) if accel else (4,8)
        assert(end-start == len(G_mass_values))

        # Get mass
        one_hot = self.particle_path[0, start:end]  # should be the same for all timesteps, so we just use the first one
        self.mass = one_hot_to_num(one_hot, G_mass_values)
        assert np.allclose(self.particle_path[:, end], 1)  # object id
        self.color = G_particle_mass2color[self.mass]
        return {'color': self.color, 'mass': self.mass}

    def reshape_path(self, path):
        """
            From: (winsize, [px, py, vx, vy])
            To: (winsize, [pos vel], [x y])
        """
        winsize = path.shape[0]
        assert path.shape[1] == 4
        return np.reshape(path, (winsize, 2, 2))
