# this is original version for 3D LIC, without acceleration
import numpy as np
from scipy.interpolate import interpn
import time


def distance(Coord1, Coord2):
    return np.sqrt(np.sum((np.array(Coord1) - np.array(Coord2)) ** 2))


def normalize(ux, uy, uz):
    mag = np.sqrt(ux**2 + uy**2 + uz**2)
    return ux / mag, uy / mag, uz / mag


# def arc_parameterize(curve):
#     '''
#     This function is to parameterize a curve by arc length.

#     Parameters:
#     ----------
#     curve: a set of coordinates, describing a space curve

#     Returns:
#     --------
#     curve_arc: a nes set of coordinates, which the intervel of each adjcent points is unit arc.
#     '''


def LIC_singleLine(input_texture, streamline):
    def arc_interval(streamline):
        return np.sum((streamline[1:] - streamline[0:-1]) ** 2, axis=1)

    def T(input_texture, streamline):
        xyz = (
            np.linspace(0, input_texture.shape[0] - 1, input_texture.shape[0]),
            np.linspace(0, input_texture.shape[1] - 1, input_texture.shape[1]),
            np.linspace(0, input_texture.shape[2] - 1, input_texture.shape[2]),
        )
        # point data to interval data, using the cerntre-average
        return (interpn(xyz, input_texture, streamline)[1:] + interpn(xyz, input_texture, streamline)[0:-1]) / 2

    Intensity = np.sum(np.hamming(streamline.shape[0] - 1) * T(input_texture, streamline) * arc_interval(streamline))
    return Intensity


class vectorfield(object):
    """
    vectorfield is a class to store a 3d vector field.
    It contains a normalized vector field for directions and a scalar field for strength.
    """

    # the vector field must be normalized
    def __init__(self, size, field_x, field_y, field_z):
        assert size == field_x.shape == field_y.shape == field_z.shape
        self.size = size
        self.magnitude = np.sqrt(field_x**2 + field_y**2 + field_z**2)
        self.field_x = field_x / self.magnitude
        self.field_y = field_y / self.magnitude
        self.field_z = field_z / self.magnitude

    def in_field(self, coord):
        non_neg = coord[0] >= 0 and coord[1] >= 0 and coord[2] >= 0
        in_size = coord[0] <= self.size[0] - 1 and coord[1] <= self.size[1] - 1 and coord[2] <= self.size[2] - 1
        return non_neg and in_size

    def magnitude_point(self, coord):
        assert len(coord) == 3
        points = (
            np.linspace(0, self.size[0] - 1, self.size[0]),
            np.linspace(0, self.size[1] - 1, self.size[1]),
            np.linspace(0, self.size[2] - 1, self.size[2]),
        )
        return interpn(points, self.magnitude, coord)

    def field_point(self, coord):  # It is normalized. To get original field: u.field_point()*u.magitude_point()
        assert len(coord) == 3
        if self.in_field(coord):
            points = (
                np.linspace(0, self.size[0] - 1, self.size[0]),
                np.linspace(0, self.size[1] - 1, self.size[1]),
                np.linspace(0, self.size[2] - 1, self.size[2]),
            )
            fx = interpn(points, self.field_x, coord)[0]
            fy = interpn(points, self.field_y, coord)[0]
            fz = interpn(points, self.field_z, coord)[0]
            return np.array([fx, fy, fz])
        else:
            return np.array([None, None, None])

    def streamline_pos(self, startpoint, length):
        line = []
        dt = 0.5
        point1 = np.array(startpoint)
        arc_len = 0
        while self.in_field(point1) and arc_len < length:
            line.append(point1)
            point2 = point1 + self.field_point(point1) * dt
            arc_len = arc_len + distance(point1, point2)
            point1 = point2
        return np.array(line)

    def streamline_neg(self, startpoint, length):
        line = []
        dt = -0.5
        point1 = np.array(startpoint)
        arc_len = 0
        while self.in_field(point1) and arc_len < length:
            line.append(point1)
            point2 = point1 + self.field_point(point1) * dt
            arc_len = arc_len + distance(point1, point2)
            point1 = point2
        return np.array(line)

    def streamline(self, startpoint, length):
        s_pos = self.streamline_pos(startpoint, length)
        s_neg = self.streamline_neg(startpoint, length)[1:, :]
        return np.vstack((s_neg[::-1], s_pos))


def LIC3d(vectorfield, length):
    np.random.seed(10)
    input_texture = np.random.randint(0, 2, size=vectorfield.size)
    output_texture = np.zeros(vectorfield.size)
    for i in np.arange(vectorfield.size[0]):
        for j in np.arange(vectorfield.size[1]):
            for k in np.arange(vectorfield.size[2]):
                # print(LIC_singleLine(input_texture, vectorfield.streamline((i,j,k), length)))
                output_texture[i][j][k] = LIC_singleLine(input_texture, vectorfield.streamline((i, j, k), length))

    return output_texture


def LIC2d(Vx, Vy, pix_len=5):
    '''It takes in two matrices, Vx and Vy, and returns a matrix of the same size, with each
    element being the sum of the elements of Vx and Vy at the same position, plus a random
    number between 0 and 1.

    Parameters
    ----------
    coord
        the coordinate of the pixel

    Returns
    -------
        The function in_field is being returned.

    '''
    Vx = np.asarray(Vx)
    Vy = np.asarray(Vy)
    assert Vx.shape == Vy.shape
    m, n = Vx.shape

    noise = np.random.rand(*(Vx.shape))

    def in_field(coord):
        non_neg = coord[0] >= 0 and coord[1] >= 0
        in_size = coord[0] <= m - 1 and coord[1] <= n - 1
        return non_neg and in_size


if __name__ == "__main__":
    shape = (10, 10, 10)
    np.random.seed(1)
    # ux = np.random.rand(5, 6, 7)
    ux = np.ones(shape)
    np.random.seed(3)
    uy = np.random.rand(shape[0], shape[1], shape[2])
    np.random.seed(10)
    uz = np.random.rand(shape[0], shape[1], shape[2])
    test_field = vectorfield(shape, ux, ux, ux)
    start_time = time.time()
    data = LIC3d(test_field, 5)

    print(data.shape)
    print("--- %.2f seconds ---" % (time.time() - start_time))
