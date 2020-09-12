import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


class Edge:
    def __init__(self, points=[], trigs=[]):
        self.points = points
        self.trigs = trigs


class Triangle:
    def __init__(self, edges=[]):
        self.edges = edges
        for e in edges:
            e.trigs.append(self)

    def points(self):
        return [self.edges[0].points[0], self.edges[1].points[0], self.edges[2].points[0]]

    def toArtist(self):
        points = np.array(list(map(lambda p: np.asarray([p.x, p.y]), self.points())))
        return plt.Polygon(points[:3, :], color='C0', alpha=0.8, fill=False, clip_on=True, linewidth=1)


class Circle:
    def __init__(self, x=0, y=0, radius=0):
        self.x = x
        self.y = y
        self.radius = radius

    def fromTriangle(self, t):
        pnts = t.points()
        p1 = [pnts[0].x, pnts[0].y]
        p2 = [pnts[1].x, pnts[1].y]
        p3 = [pnts[2].x, pnts[2].y]

        temp = p2[0] * p2[0] + p2[1] * p2[1]
        bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
        cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
        det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - \
            (p2[0] - p3[0]) * (p1[1] - p2[1])

        if abs(det) < 1.0e-6:
            return False

        self.x = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
        self.y = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
        self.radius = np.sqrt((self.x - p1[0])**2 + (self.y - p1[1])**2)

        return True

    def toArtist(self):
        return plt.Circle((self.x, self.y), self.radius, color='C0',
                          fill=False, clip_on=True, alpha=0.2)


def calculateSuperTriangle(points):
    # Find boundary box that contain all points

    p_min = Point(min(points, key=lambda p: p.x).x - 0.1,
                  min(points, key=lambda p: p.y).y - 0.1)
    p_max = Point(max(points, key=lambda p: p.x).x + 0.1,
                  max(points, key=lambda p: p.y).y + 0.1)

    a = p_max.x - p_min.x
    b = p_max.y - p_min.y

    p1 = Point(p_min.x, p_min.y)
    p2 = Point(p_min.x, p_max.y + b)
    p3 = Point(p_max.x + a, p_min.y)

    points.insert(0, p1)
    points.insert(0, p2)
    points.insert(0, p3)

    e1 = Edge([p1, p2])
    e2 = Edge([p2, p3])
    e3 = Edge([p3, p1])

    t = Triangle([e1, e2, e3])
    return t


def pointInsideCircumcircle(p, t):
    pnts = t.points()

    p3 = pnts[0]
    p2 = pnts[1]
    p1 = pnts[2]

    x0 = p.x
    y0 = p.y
    x1 = p1.x
    y1 = p1.y
    x2 = p2.x
    y2 = p2.y
    x3 = p3.x
    y3 = p3.y

    ax_ = x1-x0
    ay_ = y1-y0
    bx_ = x2-x0
    by_ = y2-y0
    cx_ = x3-x0
    cy_ = y3-y0

    return (
        (ax_*ax_ + ay_*ay_) * (bx_*cy_-cx_*by_) -
        (bx_*bx_ + by_*by_) * (ax_*cy_-cx_*ay_) +
        (cx_*cx_ + cy_*cy_) * (ax_*by_-bx_*ay_)
    ) > 0

def isSharedEdge(edge, trigs):
    for t in trigs:
        for e in t.edges:
            if e.points[0].x == edge.points[0].x and e.points[0].y == edge.points[0].y and e.points[1].x == edge.points[1].x and e.points[1].y == edge.points[1].y:
                return True
            elif e.points[0].x == edge.points[1].x and e.points[0].y == edge.points[1].y and e.points[1].x == edge.points[0].x and e.points[1].y == edge.points[0].y:
                return True

    return False


def isContainPointsFromTrig(t1, t2):
    for p1 in t1.points():
        for p2 in t2.points():
            if p1.x == p2.x and p1.y == p2.y:
                return True

    return False


def createTrigFromEdgeAndPoint(edge, point):
    e1 = Edge([edge.points[0], edge.points[1]])
    e2 = Edge([edge.points[1], point])
    e3 = Edge([point, edge.points[0]])
    t = Triangle([e1, e2, e3])

    return t


def checkDelaunay(triangle):
    for e in triangle.edges:
        for t in e.trigs:
            if t == triangle:
                continue
            for p in t.points():
                if pointInsideCircumcircle(p, triangle):
                    print('Alert')
    return 1


def calculateCircle(t):
    pnts = t.points()
    p1 = [pnts[0].x, pnts[0].y]
    p2 = [pnts[1].x, pnts[1].y]
    p3 = [pnts[2].x, pnts[2].y]

    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return (None, np.inf)

    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return ((cx, cy), radius)

# Uncomment this for user point input:

'''
points = []
N = int(input()) # Count of points
for i in range(N):
    xy = list(map(int, input().split(' ')[:2]))
    points.append(Point(xy[0], xy[1]))
'''

# And comment next 2 lines:

N = 20 # Count of points
points = list(map(lambda p: Point(p[0], p[1]), np.random.rand(N, 2)))
for p in points:
    p.x = p.x * 1.5

# Until this line

super_trig = calculateSuperTriangle(points)
trigs = [super_trig]

def init():
    np_points = np.array(list(map(lambda p: np.asarray([p.x, p.y]), points)))
    plt.scatter(np_points[:, 0], np_points[:, 1], s=15)
    return []


def animate(i):
    p = points[i]
    bad_trigs = []
    for t in trigs:
        if pointInsideCircumcircle(p, t):
            bad_trigs.append(t)
    poly = []
    for b_t in bad_trigs:
        for e in b_t.edges:
            copied_bad_trigs = bad_trigs[:]
            copied_bad_trigs.remove(b_t)
            if not isSharedEdge(e, copied_bad_trigs):
                poly.append(e)
    for b_t in bad_trigs:
        trigs.remove(b_t)
    for e in poly:
        T = createTrigFromEdgeAndPoint(e, p)
        trigs.append(T)

    plt.cla()

    # draw points

    np_points = np.array(list(map(lambda p: np.asarray([p.x, p.y]), points)))
    plt.scatter(np_points[:, 0], np_points[:, 1], s=15)

    # draw triangles and circles

    artists = []
    for t in trigs[:]:
        trig_artist = t.toArtist()
        # artists.append(trig_artist)
        plt.gca().add_patch(trig_artist)
        c = Circle()
        c.fromTriangle(t)
        circ_artist = c.toArtist()
        # artists.append(circ_artist)
        # plt.gca().add_artist(circ_artist) # Uncomment for circle drawing

    return artists


fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')

fanim = animation.FuncAnimation(
    fig, animate, init_func=init, frames=N + 3, interval=100, blit=True)

# Comment this line if you want see triangulation in window:

fanim.save('triangulation.gif', writer='pillow')

# And uncomment this line:

# plt.show()

