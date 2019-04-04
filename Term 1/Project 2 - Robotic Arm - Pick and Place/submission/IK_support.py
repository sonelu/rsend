import numpy as np
import tf

"""
	These are support functions for the IK
"""

# DH robot paramters
DH = [[0       , 0     , 0.75 , 0        ],
      [-np.pi/2, 0.35  , 0    , - np.pi/2],
      [0       , 1.25  , 0    , 0        ],
      [-np.pi/2, -0.054, 1.5  , 0        ],
      [np.pi/2 , 0     , 0    , 0        ],
      [-np.pi/2, 0     , 0    , 0        ],
      [0       , 0     , 0.303, 0        ]]

def Rot_X(roll):
    # computes a 4x4 transformation matrix for rotation around X axis
    return np.array([[ 1, 0           , 0            ],
                     [ 0, np.cos(roll), -np.sin(roll)],
                     [ 0, np.sin(roll),  np.cos(roll)]])

def Rot_Y(pitch):
    # computes a 4x4 transformation matrix for rotation around Y axis
    return np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                     [ 0            , 1, 0            ],
                     [-np.sin(pitch), 0, np.cos(pitch)]])

def Rot_Z(yaw):
    # computes a 4x4 transformation matrix for rotation around Z axis
    return np.array([[ np.cos(yaw), -np.sin(yaw), 0],
                     [ np.sin(yaw),  np.cos(yaw), 0],
                     [ 0          ,  0          , 1]])

def Rot_ZYX(yaw, pitch, roll):
    # combines rotations in Z, Y, X axis in this order using the angles
    # provided
    rot_zy = np.matmul(Rot_Z(yaw), Rot_Y(pitch))
    rot_zyx = np.matmul(rot_zy, Rot_X(roll))
    return rot_zyx


def GripperCorrection():
    # returns a correction matrix 4x4 with the pose correction
    # for the gripper: rotation in Z by pi and Y by -pi/2
    return np.matmul(Rot_Z(np.pi), Rot_Y(-np.pi/2.0))


def LinkTransform(params):
    # produces a homogenous transformation matrix 4x4 base on DH params
    # params is a list of length 4: alpha, a, d, theta in this order
    alpha, a, d, theta = params

    # returns a Matrix of transformation for the given DH paramters
    return np.array([[np.cos(theta)              , -np.sin(theta)             , 0             , a],
                     [np.sin(theta)*np.cos(alpha), np.cos(theta)*np.cos(alpha), -np.sin(alpha), -np.sin(alpha)*d],
                     [np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha), np.cos(alpha) , np.cos(alpha)*d],
                     [0                          , 0                          , 0             , 1]])


def ChainLinkTransform(DH, thetas, show = False):
    # builds the transformation matrices based on the DH and the theta paramters
    # it resurns the final transformation matrix
    for i in range(len(DH)):
        params = list(DH[i])    # we need to make a copy 
        params[3] += thetas[i]
        T = LinkTransform(params)
        if i == 0:
            result = T
        else:
            result = np.matmul(result, T)

        if show:
            print("T%d_%d = " % (i, i+1))
            print(T)
            print("T0_%d = " % (i+1))
            print(result)
            
    return result


def WCfromEE(req, x):
    # determines the positon of the WC from the end-effector's
    # orientation as provided in the req
    (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
                [req.poses[x].orientation.x, req.poses[x].orientation.y,
                    req.poses[x].orientation.z, req.poses[x].orientation.w])

    # calculate Rrpy matrix as in course notes
    Rrpy = np.matmul(Rot_ZYX(yaw, pitch, roll), GripperCorrection())

    # extract vector n
    n = Rrpy[0:3,2].T

    # vector p from request
    p = np.array([req.poses[x].position.x, 
                  req.poses[x].position.y,
                  req.poses[x].position.z])

    # arm constants from DH paramters
    dG = DH[6][2]
    l = 0.0

    # and calculate WC position
    wc = p - (dG + l)*n

    return wc, Rrpy


def AnglesFromWC(wc, Rrpy):
    # extract elements from DH table to make things easier to understand
    a1 = DH[1][1]
    a2 = DH[2][1]   
    a3 = DH[3][1]
    d1 = DH[0][2]
    d4 = DH[3][2]
    wcx = wc[0]
    wcy = wc[1]
    wcz = wc[2]

    # calculates th1, th2 and th3 from the position of WC
    th1 = np.arctan2(wcy, wcx)

    # we use the notation from the course diagram
    A = np.sqrt(a3**2 + d4**2)
    wcxy = np.sqrt(wcx**2 + wcy**2)
    B = np.sqrt( (wcxy - a1)**2 + ( wcz - d1)**2)
    C = a2

    cosa = (B**2 + C**2 - A**2) / (2*B*C)
    a = np.arccos(cosa)
    w = np.arctan2(wcz - d1, wcxy - a1)
    th2 = np.pi/2 - w - a

    cosb = (A**2 + C**2 - B**2) / (2*A*C)
    b = np.arccos(cosb)
    u = np.arctan2(a3, d4)
    th3 = np.pi/2 - b + u  # u is actually negative because s3 < 0

    T0_3 = ChainLinkTransform(DH[0:3], [th1, th2, th3])
    R0_3 = T0_3[0:3,0:3]
    R0_3inv = np.linalg.inv(R0_3)
    R3_6 = np.matmul(R0_3inv, Rrpy)

    th5 = np.arctan2(np.sqrt(R3_6[0][2]**2+R3_6[2][2]**2), R3_6[1][2])
    if np.sin(th5) < 0:
        th4 = np.arctan2(-R3_6[2][2], R3_6[0][2])
        th6 = np.arctan2(R3_6[1][1], -R3_6[1][0])
    else:
        th4 = np.arctan2(R3_6[2][2], -R3_6[0][2])
        th6 = np.arctan2(-R3_6[1][1], R3_6[1][0])

    return th1, th2, th3, th4, th5, th6


