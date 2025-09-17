import numpy as np
import math


def contactFrameFromN(contPos, t):
    """needs some work"""
    norm = np.linalg.norm(contPos)
    x, y, z = contPos[0], contPos[1], contPos[2]
    n = np.array([x/norm, y/norm, z/norm])

    if np.dot(n.reshape(1,3).flatten(), t.reshape(1,3).flatten()) != 0:
        print(f't is not a valid')
        return None
    
    m_flat = np.cross(t.reshape(1,3).flatten(), n.reshape(1,3).flatten())
    m = np.array([[m_flat[0]], [m_flat[1]], [m_flat[2]]])
    R = np.concatenate((m, t, n), axis=1)

    return R

class Finger():
    """
        id: id of finger
        contPos: 3x1 vector of position contact frame from N
        obejctPos: 3x1 vector of position CoM of object from N, outside info
        R: 3x3 rotational matrix of contract frame in reference to N
        P: 6x6 positional matrix of contact point to object point in ref to N
        R_bar: 6x6 matrix of R to find G
        pGM: 6x6 partial grasp matrix
        """
    def __init__(self, id, contPos, objectPos, R, P=None, R_bar=None, pGM=None):
        self.id = id
        self.contPos = contPos
        self.objectPos = objectPos
        self.R = R
        self.P = P
        self.R_bar = R_bar
        self.pGM = pGM

    def skew_symmetric_matrix(self, vector):
        vector = np.ravel(vector)
        vx, vy, vz = vector[0], vector[1], vector[2]

        skew_matrix = np.array([
            [0, -vz, vy],
            [vz, 0, -vx],
            [-vy, vx, 0]
        ])
        return skew_matrix

    def findR_bar(self):

        R_bar_upper = np.concatenate((self.R, np.zeros((3,3))), axis=1)
        R_bar_lower = np.concatenate((np.zeros((3,3)), self.R), axis=1)

        R_bar = np.concatenate((R_bar_upper, R_bar_lower), axis=0)

        self.R_bar = R_bar
        return R_bar

    def findP(self):
       
        dist = self.contPos - self.objectPos
        ss_dist = self.skew_symmetric_matrix(dist)
        ss_t_dist = ss_dist.T

        P_upper = np.concatenate((np.eye(3,3), ss_t_dist), axis=1)
        P_lower = np.concatenate((np.zeros((3,3)), np.eye(3,3)), axis=1)

        P = np.concatenate((P_upper, P_lower), axis=0)

        self.P = P
        return P

    def partialGraspMatrix(self):
        P = self.findP()
        G = self.findR_bar()

        pGM = np.dot(G, P)
        self.pGM = pGM

        # print(f'G:\n{G}\nP:\n{P}')
        # print(pGM)

        return pGM
    
class Manipulator():
    def __init__(self, fingers, object):
        self.fingers = fingers
        self.object = object



if __name__=="__main__":
    b = 0.015 #m
    a = 0.05*math.sin(math.radians(30)) #m

    c1 = np.array([[-b], [a], [0]])
    c2 = np.array([[b], [a], [0]])
    o = np.array([[0], [a], [0]])
    
    # find rotation matrix of contact frames with respect to N
    # t = np.array([[0], [0], [1]]) # choose suitable t
    # contactFrame1 = contactFrameFromN(c1, t)
    # contactFrame2 = contactFrameFromN(c2, t)
    cos = math.cos(math.radians(45))
    sin = math.sin(math.radians(45))
    contactFrame1 = np.array([[cos, -sin, 0], 
                              [sin, cos, 0],
                              [0, 0, 1]])
    contactFrame2 = np.array([[cos, -sin, 0], 
                              [sin, cos, 0],
                              [0, 0, 1]])    

    # find grasp matrix    
    left = Finger(id=1, contPos=c1, objectPos=o, R=contactFrame1)
    right= Finger(id=1, contPos=c2, objectPos=o, R=contactFrame2)
    pGM1 = left.partialGraspMatrix()
    pGM2 = right.partialGraspMatrix()

    GM = np.concatenate((pGM1, pGM2), axis=0)
    print("Grasp Matrix:")
    print(GM)

    # find twists of contact points
    ON_twist = np.array([[-0.2], [0.2], [0], [0], [0], [0.05]])
    cp_twists = np.dot(GM, ON_twist)

    print(f'\nTwists of Contact Points:')
    print(cp_twists)