import numpy as np
import math
from scipy.linalg import block_diag


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

class Finger_3D():
    """
        id: id of finger
        contPos: 3x1 vector of position contact frame from N
        obejctPos: 3x1 vector of position CoM of object from N, outside info
        R: 3x3 rotational matrix of contract frame in reference to N
        joint
        P: 6x6 positional matrix of contact point to object point in ref to N
        R_bar: 6x6 matrix of R to find G
        jointAngV: collention of joint velcoities, each 3x1
        jointPs: distance of joint frame to base frame, each 3x1
        pGM: 6x6 partial grasp matrix
        """
    def __init__(self, id, contPos, objectPos, R, P=None, R_bar=None, jointAngV = None, jointPs=None, pGM=None, pHJ=None):
        self.id = id
        self.contPos = contPos
        self.objectPos = objectPos
        self.R = R
        self.P = P
        self.R_bar = R_bar
        self.jointAngV = jointAngV
        self.jointPs = jointPs
        self.pGM = pGM
        self.pHJ = pHJ

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
        print(dist)
        ss_dist = self.skew_symmetric_matrix(dist)
        ss_t_dist = ss_dist.T

        P_upper = np.concatenate((np.eye(3,3), ss_t_dist), axis=1)
        P_lower = np.concatenate((np.zeros((3,3)), np.eye(3,3)), axis=1)

        P = np.concatenate((P_upper, P_lower), axis=0)

        self.P = P
        return P

    def partialGraspMatrix(self):
        """finds partial grasp matrix for each finger by finding P and R"""
        P = self.findP()
        R_bar = self.findR_bar()

        # need to transpose R_bar to go from C to N
        pGM = np.dot(R_bar.T, P)
        self.pGM = pGM

        print(f'R_bar:\n{R_bar}\nP:\n{P}')
        print(f'pGM:')
        print(pGM)

        return pGM
    
    def partialHandJacobian(self):
        
        P_ee_n = self.jointPs[-1]

        R_right = np.concatenate((self.R, np.zeros((3,3))), axis=0)
        R_left = np.concatenate((np.zeros((3,3)), self.R), axis=0)
        R_6x6 = np.concatenate((R_right, R_left), axis=1)


        pHJ_list = []
        
        for i, p in enumerate(self.jointPs):
            if i > 0:
                p_prev = self.jointPs[i-1]
                # print(f'p{i} = p_ee_n - p_{i-1}_n')
                # print(f'dist = {P_ee_n.flatten()} - {p_prev.flatten()}')
                
                dist = P_ee_n - p_prev
                # print(f'p{i} = {dist}')
                Jvi_flat = np.cross(self.jointAngV[i-1].reshape(1,3).flatten(), dist.reshape(1,3).flatten())
                # print(f'cross product: {Jvi_flat}')
                Jvi = np.array([[Jvi_flat[0]], [Jvi_flat[1]], [Jvi_flat[2]]])
                # print(f'Jvi: {Jvi}')
                Jwi = self.jointAngV[i-1]

                Ji = np.concatenate((Jvi, Jwi), axis=0)

                # J = R_6x6 @ Ji
                # print(f'Ji: {np.shape(Ji)}\n{Ji}')

                pHJ_list.append(Ji)
        # print(pHJ_list)
        # pHJ_list = list(reversed(pHJ_list))
        pHJ = np.concatenate(pHJ_list, axis=1)
        # print(pHJ_list)
        # print(pHJ)
        return pHJ

class Manipulator_3D_withoutFriction():
    """
    3D cases, meaning twists are 6x1 and the manipulator can move in x,y,z
    """
    def __init__(self, fingers=None, object=None):
        self.fingers = fingers
        self.object = object
    
    def findGraspMatrix(self):
        """finds the grasp matrix for a given manipulator"""
        partials = []
        for f in self.fingers: 
            pGM = f.partialGraspMatrix()
            partials.append(pGM)

        GM = np.concatenate(partials, axis=0)

        return GM

    def handObjectJacobian(self, G, J):
        """
        calculates hand-object jacobian given grasp matrix and hand jacobian
        """
        # print(np.shape(np.linalg.pinv(G.T)))
        # print(np.shape(J))

        # with np.printoptions(precision=2, suppress=True):
        #     print(f'\n(G)+ : \n{np.linalg.pinv(G)}')
        
        # G is already tansposed
        HOJ = np.linalg.pinv(G) @ J
        return HOJ
    
class Manipulator_3D():

    """
    built on top of manipulator without friction class

    model: (1) contact points with friction, (2) hard finger, (3) soft finger
    manipulatorWithoutFriction: original manipulator model
    """
    def __init__(self, model, manipulatorWithoutFriction: Manipulator_3D_withoutFriction):
        self.model = model
        self.manipulatorWithoutFriction = manipulatorWithoutFriction
    
    def H_matrix(self):
        if self.model == 1:
            H_iF = np.array([[0,0,0], 
                            [0,0,0], 
                            [0,0,1]])
            H_iM = np.zeros((3,3))
        if self.model == 2:
            H_iF = np.identity(3)
            H_iM = np.zeros((3,3))
        if self.model == 3:
            H_iF = np.identity(3)
            H_iM = np.array([[0,0,0], 
                            [0,0,0], 
                            [0,0,1]])
        H_top = np.concatenate((H_iF, np.zeros((3,3))), axis = 1)  
        H_bottom = np.concatenate((np.zeros(3,3), H_iM), axis = 1)     
        return np.concatenate((H_top, H_bottom), axis=0)  

    def findGraspMatrix(self):
        H = self.H_matrix()
        G = self.manipulatorWithoutFriction.findGraspMatrix()
        return H @ G

###### HW3

class Object_2D:
    """
    p_on: 2x1 positional vector of object origin frame in reference to N frame
    """
    def __init__(self, p_o_n):
        self.p_o_n = p_o_n

class Finger_2D:
    """
    id: id of finger
    p_ci_n: 2x1 positional vector of contact point in ref to N frame
    """
    def __init__(self, id, p_ci_N):
        self.id = id
        self.p_ci_N = p_ci_N

    def partialGraspMatrix(self, p_o_n):
        dist = self.p_ci_N - p_o_n
        pGM1 = np.array([[1, 0, dist[0][0]]])
        pGM2 = np.array([[0,1,dist[1][0]]])
        pGM3 = np.array([[0,0,1]])
        pGM = np.concatenate((pGM1, pGM2, pGM3), axis=0)
        return pGM
    
class Manipulator_2D_withoutFriction:
    """

    """
    def __init__(self, object: Object_2D, fingers: list[Finger_2D]):
        self.object = object
        self.fingers = fingers
    
    def findGraspMatrix(self):
        partials = []
        for f in self.fingers: 
            pGM = f.partialGraspMatrix(self.object.p_o_n)
            partials.append(pGM)
        GM = np.concatenate(partials, axis=0)
        return GM
            

class Manipulator_2D:
    """
    model: (1) contact points with friction, (2) hard finger, (3) soft finger

    """
    def __init__(self, model, manipulator: Manipulator_2D_withoutFriction):
        self.model = model
        self.manipulator = manipulator

    def H_matrix(self):
        if self.model == 1:
            H_iF = np.array([[0,0], 
                            [0,1]])
            H_iM = np.zeros((2,2))
        if self.model == 2:
            H_iF = np.identity(2)
            H_iM = np.zeros((2,2))
        if self.model == 3:
            H_iF = np.identity(2)
            H_iM = np.array([[0,0],  
                            [0,1]])
        H_top = np.concatenate((H_iF, np.zeros((2,2))), axis = 1)  
        H_bottom = np.concatenate((np.zeros((2,2)), H_iM), axis = 1)   

        return np.concatenate((H_top, H_bottom), axis=0)  

    def findGraspMatrix(self):
        H = self.H_matrix()
        G = self.manipulator.findGraspMatrix()
        print(np.shape(G))
        print(np.shape(H))
        amountOfHs = np.shape(G)
        return H @ G

