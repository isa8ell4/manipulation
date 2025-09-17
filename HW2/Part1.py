from objects import *


if __name__=="__main__":

    ### Define constants in geometry of system    
    b = 0.015 #m
    a = 2*0.05*math.sin(math.radians(30)) #m

    l = 0.05

    ### Define position and rototation of frames in relation to N
    c1 = np.array([[-b], [a], [0]])
    c2 = np.array([[b], [a], [0]])
    o = np.array([[0], [a], [0]])

    angle = 30
    cos = math.cos(math.radians(angle))
    sin = math.sin(math.radians(angle))
    contactFrame1 = np.array([[cos, -sin, 0], 
                              [sin, cos, 0],
                              [0, 0, 1]])
    contactFrame2 = np.array([[cos, -sin, 0], 
                              [sin, cos, 0],
                              [0, 0, 1]])    

    ### Find grasp matrix  

    # Define finger object and manipulator objects
    left = Finger(id=1, contPos=c1, objectPos=o, R=contactFrame1)
    right= Finger(id=1, contPos=c2, objectPos=o, R=contactFrame2)
    m = Manipulator(fingers=[left, right])

    GM = m.findGraspMatrix()
    print(f"Grasp Matrix: {np.shape(GM)}")
    print(GM)

    # find twists of contact points
    ON_twist = np.array([[-0.2], [0.2], [0], [0], [0], [0.05]])
    cp_twists = np.dot(GM, ON_twist) 

    # print(f'\nTwists of Contact Points:')
    # print(cp_twists)

    # find hand jacobian
    # frame 1, frame 2
    w = np.array([[[0], [0], [1]], [[0], [0], [1]]])
    # frame 1, frame 2, frame 3 (ee)
    r_left = np.array([[[-b], [0], [0]], [[(-l/2)*math.tan(math.radians(60))-b], [l/2], [0]], [[-b], [a], [0]]])
    r_right = np.array([[[b], [0], [0]], [[(l/2)*math.tan(math.radians(60))+b], [l/2], [0]], [[b], [a], [0]]])

    left.jointAngV = w
    left.jointPs = r_left
    pHJ1 = left.partialHandJacobian()
    print(f'\npartial hand jacobian, left: {np.shape(pHJ1)}\n{pHJ1}')
    
    right.jointAngV = w
    right.jointPs = r_right
    pHJ2 = right.partialHandJacobian()
    print(f'\npartial hand jacobian, right: {np.shape(pHJ2)}\n{pHJ2}')


    HJ = block_diag(pHJ1, pHJ2)
    print(f'\nHand Jacobian: {np.shape(HJ)}\n{HJ}')
    

    # find hand-object jacobian
    HOJ = m.handObjectJacobian(GM, HJ)
    with np.printoptions(precision=4, suppress=True):
        print(f'\nHand-Object Jacobian: {np.shape(HOJ)}\n{HOJ}')

    # Calculate the values of joint velocities in order to achieve the twist
    q = np.linalg.pinv(HOJ) @ ON_twist
    print(f'\nJoint velocities to achieve twist:{np.shape(q)}\n{q} ')



