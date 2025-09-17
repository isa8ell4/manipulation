from objects import *


if __name__=="__main__":


    ### Define constants in geometry of system    
    b = 0.015
    a = 0.05

    ### Define position and rototation of frames in relation to N
    c1 = np.array([[-b], [a], [0]])
    c2 = np.array([[b], [a], [0]])
    o = np.array([[0], [a], [0]])
    c3 = np.array([[0], [a-b], [0]])

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
    right= Finger(id=2, contPos=c2, objectPos=o, R=contactFrame2)
    middle = Finger(id=3, contPos=c3, objectPos=o, R=contactFrame2)
    m = Manipulator(fingers=[left, right, middle])

    GM = m.findGraspMatrix()
    print(f"Grasp Matrix: {np.shape(GM)}")
    print(GM)




