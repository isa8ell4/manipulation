from objects import *


if __name__=="__main__":
    # calculate grasp matrix from 2D example
    obj = Object_2D(p_o_n=np.array([[3], [1.5]]))
    f1 = Finger_2D(id=1, p_ci_N=np.array([[0], [2]]))
    f2 = Finger_2D(id=2, p_ci_N=np.array([[2], [3]]))
    f3 = Finger_2D(id=3, p_ci_N=np.array([[4], [3]]))
    f4 = Finger_2D(id=4, p_ci_N=np.array([[3], [0]]))
    
    m_noFriction = Manipulator_2D_withoutFriction(object=obj, fingers=[f1, f2, f3, f4])
    m = Manipulator_2D(model=2, manipulator=m_noFriction)

    GM_nf = m_noFriction.findGraspMatrix()
    GM = m.findGraspMatrix()

    print(f'Grasp Matrix: {np.shape(GM)}\n{GM}')
