from objects import *
import matplotlib.pyplot as plt



if __name__=="__main__":
    # calculate grasp matrix from 2D example
    obj = Object_2D(p_o_n=np.array([[0.03], [0.015]]))
    f1 = Finger_2D(id=1, p_ci_N=np.array([[0], [0.02]]))
    f2 = Finger_2D(id=2, p_ci_N=np.array([[0.02], [0.03]]))
    f3 = Finger_2D(id=3, p_ci_N=np.array([[0.04], [0.03]]))
    f4 = Finger_2D(id=4, p_ci_N=np.array([[0.03], [0]]))
    
    m_noFriction = Manipulator_2D_withoutFriction(object=obj, fingers=[f1, f2, f3, f4])
    m = Manipulator_2D(model=2, manipulator=m_noFriction) # hard finger model

    GM_nf = m_noFriction.findGraspMatrix()
    print(f'Grasp Matrix (before H):{np.shape(GM_nf)}\n{GM_nf} ')
    GM = m.findGraspMatrix()

    print(f'\nGrasp Matrix: {np.shape(GM)}\n{GM}')
    # m.G = GM_nf

    print(f'\nGrasp Rank: {m.checkGraspRank()}')

    minSingularVal = m.minSingularValue()
    print(f'\nMinimum Singular Value: {minSingularVal}')

    VolWrenchSpace = m.volEllipsoidWrenchSpace()
    print(f'\nVolWrenchSpace: {VolWrenchSpace}')

    gii = m.graspIsotropyIndex()
    print(f'\nGrasp Isotopy Index: {gii}')

    msv = []
    vws = []
    gii =[]


    # points = generateRectPoints(xmin=0,ymin=0,xmax=6,ymax=3)
    # for i, (x,y) in enumerate(points):
    #     f5 = Finger_2D(id=5, p_ci_N=np.array([[x], [y]]))

    #     m_noFriction = Manipulator_2D_withoutFriction(object=obj, fingers=[f1, f2, f3, f4, f5])
    #     m = Manipulator_2D(model=2, manipulator=m_noFriction) # hard finger model

    #     GM_nf = m_noFriction.findGraspMatrix()
    #     GM = m.findGraspMatrix()

    #     # print(f'\nGrasp Rank: {m.checkGraspRank()}')

    #     minSingularVal = m.minSingularValue()

    #     VolWrenchSpace = m.volEllipsoidWrenchSpace()

    #     isotropyIndex = m.graspIsotropyIndex()

    #     msv.append(minSingularVal)
    #     vws.append(VolWrenchSpace)
    #     gii.append(isotropyIndex)

    # labels = [f"({x},{y})" for x,y in points]
   

    # plt.figure(figsize=(6,5))
    # plt.scatter(range(len(points)), np.array(msv), marker='o')    

    # plt.xlabel("5th Force Vector Position")
    # plt.ylabel("Minimum Singular Value")
    # step = 10
    # plt.xticks(
    #     range(0, len(points), step),  # positions
    #     labels[::step],               # corresponding (x,y) labels
    #     rotation=90
    # )    

    # plt.tight_layout()
    # plt.show()
