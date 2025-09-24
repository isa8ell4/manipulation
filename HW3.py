from objects import *



if __name__=="__main__":
    # calculate grasp matrix from 2D example
    obj = Object_2D(p_o_n=np.array([[3], [1.5]]))
    f1 = Finger_2D(id=1, p_ci_N=np.array([[0], [2]]))
    f2 = Finger_2D(id=2, p_ci_N=np.array([[2], [3]]))
    f3 = Finger_2D(id=3, p_ci_N=np.array([[4], [3]]))
    f4 = Finger_2D(id=4, p_ci_N=np.array([[3], [0]]))
    
    m_noFriction = Manipulator_2D_withoutFriction(object=obj, fingers=[f1, f2, f3, f4])
    m = Manipulator_2D(model=2, manipulator=m_noFriction) # hard finger model

    GM_nf = m_noFriction.findGraspMatrix()
    print(f'Grasp Matrix (before H):{np.shape(GM_nf)}\n{GM_nf} ')
    # GM = m.findGraspMatrix()

    # print(f'\nGrasp Matrix: {np.shape(GM)}\n{GM}')
    m.G = GM_nf

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

    maxMSV = 0.0
    points = generateRectPoints(xmin=0,ymin=0,xmax=6,ymax=3)
    for i, (x,y) in enumerate(points):
        f5 = Finger_2D(id=5, p_ci_N=np.array([[x], [y]]))

        m_noFriction = Manipulator_2D_withoutFriction(object=obj, fingers=[f1, f2, f3, f4, f5])
        m = Manipulator_2D(model=2, manipulator=m_noFriction) # hard finger model

        GM_nf = m_noFriction.findGraspMatrix()
        # GM = m.findGraspMatrix()
        m.G = GM_nf

        # print(f'\nGrasp Rank: {m.checkGraspRank()}')

        minSingularVal = m.minSingularValue()
 

        VolWrenchSpace = m.volEllipsoidWrenchSpace()
        # if VolWrenchSpace > 0:
        #     print(VolWrenchSpace)

        isotropyIndex = m.graspIsotropyIndex()

        msv.append(minSingularVal)
        vws.append(VolWrenchSpace)
        gii.append(isotropyIndex)


    plotAgainstPoints(points=points, yVals=msv, yLabel="Minimum Singular Value")

    plotAgainstPoints(points=points, yVals=vws, yLabel="Volume of Ellipoid")

    plotAgainstPoints(points=points, yVals=gii, yLabel="Isotropy Index")

    msvTop5 = maxValues(points, msv)
    print(f'\ntop 5 minimum singular values: {msvTop5}')

    vwsTop5 = maxValues(points, vws)
    print(f'\ntop 5 vol wrench spaces: {vwsTop5}')


    giiTop5 = maxValues(points, gii)
    print(f'\ntop 5 isotropy indicies: {giiTop5}')