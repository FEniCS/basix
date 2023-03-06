import basix
import numpy as np
np.set_printoptions(suppress=True)

def test_ordering():

    pt = np.array([[1/3,1/3]])

    # reordered element
    el = basix.create_element(basix.ElementFamily.P, basix.CellType.triangle, 2, dof_layout=[0,3,5,1,2,4])
    ord = el.dof_ordering
    result1 = el.tabulate(1, pt)

    # standard element
    el = basix.create_element(basix.ElementFamily.P, basix.CellType.triangle, 2)
    result2 = el.tabulate(1, pt)
    presult = np.zeros_like(result2)

    # permute standard data
    presult[:,:,ord,:] = result2

    assert np.allclose(result1, presult)
