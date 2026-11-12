import numpy as np
from stereo_geometry import triangulate_points, measure_shape

def test_triangulation_rectified_known_pose():
    K=np.array([[1000,0,500],[0,1000,400],[0,0,1]],dtype=float)
    points=np.array([[-8,10,80],[8,10,80],[8,-10,80],[-8,-10,80]],float)
    # P2 = [I | t] with t=-B yields camera 2 center B to right.
    baseline=15.; t=np.array([-1.,0,0]);R=np.eye(3)
    l=np.stack([K[0,0]*points[:,0]/points[:,2]+K[0,2], K[1,1]*points[:,1]/points[:,2]+K[1,2]],axis=1)
    r=np.stack([K[0,0]*(points[:,0]-baseline)/points[:,2]+K[0,2],K[1,1]*points[:,1]/points[:,2]+K[1,2]],axis=1)
    xyz=triangulate_points(l,r,K,np.zeros(5),R,t,baseline)
    assert np.allclose(xyz,points,atol=1e-6)
    m=measure_shape(xyz,'Rectangle')
    assert abs(m['width_cm']-16)<1e-6
    assert abs(m['height_cm']-20)<1e-6
    assert abs(m['mean_depth_cm']-80)<1e-6

def test_polygon_edges():
    p=np.array([[0,0,100],[3,0,100],[3,4,100]],float)
    assert np.allclose(measure_shape(p,'Polygon')['edge_lengths_cm'],[3,4,5])
