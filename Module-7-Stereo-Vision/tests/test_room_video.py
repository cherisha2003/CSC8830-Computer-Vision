import numpy as np
from room_video import scale_camera_matrix, reconstruct_matched_points, point_pair_distance

def test_camera_resize():
    K=np.array([[1200,0,768],[0,1200,576],[0,0,1]],float)
    scaled=scale_camera_matrix(K,(1536,1152),(768,576))
    assert np.allclose(scaled[0],[600,0,384])
    assert np.allclose(scaled[1],[0,600,288])

def test_reference_scale_on_synthetic_two_view():
    K=np.array([[800.,0,320],[0,800,240],[0,0,1.]])
    XYZ=np.array([[-.5,0,4],[.5,0,4],[0,0,4],[0,.5,4]])
    t=np.array([-1.,0,0]);R=np.eye(3)
    left=np.column_stack([K[0,0]*XYZ[:,0]/XYZ[:,2]+320,K[1,1]*XYZ[:,1]/XYZ[:,2]+240])
    right=np.column_stack([K[0,0]*(XYZ[:,0]-1)/XYZ[:,2]+320,K[1,1]*XYZ[:,1]/XYZ[:,2]+240])
    out,scale,_=reconstruct_matched_points(left,right,K,np.zeros(5),R,t,reference_cm=100.)
    assert abs(point_pair_distance(out,0,1)-100)<1e-5
    assert abs(point_pair_distance(out,2,3)-50)<1e-5
    assert abs(scale-100)<1e-5
