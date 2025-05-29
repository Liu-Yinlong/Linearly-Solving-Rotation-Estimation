import numpy as np 
import matplotlib.pyplot as plt 
from scipy.spatial.transform import Rotation
import scipy.linalg
import pyvista as pv 
import pyvista.plotting.plotter as ppl 


def get_3d_curve(x,y):

    z_raw=np.cross(x,y,axis=1)
    z_length=scipy.linalg.norm(z_raw,axis=1)

    z=np.zeros((x.shape[0],3))
    half_theta=np.zeros((x.shape[0],1))

    ind_zero=z_length<1e-13
    ind_non_zero=np.logical_not(ind_zero)

    half_theta[ind_non_zero,:]=np.arccos(np.sum(x[ind_non_zero,:]*y[ind_non_zero,:],axis=1)).reshape((-1,1))*0.5
    z[ind_non_zero,:]=z_raw[ind_non_zero,:]/z_length[ind_non_zero].reshape((-1,1))

    if(np.sum(ind_zero)>0):
        any_direction=np.random.random((ind_zero.shape[0],3))
        z_zero_raw=any_direction-y[ind_zero,:]*np.sum(any_direction*y[ind_zero,:],axis=1).reshape((-1,1))
        z[ind_zero,:]=z_zero_raw/scipy.linalg.norm(z_zero_raw,axis=1).reshape((-1,1))
        half_theta[ind_zero,:]=0

    cos_half_theta=np.cos(half_theta)
    sin_half_theta=np.sin(half_theta)

    q123=z*sin_half_theta

    B1=np.concatenate((cos_half_theta,q123),axis=1)
    B2=np.concatenate((np.zeros_like(half_theta),cos_half_theta*y+np.cross(y,q123,axis=1)),axis=1)
    
    theta_num=360*2 #<-------------This is to seprate the quaternion circle

    theta_list=np.linspace(0,np.pi,theta_num,endpoint=False)
    
    sin_theta_list=np.sin(theta_list).reshape((theta_num,1,1))
    cos_theta_list=np.cos(theta_list).reshape((theta_num,1,1))

    b1=np.broadcast_to(B1,shape=(theta_num,B1.shape[0],B1.shape[1]))
    b2=np.broadcast_to(B2,shape=(theta_num,B2.shape[0],B2.shape[1]))
    q=b1*sin_theta_list+b2*cos_theta_list

    ind_positive=q[:,:,3]>0
    q[ind_positive,:]*=-1
    p=q[:,:,0:3]/(1-q[:,:,3:4])

    return p.reshape((-1,3))

def get_ball_data():
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    return x,y,z

def gen_data(num=20):
    gt=Rotation.from_quat([0, 0.6, 0.8, 0.4])
    x_=np.random.random((num,3))*2-1
    x=x_/scipy.linalg.norm(x_,axis=1).reshape((-1,1))
    y=gt.apply(x)
    return x,y
    
def vtk_show():
    
    pt=ppl.Plotter(off_screen=False,window_size=(600,300),shape=(1,2))
    # pt=ppl.Plotter(off_screen=False)

    pt.subplot(0,0)
    ball_mesh=pv.Sphere(radius=1,).extract_all_edges()
    pt.add_mesh(ball_mesh,color='gray',line_width=0.8,opacity=0.8)
    x,y=gen_data(num=2)
    p=get_3d_curve(x,y)
    quat_circle=pv.PolyData(p)
    pt.add_mesh(quat_circle,render_points_as_spheres=True,point_size=3,color='r')
    # pt.add_text('(a)', font_size=10,font='courier',position='lower_edge')
    # pt.show_grid(all_edges=True)


    pt.subplot(0,1)
    ball_mesh=pv.Sphere(radius=1,).extract_all_edges()
    pt.add_mesh(ball_mesh,color='gray',line_width=0.8,opacity=0.8)
    x,y=gen_data(num=20)
    p=get_3d_curve(x,y)
    quat_circle=pv.PolyData(p)
    pt.add_mesh(quat_circle,render_points_as_spheres=True,point_size=3,color='r')    
    # pt.add_text('(b)', font_size=10,font='courier',position='lower_edge')
    # pt.show_grid(all_edges=True)

    pt.link_views()
    # orbit_path=pt.generate_orbital_path(n_points=90)
    # pt.open_gif('quaternion_circle.gif',fps=20)
    # pt.orbit_on_path(orbit_path,write_frames=True,step=0.1)



    pt.show()

if __name__=='__main__':
    
    
    vtk_show()          













