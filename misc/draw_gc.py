import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv

# 定义球面参数
r = 1  # 半径
theta = np.linspace(0, 2 * np.pi, 25)  # 经度
phi = np.linspace(0, np.pi, 30)        # 纬度
theta, phi = np.meshgrid(theta, phi)

# 转换为笛卡尔坐标
x = r * np.sin(phi) * np.cos(theta)
y = r * np.sin(phi) * np.sin(theta)
z = r * np.cos(phi)

Q,R=np.linalg.qr(np.random.random((3,3)))

r1=Q[0]
r2=Q[1]
theta=np.linspace(0,np.pi*2,180)
sin_theta=np.sin(theta[:,None])
cos_theta=np.cos(theta[:,None])

gc_point_1=sin_theta*r1+cos_theta*r2




pt = pv.Plotter(shape=(1, 2),window_size=(800,400))

pt.subplot(0,0)
sphere=pv.StructuredGrid(x,y,z)
disk=pv.Circle(radius=1)
print(gc_point_1.shape)
gc_1=pv.Spline(gc_point_1)
pt.add_mesh(sphere,style='wireframe',
            render_lines_as_tubes=True,line_width=1,color='grey')
pt.add_mesh(disk,color='gray',opacity=0.6)
pt.add_text('(a)', font_size=14,font='courier',position='lower_edge')
pt.add_mesh(gc_1,render_lines_as_tubes=True,line_width=3,color='r')


r1=Q[2]
r2=Q[1]

gc_point_2=sin_theta*r1+cos_theta*r2
gc_2=pv.Spline(gc_point_2)
pt.subplot(0,1)
sphere=pv.StructuredGrid(x,y,z)
pt.add_mesh(sphere,style='wireframe',
            render_lines_as_tubes=True,line_width=1,color='grey')

pt.add_mesh(disk,color='gray',opacity=0.6)
pt.add_text('(b)', font_size=14,font='courier',position='lower_edge')
pt.add_mesh(gc_1,render_lines_as_tubes=True,line_width=3,color='r')
pt.add_mesh(gc_2,render_lines_as_tubes=True,line_width=3,color='r')





pt.link_views()  # link all the views
path = pt.generate_orbital_path(n_points=72,factor=1,viewup=(1,-1,1))
pt.open_gif('orbit.gif')
pt.orbit_on_path(path, write_frames=True)
pt.show()

