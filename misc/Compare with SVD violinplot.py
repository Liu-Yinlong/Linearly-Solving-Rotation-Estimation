import numpy as np 
import matplotlib.pyplot as plt 
from scipy.spatial.transform import Rotation
import scipy.linalg
import  time
#import scienceplots
plt.rcParams['font.family']=['arial','DengXian']

def get_R_svd(x,y):
    M=y.T@x
    u,s,v=np.linalg.svd(M)
    if np.linalg.det(u@v)>0:
        S=np.asarray([[1,0,0],[0,1,0],[0,0,1]])
    else:
        S=np.asarray([[1,0,0],[0,1,0],[0,0,-1]])
    R_m=u@S@v
    return Rotation.from_matrix(R_m)

def calc_alg(alg,x,y,R_gt):

    tim_start=time.perf_counter_ns()
    R_opt=alg(x,y)
    tim_end=time.perf_counter_ns()

    R_err=R_gt.inv()*R_opt
    err_deg=R_err.magnitude()/np.pi*180
    tim_ms=(tim_end-tim_start)/1000000
    return err_deg.item(),tim_ms

def get_R_QC(x,y):

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

    A1=np.concatenate((cos_half_theta,q123),axis=1)
    A2=np.concatenate((np.zeros_like(half_theta),cos_half_theta*y+np.cross(y,q123,axis=1)),axis=1)

    B1=np.zeros_like(A1)
    B2=np.zeros_like(A2)

    for ii in range(A1.shape[0]):
        A=np.vstack((A1[ii,:],A2[ii,:]))
        C=scipy.linalg.null_space(A)
        B1[ii,:]=C[:,0].T
        B2[ii,:]=C[:,1].T

    #############This is only for non-singular cases, FASTER and ROBUST#######################
    # A1=np.column_stack((-x[:,2]+y[:,2],-x[:,1]-y[:,1],x[:,0]+y[:,0],np.zeros(x.shape[0])))
    # A2=np.column_stack((x[:,1]-y[:,1],-x[:,2]-y[:,2],np.zeros(x.shape[0]),x[:,0]+y[:,0]))
    
    # B1_=A1/scipy.linalg.norm(A1,axis=1).reshape(-1,1)
    # B_hat=A2-B1_*np.sum(A2*B1_,axis=1).reshape(-1,1)
    # B2_=B_hat/scipy.linalg.norm(B_hat,axis=1).reshape(-1,1)
    ##########################################################################################

    A=np.concatenate((B1,B2),axis=0)

    Q=A.T@A
    u,s,v=np.linalg.svd(Q)
    R_est=Rotation.from_quat((v[-1,1],v[-1,2],v[-1,3],v[-1,0]))#<---------- This is because scipy uses{sin(theta)*n cos(theta)} for quaternion
    return R_est


if __name__=='__main__':
    
    gt=Rotation.random(1)

    num=1000

    repeat=100

    err_my=np.zeros((repeat,10))
    tim_my=np.zeros((repeat,10))

    err_svd=np.zeros((repeat,10))
    tim_svd=np.zeros((repeat,10))

    noise_level=0.01*np.linspace(1,10,10)

    for ii in range(repeat):
        for jj in range(10):

            x_=np.random.random((num,3))*2-1
            x=x_/scipy.linalg.norm(x_,axis=1).reshape((-1,1))
            y_=gt.apply(x)+np.random.randn(num,3)*noise_level[jj]

            y=y_/scipy.linalg.norm(y_,axis=1).reshape((-1,1))

            
            err_my[ii,jj],tim_my[ii,jj]=calc_alg(get_R_QC,x,y,gt)
            err_svd[ii,jj],tim_svd[ii,jj]=calc_alg(get_R_svd,x,y,gt)

            print(ii,jj)

    
    plt.figure(figsize=(8,4))

    bp1=plt.violinplot(err_svd,positions=np.linspace(1,10,10)+0.25,widths=0.35)

    bp2=plt.violinplot(err_my,positions=np.linspace(1,10,10)-0.25,widths=0.35)

    for f1 in bp1['bodies']:
        f1.set_facecolor('blue')
        f1.set_alpha(1)   
    f1.set_label('SVD')

    for f2 in bp2['bodies']:
        f2.set_facecolor('lime')
        f2.set_alpha(1)  
    f2.set_label('Linear method')

    plt.legend()
    plt.xticks(np.linspace(1,10,10), labels=[f'{0.01*x:.2f}' for x in np.linspace(1,10,10)], rotation=45)
    plt.xlabel('Noise level')
    plt.ylabel('Rotation error (deg)')
    plt.tight_layout()
    plt.grid()

    plt.savefig('outlier_free_err_violinplot.svg')
