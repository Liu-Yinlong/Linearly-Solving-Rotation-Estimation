import numpy as np 
import matplotlib.pyplot as plt 
from scipy.spatial.transform import Rotation
import scipy.linalg
import  time
import profile

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

def get_R_voting(x,y):

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

    #############This is only for non-singular cases, FASTER and ROBUST#######################
    # A1=np.column_stack((-x[:,2]+y[:,2],-x[:,1]-y[:,1],x[:,0]+y[:,0],np.zeros(x.shape[0])))
    # A2=np.column_stack((x[:,1]-y[:,1],-x[:,2]-y[:,2],np.zeros(x.shape[0]),x[:,0]+y[:,0]))
    
    # B1_=A1/scipy.linalg.norm(A1,axis=1).reshape(-1,1)
    # B_hat=A2-B1_*np.sum(A2*B1_,axis=1).reshape(-1,1)
    # B2_=B_hat/scipy.linalg.norm(B_hat,axis=1).reshape(-1,1)
    ##########################################################################################
    
    theta_num=180 #<-------------This is to seprate the quaternion circle
    blocks=180 #<----------------This is the block number for the accumulator

    hist_all=np.zeros((blocks,blocks,blocks))

    theta_list=np.linspace(0,np.pi,theta_num,endpoint=False)
    sin_theta_list=np.sin(theta_list).reshape((theta_num,1,1))
    cos_theta_list=np.cos(theta_list).reshape((theta_num,1,1))

    divided_num=1      #< divided inputs number depend on your computer memory
                        # if "Out of memory allocating", the increase the num
                        # if you have large RAM, decrease this number

    list_ind=np.array_split(np.arange(theta_num-1),divided_num)

    for list_ii in list_ind:
        b1=np.broadcast_to(B1,shape=(len(list_ii),B1.shape[0],B1.shape[1]))
        b2=np.broadcast_to(B2,shape=(len(list_ii),B2.shape[0],B2.shape[1]))
        q=b1*sin_theta_list[list_ii,:,:]+b2*cos_theta_list[list_ii,:,:]

        ind_positive=q[:,:,3]>0
        q[ind_positive,:]*=-1

        p=q[:,:,0:3]/(1-q[:,:,3:4])

        
        bins=np.linspace(-1,1,blocks+1)
        hist,bin_edge=np.histogramdd(p.reshape(-1,3),(bins,bins,bins))

        hist_all+=hist


    ind_=np.argmax(hist_all)
    ind_xyz=np.unravel_index(ind_,hist.shape)
    
    r_opt=np.zeros(3)
    r_opt[0]=0.5*(bins[ind_xyz[0]]+bins[ind_xyz[0]+1])
    r_opt[1]=0.5*(bins[ind_xyz[1]]+bins[ind_xyz[1]+1])
    r_opt[2]=0.5*(bins[ind_xyz[2]]+bins[ind_xyz[2]+1])

    r_2=1+np.sum(r_opt*r_opt)
    v=np.asarray((2*r_opt[1]/r_2,2*r_opt[2]/r_2,(r_2-2)/r_2,2*r_opt[0]/r_2,)) #<---------- This is because scipy uses{sin(theta)*n cos(theta)} for quaternion
    R_est=Rotation.from_quat((v))
    return R_est

def rotation_voting_liu(x,y):

    x_data=np.asarray(x) # 注意这里要求是unit输入  Nx3
    y_data=np.asarray(y)
    R_est=get_R_voting(x,y)

    x_data_trans=R_est.apply(x_data)
    x_y_cos=np.sum(x_data_trans*y_data,axis=1)

    epsilon_inlier=3 #inlier threshold for refinement (degree)
    ind_inlier=x_y_cos>np.cos(epsilon_inlier*np.pi/180)

    R_=get_R_svd(x_data[ind_inlier,:],y_data[ind_inlier,:])

    return R_

if __name__=='__main__':
    
    gt=Rotation.random(1)

    num=2000
    repeat=100

    err_my=np.zeros((repeat,9))
    tim_my=np.zeros((repeat,9))

    outlier_level=0.1*np.linspace(1,9,9)
    noise_level=0.01

    for ii in range(repeat):
        for jj in range(9):

            x_=np.random.random((num,3))
            x=x_/scipy.linalg.norm(x_,axis=1).reshape((-1,1))
            y_=gt.apply(x)+np.random.randn(num,3)*noise_level
            outlier_num=round(num*outlier_level[jj])
            if(outlier_num>0):
                y_[0:outlier_num,:]=np.random.random((outlier_num,3))*2-1
            y=y_/scipy.linalg.norm(y_,axis=1).reshape((-1,1))
            err_my[ii,jj],tim_my[ii,jj]=calc_alg(rotation_voting_liu,x,y,gt)

            print(ii,jj)

    plt.figure()
    boxprops = dict(linewidth=0.2,)
    bp=plt.boxplot(err_my,positions=np.linspace(0,8,9),widths=0.35,showfliers=True,boxprops=boxprops,patch_artist=True,whiskerprops=dict(linewidth=0.5),capprops=dict(linewidth=0.5))

    for f in bp['boxes']:
        f.set_facecolor('lime')
    f.set_label('quaternion circle based method')

    plt.legend()
    plt.xticks(np.linspace(0,8,9), labels=[f'{x:.2f}' for x in outlier_level], rotation=45)
    plt.xlabel('Outlier level')
    plt.ylabel('Rotation error (deg)')
    plt.tight_layout()
    plt.grid()
    plt.savefig('outlier-err.pdf')

    plt.figure()
    boxprops = dict(linewidth=0.2,)
    bp=plt.boxplot(tim_my,positions=np.linspace(0,8,9),widths=0.35,showfliers=True,boxprops=boxprops,patch_artist=True,whiskerprops=dict(linewidth=0.5),capprops=dict(linewidth=0.5))

    for f in bp['boxes']:
        f.set_facecolor('lime')
    f.set_label('quaternion circle based method')
    plt.legend()
    plt.xticks(np.linspace(0,8,9), labels=[f'{x:.2f}' for x in outlier_level], rotation=45)
    plt.xlabel('Outlier level')
    plt.ylabel('Time (ms)')
    plt.tight_layout()
    plt.grid()
    plt.savefig('outlier-time.pdf')













