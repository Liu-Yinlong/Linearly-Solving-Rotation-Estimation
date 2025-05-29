import numpy as np 
import matplotlib.pyplot as plt 
from scipy.spatial.transform import Rotation
import scipy.linalg
import  time
import cupy as cp 

def get_R_svd(x,y):
    M=y.T@x
    u,s,v=cp.linalg.svd(M)
    if cp.linalg.det(u@v)>0:
        S=cp.asarray([[1,0,0],[0,1,0],[0,0,1]])
    else:
        S=cp.asarray([[1,0,0],[0,1,0],[0,0,-1]])
    R_m=u@S@v
    return R_m

def calc_alg(alg,x,y,R_gt):

    tim_start=time.perf_counter_ns()
    R_opt=alg(x,y)
    tim_end=time.perf_counter_ns()

    R_err=R_gt.inv()*Rotation.from_matrix(R_opt)
    err_deg=R_err.magnitude()/np.pi*180
    tim_ms=(tim_end-tim_start)/1000000
    return err_deg,tim_ms

def get_R_voting(x,y):
    x=cp.asarray(x)
    y=cp.asarray(y)

    z_raw=cp.cross(x,y,axis=1)
    z_length=cp.linalg.norm(z_raw,axis=1)

    z=cp.zeros((x.shape[0],3))
    half_theta=cp.zeros((x.shape[0],1))

    ind_zero=z_length<1e-13
    ind_non_zero=cp.logical_not(ind_zero)

    half_theta[ind_non_zero,:]=cp.arccos(cp.sum(x[ind_non_zero,:]*y[ind_non_zero,:],axis=1)).reshape((-1,1))*0.5
    z[ind_non_zero,:]=z_raw[ind_non_zero,:]/z_length[ind_non_zero].reshape((-1,1))

    if(cp.sum(ind_zero)>0):
        any_direction=cp.random.random((cp.sum(ind_zero).item(),3))
        z_zero_raw=any_direction-y[ind_zero,:]*cp.sum(any_direction*y[ind_zero,:],axis=1).reshape((-1,1))
        z_zero_raw_length=cp.linalg.norm(z_zero_raw,axis=1).reshape((-1,1))
        z[ind_zero,:]=z_zero_raw/cp.column_stack((z_zero_raw_length,z_zero_raw_length,z_zero_raw_length))
        half_theta[ind_zero,:]=0

    cos_half_theta=cp.cos(half_theta)
    sin_half_theta=cp.sin(half_theta)
    q123=z*sin_half_theta

    B1=np.concatenate((cos_half_theta,q123),axis=1)
    B2=np.concatenate((cp.zeros_like(half_theta),cos_half_theta*y+cp.cross(y,q123,axis=1)),axis=1)

    #############This is only for non-singular cases, FASTER and ROBUST#######################
    # A1=cp.column_stack((-x[:,2]-y[:,2],-x[:,1]+y[:,1],x[:,0]-y[:,0],cp.zeros(x.shape[0])))
    # A2=cp.column_stack((x[:,1]+y[:,1],-x[:,2]+y[:,2],cp.zeros(x.shape[0]),x[:,0]-y[:,0]))
    
    # B1=A1/cp.linalg.norm(A1,axis=1).reshape(-1,1)
    # B_hat=A2-B1*cp.sum(A2*B1,axis=1).reshape(-1,1)
    # B2=B_hat/cp.linalg.norm(B_hat,axis=1).reshape(-1,1)
    ##########################################################################################

    theta_num=180 #<-------------This is to seprate the quaternion circle
    blocks=180 #<----------------This is the block number for the accumulator

    hist_all=cp.zeros((blocks,blocks,blocks))

    theta_list=cp.linspace(0,cp.pi,theta_num,endpoint=False)
    sin_theta_list=cp.sin(theta_list).reshape((theta_num,1,1))
    cos_theta_list=cp.cos(theta_list).reshape((theta_num,1,1))

    divided_num=1       #< divided inputs number depend on your GPU memory
                        # if "Out of memory allocating", the increase the num
                        # if you have better GPU, decrease this number

    list_ind=cp.array_split(cp.arange(theta_num-1),divided_num)

    for list_ii in list_ind:
        b1=cp.broadcast_to(B1,shape=(len(list_ii),B1.shape[0],B1.shape[1]))
        b2=cp.broadcast_to(B2,shape=(len(list_ii),B2.shape[0],B2.shape[1]))
        q=b1*sin_theta_list[list_ii,:,:]+b2*cos_theta_list[list_ii,:,:]

        ind_positive=q[:,:,3]>0
        q[ind_positive,:]*=-1

        p=q[:,:,0:3]/(1-q[:,:,3:4])

        
        bins=cp.linspace(-1,1,blocks+1)
        hist,bin_edge=cp.histogramdd(p.reshape(-1,3),(bins,bins,bins))

        hist_all+=hist


    ind_=cp.argmax(hist_all)
    ind_xyz=cp.unravel_index(ind_,hist.shape)

    r_opt=cp.zeros(3)
    r_opt[0]=0.5*(bins[ind_xyz[0]]+bins[ind_xyz[0]+1])
    r_opt[1]=0.5*(bins[ind_xyz[1]]+bins[ind_xyz[1]+1])
    r_opt[2]=0.5*(bins[ind_xyz[2]]+bins[ind_xyz[2]+1])

    r_2=1+cp.sum(r_opt*r_opt)
    quat=cp.asarray((2*r_opt[1]/r_2,2*r_opt[2]/r_2,(r_2-2)/r_2,2*r_opt[0]/r_2,)) #<---------- This is because scipy uses{sin(theta)*n cos(theta)} for quaternion
    # R_est=Rotation.from_quat((v.get()))
    return quat
def rotation_voting_liu(x,y):

    x_data=cp.asarray(x) # 注意这里要求是unit输入  Nx3
    y_data=cp.asarray(y)
    quat_est=get_R_voting(x,y)
    r_theta=2*cp.arccos(quat_est[3])
    r_unit=(quat_est[0:3]/cp.sin(0.5*r_theta))

    x_data_trans=x_data*cp.cos(r_theta)+cp.cross(r_unit,x_data)*cp.sin(r_theta)+\
                r_unit*cp.reshape(x_data@r_unit,(-1,1))*(1-cp.cos(r_theta))
    
    x_y_cos=cp.sum(x_data_trans*y_data,axis=1)

    epsilon_inlier=3 #inlier threshold for refinement (degree) normally 3 degree is sufficient
    ind_inlier=x_y_cos>cp.cos(epsilon_inlier*cp.pi/180)

    R_est=get_R_svd(x_data[ind_inlier,:],y_data[ind_inlier,:])

    return R_est.get()
def test_rot_voting(num,outlier_rate,noise_level):
    
    R_gt=Rotation.random()
    x_data_=np.random.rand(num,3)*2-1
    x_data=x_data_/np.linalg.norm(x_data_,axis=1).reshape((-1,1))
    y_data=R_gt.apply(x_data)+noise_level*np.random.standard_normal(size=x_data.shape)

    outlier_num=round(num*(outlier_rate))
    # print(num-outlier_num)
    if(outlier_num>0):
        y_data[0:outlier_num,:]=np.random.rand(outlier_num,3)
        
    y_data=y_data/np.reshape(np.linalg.norm(y_data,axis=1),(-1,1))

    T1=time.perf_counter_ns()

    #=====开始计算===============

    R_opt=rotation_voting_liu(x_data,y_data)
    #====结束计算=================
    T2=time.perf_counter_ns()

    # print(f'时间是:{(T2-T1)/1000000}ms')
    tim=(T2-T1)/1000000

    R_err_=R_gt.inv()*Rotation.from_matrix(R_opt)
    R_err=R_err_.as_rotvec(degrees=True)
    err=np.linalg.norm(R_err)

    return tim,err
if __name__=='__main__':
    
    gt=Rotation.random(1)
    # gt=Rotation.from_rotvec([0,0,0])

    num=100000

    repeat=100
        
    outlier_level=0.1*np.linspace(0,8,9)
    
    noise_level=0.00
    
    err_my=np.zeros((repeat,9))
    tim_my=np.zeros((repeat,9))


    for ii in range(repeat):
        for jj in range(9):

            x_=np.random.random((num,3))
            x=x_/scipy.linalg.norm(x_,axis=1).reshape((-1,1))
            y_=gt.apply(x)+np.random.randn(num,3)*noise_level
            outlier_num=round(num*outlier_level[jj])
            
            if(outlier_num>0):
                y_[0:outlier_num,:]=np.random.random((outlier_num,3))*2-1
            y=y_/scipy.linalg.norm(y_,axis=1).reshape((-1,1))

            x=cp.asarray(x)
            y=cp.asarray(y)
            
            err_my[ii,jj],tim_my[ii,jj]=calc_alg(rotation_voting_liu,x,y,gt)


            print(ii,jj)

    plt.figure()
    boxprops = dict(linewidth=0.2,)
    bp=plt.boxplot(err_my,positions=np.linspace(1,9,9),widths=0.35,showfliers=True,boxprops=boxprops,patch_artist=True,whiskerprops=dict(linewidth=0.5),capprops=dict(linewidth=0.5))

    for f in bp['boxes']:
        f.set_facecolor('lime')
    f.set_label('rotation voting')

    plt.legend()
    plt.xticks(np.linspace(1,9,9), labels=[f'{x:.2f}' for x in outlier_level], rotation=45)
    plt.xlabel('outlier level')
    plt.ylabel('Rotation error (deg)')
    plt.tight_layout()
    plt.grid()
    plt.savefig('rotation_error_multi_GPU.pdf')













