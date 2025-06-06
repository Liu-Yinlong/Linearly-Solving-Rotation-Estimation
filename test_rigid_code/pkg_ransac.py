import numpy as np
from numba import prange, njit
import time
import matplotlib.pyplot as plt


from scipy.spatial.transform import Rotation
def sample_spherical_normal(n_points=2):
    Q,_=np.linalg.qr(np.random.random((4,4)),)
    q1=Q[0,:]
    q2=Q[1,:]
    theta=np.linspace(0,np.pi/2,n_points)
    q=q1*np.cos(theta[:,None])+q2*np.sin(theta[:,None])
    return q   # 缩放到指定半径
def gen_data_one(num,gt,t_gt,noise_level):
    
    x=np.random.random((num,3))*2-1
    # x=x_/scipy.linalg.norm(x_,axis=1).reshape((-1,1))
    y=t_gt.T+gt.apply(x)+np.random.randn(num,3)*noise_level
    # y=y_/scipy.linalg.norm(y_,axis=1).reshape((-1,1))
    return x,y,gt,t_gt
def gen_data(num=1000,num_model=3):

    noise_level=0.01
    # num=1000
    R_gt_list=[]
    t_gt_list=[]
    w=sample_spherical_normal(n_points=num_model)
    for i in range(num_model):
        t_gt_=np.random.random((3,1))*2-1
        x_,y_,gt_,_=gen_data_one(num,Rotation.from_quat(w[i]),t_gt_,noise_level)
        if i==0:
            x=x_
            y=y_
        else:
            x=np.vstack((x,x_))
            y=np.vstack((y,y_))
        R_gt_list.append(gt_)
        t_gt_list.append(t_gt_)

    return x.T,y.T,R_gt_list,t_gt_list

def solver_svd(x,y):
    center_x=np.mean(x,axis=1,keepdims=True)
    center_y=np.mean(y,axis=1,keepdims=True)
    x_=x-center_x
    y_=y-center_y

    M=y_@x_.T
    u,s,v=np.linalg.svd(M)
    if np.linalg.det(u@v)>0:
        S=np.asarray([[1,0,0],[0,1,0],[0,0,1]])
    else:
        S=np.asarray([[1,0,0],[0,1,0],[0,0,-1]])
    R_opt=u@S@v

    t_opt=center_y-R_opt@center_x
    return R_opt,t_opt


def multi_ransac(x,y,k_num,iter_num=1000,th=0.01):
    M=x.shape[1]
    random_ind=np.random.randint(0,M,size=(iter_num,3))
    sol=np.zeros(iter_num)
    
    for ii in range(iter_num):
        ind=random_ind[ii,:]
        R_,t_=solver_svd(x[:,ind],y[:,ind])
        err_=t_+R_@x-y
        sol[ii]=np.sum(np.sqrt(np.sum(err_*err_,axis=0))<th)  
    
    largest_k_indices = np.argpartition(sol, kth=-k_num)[-k_num:]
    R_opt_list=[]
    t_opt_list=[]
    for ind_max in largest_k_indices:
        ind=random_ind[ind_max,:]
        R_,t_=solver_svd(x[:,ind],y[:,ind])
        R_opt_list.append(R_)
        t_opt_list.append(t_)
    
    return R_opt_list,t_opt_list

def ransac(x,y,iter_num=1000,th=0.01):
    M=x.shape[1]
    random_ind=np.random.randint(0,M,size=(iter_num,3))
    sol=np.zeros(iter_num)
    
    for ii in range(iter_num):
        ind=random_ind[ii,:]
        R_,t_=solver_svd(x[:,ind],y[:,ind])
        err_=t_+R_@x-y
        sol[ii]=np.sum(np.sqrt(np.sum(err_*err_,axis=0))<th)  
    
    ind_max = np.argmax(sol)
    ind=random_ind[ind_max,:]
    R_opt,t_opt=solver_svd(x[:,ind],y[:,ind])

    return R_opt,t_opt

def sequential_ransac(x,y,k_num,iter_num=1000,th=0.01):

    R_opt_list=[]
    t_opt_list=[]
    for ii in range(k_num):
        R_,t_=ransac(x,y,iter_num=iter_num,th=th)
        err_=t_+R_@x-y
        H=np.sqrt(np.sum(err_*err_,axis=0))<th
        x=x[:,~H] 
        y=y[:,~H]
        R_opt_list.append(R_)
        t_opt_list.append(t_)
    
    return R_opt_list,t_opt_list
def calc_one_err(R_gt,R_opt):
    err_rad=np.acos(0.5*(np.trace(R_gt.T@R_opt)-1))
    err_d=err_rad/np.pi*180
    return err_d

def calc_err(R_gt_list,R_opt_list,t_gt_list,t_opt_list):
    err_deg_all=np.zeros(len(R_gt_list))
    err_t_all=np.zeros(len(R_gt_list))
    for ii,R_gt in enumerate(R_gt_list):
        err_deg_=np.zeros(len(R_opt_list))
        err_t_=np.zeros(len(R_opt_list))
        t_gt=t_gt_list[ii]
        for jj,R_opt in enumerate(R_opt_list):
            err_deg_[jj]=calc_one_err(R_gt.as_matrix(),R_opt)
            err_t_[jj]=np.sqrt(np.sum((t_gt-t_opt_list[jj])**2))
        err_deg_all[ii]=np.min(err_deg_)

        err_t_all[ii]=np.min(err_t_)


    err_deg_ave=np.average(err_deg_all)
    err_t_ave=np.average(err_t_all)
    return err_deg_ave,err_t_ave

def test_ransac(num):
    repeat_time=100
    num_model=np.arange(8)+2

    tim=np.zeros((repeat_time,len(num_model)))
    ave_err_r=np.zeros((repeat_time,len(num_model)))
    ave_err_t=np.zeros((repeat_time,len(num_model)))
    for ii in range(len(num_model)):
        for jj in range(repeat_time):
                x,y,R_gt_list,t_gt_list=gen_data(num=num,num_model=num_model[ii])

                T1=time.perf_counter_ns()
                R_opt_list,t_opt_list=sequential_ransac(x,y,k_num=num_model[ii],iter_num=10000,th=0.01)
                T2=time.perf_counter_ns()
                
                tim[jj,ii]=(T2-T1)/1000000 #(变为毫秒)
                ave_err_r[jj,ii],ave_err_t[jj,ii]=calc_err(R_gt_list,R_opt_list,t_gt_list,t_opt_list)
                
                print([ii,jj])

    np.savetxt(f'./results/tim-{num}_ransac_10k.txt',tim)
    np.savetxt(f'./results/r_err-{num}_ransac_10k.txt',ave_err_r)
    np.savetxt(f'./results/t_err-{num}_ransac_10k.txt',ave_err_t)

    fig=plt.figure(figsize=(12,4))

    flierprops = dict(marker='.')
    
    ax0,ax1,ax2=fig.subplots(1,3)
    ax0.boxplot(tim,positions=np.arange(len(num_model))+2,showfliers=False)
    ax0.grid()
    ax0.set_xlabel(r'#model number')
    ax0.set_ylabel('Running time (ms)')


    ax1.boxplot(ave_err_r,positions=np.arange(len(num_model))+2,flierprops=flierprops)
    ax1.grid()
    ax1.set_xlabel(r'# model number')
    ax1.set_ylabel('Average rotation error (deg)')

    ax2.boxplot(ave_err_t,positions=np.arange(len(num_model))+2,flierprops=flierprops)
    ax2.grid()
    ax2.set_xlabel(r'# model number')
    ax2.set_ylabel('Average transaltion error')

    plt.tight_layout()
    plt.savefig(f'multi-model-{num}_10k.pdf')
    
    
    #=======================================================================================================
    tim=np.zeros((repeat_time,len(num_model)))
    ave_err_r=np.zeros((repeat_time,len(num_model)))
    ave_err_t=np.zeros((repeat_time,len(num_model)))
    for ii in range(len(num_model)):
        for jj in range(repeat_time):
                x,y,R_gt_list,t_gt_list=gen_data(num=num,num_model=num_model[ii])

                T1=time.perf_counter_ns()
                R_opt_list,t_opt_list=sequential_ransac(x,y,k_num=num_model[ii],iter_num=5000,th=0.01)
                T2=time.perf_counter_ns()
                
                tim[jj,ii]=(T2-T1)/1000000 #(变为毫秒)
                ave_err_r[jj,ii],ave_err_t[jj,ii]=calc_err(R_gt_list,R_opt_list,t_gt_list,t_opt_list)
                
                print([ii,jj])

    np.savetxt(f'./results/tim-{num}_ransac_5k.txt',tim)
    np.savetxt(f'./results/r_err-{num}_ransac_5k.txt',ave_err_r)
    np.savetxt(f'./results/t_err-{num}_ransac_5k.txt',ave_err_t)

    fig=plt.figure(figsize=(12,4))

    flierprops = dict(marker='.')
    
    ax0,ax1,ax2=fig.subplots(1,3)
    ax0.boxplot(tim,positions=np.arange(len(num_model))+2,showfliers=False)
    ax0.grid()
    ax0.set_xlabel(r'#model number')
    ax0.set_ylabel('Running time (ms)')


    ax1.boxplot(ave_err_r,positions=np.arange(len(num_model))+2,flierprops=flierprops)
    ax1.grid()
    ax1.set_yscale()
    ax1.set_xlabel(r'# model number')
    ax1.set_ylabel('Average rotation error (deg)')

    ax2.boxplot(ave_err_t,positions=np.arange(len(num_model))+2,flierprops=flierprops)
    ax2.grid()
    ax2.set_yscale()
    ax2.set_xlabel(r'# model number')
    ax2.set_ylabel('Average transaltion error')

    plt.tight_layout()
    plt.savefig(f'multi-model-{num}_5k.pdf')
if __name__=='__main__':
    # k_num=3
    # x,y,R_gt_list,t_gt_list=gen_data(num=200,num_model=k_num)
    # R_opt_list,t_opt_list=sequential_ransac(x,y,k_num,iter_num=10000,th=0.01)
    # err_deg_ave,err_t_ave=calc_err(R_gt_list,R_opt_list,t_gt_list,t_opt_list)
    # for R in R_gt_list:
    #     print(R.as_matrix())
        
    # print('++++++++++++++++')
    # for R in R_opt_list:
    #     print(R)
    # print(err_deg_ave)
    # print(err_t_ave)
    test_ransac(200)



    
