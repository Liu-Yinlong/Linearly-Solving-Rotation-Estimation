import numpy as np 
import matplotlib.pyplot as plt 
from scipy.spatial.transform import Rotation
import scipy.linalg
import  time
import cupy as cp 
from cupyx.scipy.ndimage import maximum_filter

plt.style.use('classic')

def get_R_svd(x,y):
    M=y.T@x
    u,s,v=np.linalg.svd(M)
    if np.linalg.det(u@v)>0:
        S=np.asarray([[1,0,0],[0,1,0],[0,0,1]])
    else:
        S=np.asarray([[1,0,0],[0,1,0],[0,0,-1]])
    R_m=u@S@v
    return R_m


def refine_R(R_opt_list_in,x,y):
    R_opt_list=[]
    for R_opt in R_opt_list_in:
        for ii in range(10):
            x_=R_opt.apply(x)
            err=x_-y
            ind=np.sqrt(np.sum(err*err,axis=1))<=0.05
            R_m=get_R_svd(x[ind,:],y[ind,:])
            R_opt.from_matrix(R_m)
        
        R_opt_list.append(R_opt)

    return R_opt_list

def calc_one_err(R_gt,R_opt):
    err_rad=np.acos(0.5*(np.trace(R_gt.T@R_opt)-1))
    err_d=err_rad/np.pi*180
    return err_d

def calc_err(R_gt_list,R_opt_list):
    err_deg_all=np.zeros(len(R_gt_list))
    for ii,R_gt in enumerate(R_gt_list):
        err_deg_=np.zeros(len(R_opt_list))
        for jj,R_opt in enumerate(R_opt_list):
            err_deg_[jj]=calc_one_err(R_gt.as_matrix(),R_opt.as_matrix())
        err_deg_all[ii]=np.min(err_deg_)
        # if(np.min(err_deg_))>5:
        #     print('error!')


    err_deg_ave=np.average(err_deg_all)
    return err_deg_ave
def find_topk_peaks_3d_gpu(data: cp.ndarray, k: int, radius: int):
    """
    在三维数组上使用GPU加速查找Top-K峰值（带非极大值抑制）
    
    参数:
        data : 三维输入数组 (shape: [Depth, Height, Width])
        k    : 返回的最大峰值数量
        radius: 非极大值抑制半径（欧几里得距离）
    
    返回:
        peaks: 峰值坐标数组, shape=[N, 3] (每行是(z,y,x)坐标)
        values: 对应的峰值强度值, shape=[N]
        (N = min(k, 找到的峰值数))
    """
    # 创建排除中心点的footprint
    footprint = cp.ones((3, 3, 3), dtype=cp.bool_)
    footprint[1, 1, 1] = False  # 排除中心点
    
    # 计算邻域最大值（排除中心点）
    neighbor_max = maximum_filter(data, footprint=footprint, mode='constant', cval=-cp.inf)
    
    # 标记严格局部极大值 (且排除NaN值)
    valid_mask = cp.logical_not(cp.isnan(data))
    is_peak = cp.logical_and(data > neighbor_max, valid_mask)
    
    # 提取峰值坐标和值
    coords = cp.stack(cp.where(is_peak), axis=1)
    values = data[is_peak]
    
    if len(values) == 0:
        return np.empty((0, 3)), np.empty(0)
    
    # 按值降序排序 (GPU上)
    sorted_indices = cp.argsort(values)[::-1]
    coords_sorted = coords[sorted_indices]
    values_sorted = values[sorted_indices]
    
    # 传输到CPU进行非极大值抑制处理
    coords_cpu = cp.asnumpy(coords_sorted)
    values_cpu = cp.asnumpy(values_sorted)
    
    # 非极大值抑制
    suppressed = np.zeros(len(coords_cpu), dtype=bool)
    output_coords = []
    output_values = []
    r_sq = radius ** 2  # 使用平方距离加速计算
    
    for i in range(len(coords_cpu)):
        if suppressed[i]:
            continue
            
        # 添加到输出
        output_coords.append(coords_cpu[i])
        output_values.append(values_cpu[i])
        
        # 如果达到K则提前终止
        if len(output_coords) >= k:
            break
        
        # 计算与后续所有点的距离
        if i + 1 < len(coords_cpu):
            # 仅计算后续未抑制的点
            mask = ~suppressed[i+1:]
            if not mask.any():
                continue
                
            j = i + 1
            active_coords = coords_cpu[j:]
            diffs = active_coords - coords_cpu[i]
            dists_sq = np.sum(diffs**2, axis=1)
            
            # 标记在抑制半径内的点
            to_suppress = dists_sq <= r_sq
            suppressed[j:] = np.logical_or(suppressed[j:], to_suppress)
    
    return np.array(output_coords), np.array(output_values)


def get_opt_multi(hist_all,k,bins):

    # ind_=cp.argmax(hist_all)
    # ind_xyz=cp.unravel_index(ind_,hist_all.shape)
    ind_xyz_all,_=find_topk_peaks_3d_gpu(hist_all, k,radius=5)
    # print('======')
    R_opt_list=[]
    for ind_xyz in ind_xyz_all:
        r_opt=np.zeros(3)
        r_opt[0]=0.5*(bins[ind_xyz[0]]+bins[ind_xyz[0]+1])
        r_opt[1]=0.5*(bins[ind_xyz[1]]+bins[ind_xyz[1]+1])
        r_opt[2]=0.5*(bins[ind_xyz[2]]+bins[ind_xyz[2]+1])

        r_2=1+np.sum(r_opt*r_opt)
        quat=np.asarray((2*r_opt[1]/r_2,2*r_opt[2]/r_2,(r_2-2)/r_2,2*r_opt[0]/r_2,)) #<---------- This is because scipy uses{sin(theta)*n cos(theta)} for quaternion
        R_est=Rotation.from_quat((quat))
        R_opt_list.append(R_est)

    return R_opt_list



def get_R_voting(x,y,k_num):
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

    theta_num=180*3 #<-------------This is to seprate the quaternion circle
    blocks=180*1 #<----------------This is the block number for the accumulator

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


    R_opt_list=get_opt_multi(hist_all,k=k_num,bins=bins)
    # R_opt_list=refine_R(R_opt_list,x.get(),y.get())
    return R_opt_list


def gen_data_one(num,gt,noise_level):
    
    x_=np.random.random((num,3))
    x=x_/scipy.linalg.norm(x_,axis=1).reshape((-1,1))
    y_=gt.apply(x)+np.random.randn(num,3)*noise_level
    y=y_/scipy.linalg.norm(y_,axis=1).reshape((-1,1))
    

    return x,y,gt
def sample_spherical_normal(n_points=2):
    Q,_=np.linalg.qr(np.random.random((4,4)),)
    q1=Q[0,:]
    q2=Q[1,:]
    theta=np.linspace(0,np.pi,n_points)
    q=q1*np.cos(theta[:,None])+q2*np.sin(theta[:,None])
    return q   # 缩放到指定半径

def gen_data(num=1000,num_model=3):

    noise_level=0.001
    # num=1000
    R_gt_list=[]
    w=sample_spherical_normal(n_points=num_model)
    for i in range(num_model):
        x_,y_,gt_=gen_data_one(num,Rotation.from_quat(w[i]),noise_level)
        if i==0:
            x=x_
            y=y_
        else:
            x=np.vstack((x,x_))
            y=np.vstack((y,y_))
        R_gt_list.append(gt_)

    x=cp.asarray(x)
    y=cp.asarray(y)

    return x,y,R_gt_list

def test_rot_voting(num):
    
    repeat_time=100
    num_model=np.arange(8)+2

    tim=np.zeros((repeat_time,len(num_model)))
    err_deg=np.zeros((repeat_time,len(num_model)))

    for ii in range(len(num_model)):
        for jj in range(repeat_time):
                x,y,R_gt_list=gen_data(num=num,num_model=num_model[ii])

                T1=time.perf_counter_ns()
                R_opt_list=get_R_voting(x,y,k_num=num_model[ii])
                T2=time.perf_counter_ns()
                
                tim[jj,ii]=(T2-T1)/1000000 #(变为毫秒)
                err_deg[jj,ii]=calc_err(R_gt_list,R_opt_list)
                
                print([ii,jj])

    np.savetxt(f'./results/tim-{num}.txt',tim)
    np.savetxt(f'./results/err-{num}.txt',err_deg)
    

    tim=np.loadtxt(f'./results/tim-{num}.txt')
    err=np.loadtxt(f'./results/err-{num}.txt')

    fig=plt.figure(figsize=(12,4))
    ax0,ax1=fig.subplots(1,2)
    ax0.boxplot(tim,positions=np.arange(len(num_model))+2,showfliers=False)
    ax0.grid()
    ax0.set_xlabel(r'#model number')
    ax0.set_ylabel('time (ms)')


    ax1.boxplot(err_deg,positions=np.arange(len(num_model))+2)
    ax1.grid()
    ax1.set_xlabel(r'# model number')
    ax1.set_ylabel('average rotation error (deg)')

    plt.tight_layout()
    plt.savefig(f'multi-model-{num}.pdf')
    

    return tim,err_deg
if __name__=='__main__':
    num_list=1000*(np.arange(10)+1)
    for num in num_list:
        test_rot_voting(num=num)
    test_rot_voting(num=6000)
   
    # num_model=np.arange(8)+2
    # num_list=[3000,6000,9000]

    # fig=plt.figure(figsize=(8,3))
    # plt.rcParams['font.size']=12
    # ax0,ax1=fig.subplots(1,2)
    # hatches=['\\\\','////','++']
    # for ii,num in enumerate(num_list):
        
    #     tim=np.loadtxt(f'./results/tim-{num}.txt')
    #     err_deg=np.loadtxt(f'./results/err-{num}.txt')

    #     positions=np.arange(len(num_model))+2-0.3+ii*0.3
    #     print(positions)

    #     # bp=ax0.boxplot(tim,positions=positions,showfliers=False,patch_artist=True,widths=0.2)
    #     # for f1 in bp['boxes']:
    #     #     f1.set_facecolor('lime')
    #     # f1.set_label('rotation voting')

    #     bp=ax0.bar(x=positions,height=np.median(tim,axis=0),bottom=0,width=0.3)
    #     for f1 in bp.patches:
    #         # f1.set_facecolor('lime')
    #         f1.set_hatch(hatches[ii])
    #     f1.set_label(f'#{num}')
      
    #     ax0.grid()
    #     ax0.set_xlabel(r'# model number')
    #     ax0.set_title('Median time (ms)')
        
    #     ax0.legend(loc=0,frameon=False)


    #     # ax1.violinplot(err_deg,positions=positions,showextrema=True,showmeans=False, showmedians=False,widths=0.15 )
    #     # bp=ax1.boxplot(err_deg,positions=positions,widths=0.1,showfliers=False,showcaps=False,patch_artist=True )
    #     # for f2 in bp['boxes']:
    #     #     f2.set_facecolor('lime')
    #     # f2.set_label('rotation voting')
                
    #     bp=ax1.bar(x=positions,height=np.mean(err_deg,axis=0),bottom=0,width=0.3)
    #     for f1 in bp.patches:
    #         # f1.set_facecolor('lime')
    #         f1.set_hatch(hatches[ii])
    #     # f1.set_label(f'Each model #{num}')

    #     # ax1.legend(loc=0)

    #     ax1.grid()
        
    #     ax1.set_xlabel(r'# model number')
    #     ax1.set_title('Average rotation error (deg)')
    
    # # fig.legend(loc='outside upper center',ncols=3,frameon=True,bbox_to_anchor=(0., 1.1, 1., .102))

    # ax0.set_xticks(np.arange(len(num_model))+2)
    # ax1.set_xticks(np.arange(len(num_model))+2)
    # plt.tight_layout(pad=0)
    # plt.savefig('multi-model.pdf')
        














