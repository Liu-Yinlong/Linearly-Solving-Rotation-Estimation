from rot_voting_cupy import *
import scienceplots
plt.style.use(['science','ieee','no-latex'])

# ######============这里是noise =============================

num=5000


noise_len=10
noise_list=np.arange(noise_len)*0.01+0.01
outlier_rate=0.9

repeat_time=500 

tim=np.zeros((repeat_time,noise_len))
r_err=np.zeros((repeat_time,noise_len))
t_err=np.zeros((repeat_time,noise_len))

for ii in range(repeat_time):
    for jj in range(noise_len):

        [tim[ii,jj],r_err[ii,jj],t_err[ii,jj]]=test_rigid_pose(num,outlier_rate,noise_list[jj])
        print([ii,jj])

np.savetxt('./save_data/tim_noise_gpu_only_rigid.txt',tim)
np.savetxt('./save_data/r_err_noise_gpu_only_rigid.txt',r_err)
np.savetxt('./save_data/t_err_noise_gpu_only_rigid.txt',t_err)

tim=np.loadtxt('./save_data/tim_noise_gpu_only_rigid.txt')
r_err=np.loadtxt('./save_data/r_err_noise_gpu_only_rigid.txt')
t_err=np.loadtxt('./save_data/t_err_noise_gpu_only_rigid.txt')

plt.figure(figsize=(2,1.6))
plt.boxplot(tim,showfliers=False)
plt.gca().set_xticklabels(labels=[f'{x:0.2f}' for x in noise_list],rotation=45)
plt.title('N=5k')
plt.xlabel('Noise level')
plt.ylabel('Run time (ms)')
plt.gca().ticklabel_format(style='sci',axis='y',scilimits=[-1,2])
plt.grid()
plt.savefig('./save_img/tim_noise_gpu_only_rigid.pdf')


plt.figure(figsize=(2,1.6))
plt.boxplot(r_err,sym='.')
plt.gca().set_xticklabels(labels=[f'{x:0.2f}' for x in noise_list],rotation=45)
plt.title('N=5k')
plt.xlabel('Noise level')
plt.ylabel('Rotation error (deg)')
plt.grid()
plt.savefig('./save_img/r_err_noise_gpu_only_rigid.pdf')

plt.figure(figsize=(2,1.6))
plt.boxplot(t_err,sym='.')
plt.gca().set_xticklabels(labels=[f'{x:0.2f}' for x in noise_list],rotation=45)
plt.title('N=5k')
plt.xlabel('Noise level')
plt.ylabel('Trans. error ')
plt.grid()
plt.savefig('./save_img/t_err_noise_gpu_only_rigid.pdf')

# ######===========这里是很高的outlier=============================

num=10000
noise_level=0.01

rate_len=10
rate_list=np.arange(rate_len)*0.01+0.9

repeat_time=500

tim=np.zeros((repeat_time,rate_len))
r_err=np.zeros((repeat_time,rate_len))
t_err=np.zeros((repeat_time,rate_len))

for ii in range(repeat_time):
    for jj in range(rate_len):

        [tim[ii,jj],r_err[ii,jj],t_err[ii,jj]]=test_rigid_pose(num,rate_list[jj],noise_level)
        print([ii,jj])

np.savetxt('./save_data/tim_high_outlier_gpu_only_rigid.txt',tim)
np.savetxt('./save_data/r_err_high_outlier_gpu_only_rigid.txt',r_err)
np.savetxt('./save_data/t_err_high_outlier_gpu_only_rigid.txt',t_err)

tim=np.loadtxt('./save_data/tim_high_outlier_gpu_only_rigid.txt')
r_err=np.loadtxt('./save_data/r_err_high_outlier_gpu_only_rigid.txt')
t_err=np.loadtxt('./save_data/t_err_high_outlier_gpu_only_rigid.txt')

plt.figure(figsize=(2,1.6))
plt.boxplot(tim,labels=[f'{int(x*100)}\%' for x in rate_list],showfliers=False)
plt.gca().set_xticklabels(labels=[f'{int(x*100)}\%' for x in rate_list],rotation=45)
plt.title('N=10k')
plt.xlabel('Outlier ratio')
plt.ylabel('Run time (ms)')
plt.gca().ticklabel_format(style='sci',axis='y',scilimits=[-1,2])
plt.grid()
plt.savefig('./save_img/tim_high_outlier_gpu_only_rigid.pdf')


plt.figure(figsize=(2,1.6))
plt.boxplot(r_err,labels=[f'{int(x*100)}\%' for x in rate_list],sym='.')
plt.gca().set_xticklabels(labels=[f'{int(x*100)}\%' for x in rate_list],rotation=45)
plt.title('N=10k')
plt.xlabel('Outlier ratio')
plt.ylabel('Rotation error (deg)')
plt.grid()
plt.savefig('./save_img/r_err_high_outlier_gpu_only_rigid.pdf')

plt.figure(figsize=(2,1.6))
plt.boxplot(t_err,labels=[f'{int(x*100)}\%' for x in rate_list],sym='.')
plt.gca().set_xticklabels(labels=[f'{int(x*100)}\%' for x in rate_list],rotation=45)
plt.title('N=10k')
plt.xlabel('Outlier ratio')
plt.ylabel('Trans. error')
plt.grid()
plt.savefig('./save_img/t_err_high_outlier_gpu_only_rigid.pdf')


# # ##============这里是一般的outlier=============================================

num=5000

noise_level=0.01

rate_len=10
rate_list=np.arange(rate_len)*0.1

repeat_time=500

tim=np.zeros((repeat_time,rate_len))
r_err=np.zeros((repeat_time,rate_len))
t_err=np.zeros((repeat_time,rate_len))

for ii in range(repeat_time):
    for jj in range(rate_len):

        [tim[ii,jj],r_err[ii,jj],t_err[ii,jj]]=test_rigid_pose(num,rate_list[jj],noise_level)
        print([ii,jj])

np.savetxt('./save_data/tim_outlier_gpu_only_rigid.txt',tim)
np.savetxt('./save_data/r_err_outlier_gpu_only_rigid.txt',r_err)
np.savetxt('./save_data/t_err_outlier_gpu_only_rigid.txt',t_err)

tim=np.loadtxt('./save_data/tim_outlier_gpu_only_rigid.txt')
r_err=np.loadtxt('./save_data/r_err_outlier_gpu_only_rigid.txt')
t_err=np.loadtxt('./save_data/t_err_outlier_gpu_only_rigid.txt')

plt.figure(figsize=(2,1.6))
plt.boxplot(tim,labels=[f'{int(x*100)}\%' for x in rate_list],showfliers=False)
plt.gca().set_xticklabels(labels=[f'{int(x*100)}\%' for x in rate_list],rotation=45)
plt.title('N=5k')
plt.xlabel('Outlier ratio')
plt.ylabel('Run time (ms)')
plt.gca().ticklabel_format(style='sci',axis='y',scilimits=[-1,2])
plt.grid()
plt.savefig('./save_img/tim_outlier_gpu_only_rigid.pdf')


plt.figure(figsize=(2,1.6))
plt.boxplot(r_err,labels=[f'{int(x*100)}\%' for x in rate_list],sym='.')
plt.gca().set_xticklabels(labels=[f'{int(x*100)}\%' for x in rate_list],rotation=45)
plt.title('N=5k')
plt.xlabel('Outlier ratio')
plt.ylabel('Rotation error (deg)')
plt.grid()
plt.savefig('./save_img/r_err_outlier_gpu_only_rigid.pdf')

plt.figure(figsize=(2,1.6))
plt.boxplot(t_err,labels=[f'{int(x*100)}\%' for x in rate_list],sym='.')
plt.gca().set_xticklabels(labels=[f'{int(x*100)}\%' for x in rate_list],rotation=45)
plt.title('N=5k')
plt.xlabel('Outlier ratio')
plt.ylabel('Trans. error')
plt.grid()
plt.savefig('./save_img/t_err_outlier_gpu_only_rigid.pdf')

# # ##==============这是测数量============================

num=1000

noise_level=0.01

num_len=10
num_list=(np.arange(num_len)+1)*num

repeat_time=500

tim=np.zeros((repeat_time,num_len))
r_err=np.zeros((repeat_time,num_len))
t_err=np.zeros((repeat_time,num_len))

rate=0.90

for ii in range(repeat_time):
    for jj in range(num_len):

        [tim[ii,jj],r_err[ii,jj],t_err[ii,jj]]=test_rigid_pose(num_list[jj],rate,noise_level)
        print([ii,jj])

np.savetxt('./save_data/tim_num_gpu_only_rigid.txt',tim)
np.savetxt('./save_data/r_err_num_gpu_only_rigid.txt',r_err)
np.savetxt('./save_data/t_err_num_gpu_only_rigid.txt',t_err)

tim=np.loadtxt('./save_data/tim_num_gpu_only_rigid.txt')
r_err=np.loadtxt('./save_data/r_err_num_gpu_only_rigid.txt')
t_err=np.loadtxt('./save_data/t_err_num_gpu_only_rigid.txt')

plt.figure(figsize=(2,1.6))
plt.boxplot(tim,labels=[f'{round(x/1000)}k' for x in num_list],showfliers=False)
plt.title('Outlier ratio (90\\%)')
plt.xlabel('Number of input')
plt.ylabel('Run time (ms)')
plt.gca().ticklabel_format(style='sci',axis='y',scilimits=[-1,2])
plt.gca().set_xticklabels(labels=[f'{round(x/1000)}k' for x in num_list],rotation=45)
plt.grid()
plt.savefig('./save_img/tim_num_gpu_only_rigid.pdf')


plt.figure(figsize=(2,1.6))
plt.boxplot(r_err,sym='.')
plt.title('Outlier ratio (90\%)')
plt.gca().set_xticklabels(labels=[f'{round(x/1000)}k' for x in num_list],rotation=45)
plt.xlabel('Number of input')
plt.ylabel('Rotation error (deg)')
plt.grid()
plt.savefig('./save_img/r_err_num_gpu_only_rigid.pdf')

plt.figure(figsize=(2,1.6))
plt.boxplot(t_err,sym='.')
plt.title('Outlier ratio (90\%)')
plt.gca().set_xticklabels(labels=[f'{round(x/1000)}k' for x in num_list],rotation=45)
plt.xlabel('Number of input')
plt.ylabel('trans. error')
plt.grid()
plt.savefig('./save_img/t_err_num_gpu_only_rigid.pdf')













