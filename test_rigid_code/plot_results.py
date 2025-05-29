import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scienceplots
plt.style.use(['science','ieee','no-latex'])
matplotlib.rcParams['font.family']='sans'
plt.figure(figsize=(10,1.8*3))
# ######============这里是noise =============================

num=5000


noise_len=10
noise_list=np.arange(noise_len)*0.01+0.01
outlier_rate=0.9

repeat_time=500 



tim=np.loadtxt('./save_data/tim_noise_gpu_only_rigid.txt')
r_err=np.loadtxt('./save_data/r_err_noise_gpu_only_rigid.txt')
t_err=np.loadtxt('./save_data/t_err_noise_gpu_only_rigid.txt')

# plt.figure(figsize=(2,1.6))
plt.subplot(3,4,9)
plt.boxplot(tim,showfliers=False)
plt.gca().set_xticklabels(labels=[f'{x:0.2f}' for x in noise_list],rotation=45)
plt.title('N=5k')
plt.xlabel('Noise level')
plt.ylabel('Run time (ms)')
# plt.gca().ticklabel_format(style='sci',axis='y',scilimits=[-1,2])
plt.grid()
# plt.savefig('./save_img_/tim_noise_gpu_only_rigid.pdf')


# plt.figure(figsize=(2,1.6))
plt.subplot(3,4,1)
plt.boxplot(r_err,flierprops=dict(markersize=1))
plt.gca().set_xticklabels(labels=[f'{x:0.2f}' for x in noise_list],rotation=45)
plt.title('N=5k')
# plt.xlabel('Noise level')
plt.ylabel('Rotation error (deg)')
plt.grid()
# plt.savefig('./save_img_/r_err_noise_gpu_only_rigid.pdf')

# plt.figure(figsize=(2,1.6))
plt.subplot(3,4,5)
plt.boxplot(t_err,flierprops=dict(markersize=1))
plt.gca().set_xticklabels(labels=[f'{x:0.2f}' for x in noise_list],rotation=45)
plt.title('N=5k')
# plt.xlabel('Noise level')
plt.ylabel('Trans. error ')
plt.grid()
# plt.savefig('./save_img_/t_err_noise_gpu_only_rigid.pdf')

# ######===========这里是很高的outlier=============================

num=10000
noise_level=0.01

rate_len=10
rate_list=np.arange(rate_len)*0.01+0.9

repeat_time=500



tim=np.loadtxt('./save_data/tim_high_outlier_gpu_only_rigid.txt')
r_err=np.loadtxt('./save_data/r_err_high_outlier_gpu_only_rigid.txt')
t_err=np.loadtxt('./save_data/t_err_high_outlier_gpu_only_rigid.txt')

# plt.figure(figsize=(2,1.6))
plt.subplot(3,4,10)
plt.boxplot(tim,tick_labels=[f'{int(x*100)}%' for x in rate_list],showfliers=False)
plt.gca().set_xticklabels(labels=[f'{round(x*100)}%' for x in rate_list],rotation=45)
plt.title('N=10k')
plt.xlabel('Outlier ratio')
# plt.ylabel('Run time (ms)')
# plt.ylim([550,700])
plt.gca().ticklabel_format(style='sci',axis='y',scilimits=[-1,2])
plt.grid()
# plt.savefig('./save_img_/tim_high_outlier_gpu_only_rigid.pdf')


# plt.figure(figsize=(2,1.6))
plt.subplot(3,4,2)
plt.boxplot(r_err,tick_labels=[f'{int(x*100)}%' for x in rate_list],flierprops=dict(markersize=1))
plt.gca().set_xticklabels(labels=[f'{round(x*100)}%' for x in rate_list],rotation=45)
plt.title('N=10k')
# plt.xlabel('Outlier ratio')
# plt.ylabel('Rotation error (deg)')
plt.grid()
# plt.savefig('./save_img_/r_err_high_outlier_gpu_only_rigid.pdf')

# plt.figure(figsize=(2,1.6))
plt.subplot(3,4,6)
plt.boxplot(t_err,tick_labels=[f'{int(x*100)}%' for x in rate_list],flierprops=dict(markersize=1))
plt.gca().set_xticklabels(labels=[f'{round(x*100)}%' for x in rate_list],rotation=45)
plt.title('N=10k')
# plt.xlabel('Outlier ratio')
# plt.ylabel('Trans. error')
plt.grid()
# plt.savefig('./save_img_/t_err_high_outlier_gpu_only_rigid.pdf')


# ##============这里是一般的outlier=============================================

num=5000

noise_level=0.01

rate_len=10
rate_list=np.arange(rate_len)*0.1

repeat_time=500


tim=np.loadtxt('./save_data/tim_outlier_gpu_only_rigid.txt')
r_err=np.loadtxt('./save_data/r_err_outlier_gpu_only_rigid.txt')
t_err=np.loadtxt('./save_data/t_err_outlier_gpu_only_rigid.txt')

# plt.figure(figsize=(2,1.6))
plt.subplot(3,4,11)
plt.boxplot(tim,tick_labels=[f'{int(x*100)}%' for x in rate_list],showfliers=False)
plt.gca().set_xticklabels(labels=[f'{round(x*100)}%' for x in rate_list],rotation=45)
plt.title('N=5k')
plt.xlabel('Outlier ratio')
# plt.ylabel('Run time (ms)')
# plt.ylim([0,4500])
plt.gca().ticklabel_format(style='sci',axis='y',scilimits=[-1,2])
plt.grid()
# plt.savefig('./save_img_/tim_outlier_gpu_only_rigid.pdf')


# plt.figure(figsize=(2,1.6))
plt.subplot(3,4,3)
plt.boxplot(r_err,tick_labels=[f'{int(x*100)}%' for x in rate_list],flierprops=dict(markersize=1))
plt.gca().set_xticklabels(labels=[f'{round(x*100)}%' for x in rate_list],rotation=45)
plt.title('N=5k')
# plt.xlabel('Outlier ratio')
# plt.ylabel('Rotation error (deg)')
plt.grid()
# plt.savefig('./save_img_/r_err_outlier_gpu_only_rigid.pdf')

# plt.figure(figsize=(2,1.6))
plt.subplot(3,4,7)
plt.boxplot(t_err,tick_labels=[f'{round(x*100)}%' for x in rate_list],flierprops=dict(markersize=1))
plt.gca().set_xticklabels(labels=[f'{round(x*100)}%' for x in rate_list],rotation=45)
plt.title('N=5k')
# plt.xlabel('Outlier ratio')
# plt.ylabel('Trans. error')
plt.grid()
# plt.savefig('./save_img_/t_err_outlier_gpu_only_rigid.pdf')

# ##==============这是测数量============================

num=1000

noise_level=0.01

num_len=10
num_list=(np.arange(num_len)+1)*num

repeat_time=500


tim=np.loadtxt('./save_data/tim_num_gpu_only_rigid.txt')
r_err=np.loadtxt('./save_data/r_err_num_gpu_only_rigid.txt')
t_err=np.loadtxt('./save_data/t_err_num_gpu_only_rigid.txt')

# plt.figure(figsize=(2,1.6))
plt.subplot(3,4,12)
plt.boxplot(tim,tick_labels=[f'{round(x/1000)}k' for x in num_list],showfliers=False)
plt.title('Outlier ratio (90%)')
plt.xlabel('Number of input')
# plt.ylabel('Run time (ms)')
# plt.gca().ticklabel_format(style='sci',axis='y',scilimits=[-1,2])
plt.gca().set_xticklabels(labels=[f'{round(x/1000)}k' for x in num_list],rotation=45)
plt.grid()
# plt.savefig('./save_img_/tim_num_gpu_only_rigid.pdf')


# plt.figure(figsize=(2,1.6))
plt.subplot(3,4,4)
plt.boxplot(r_err,flierprops=dict(markersize=1))
plt.title('Outlier ratio (90%)')
plt.gca().set_xticklabels(labels=[f'{round(x/1000)}k' for x in num_list],rotation=45)
# plt.xlabel('Number of input')
# plt.ylabel('Rotation error (deg)')
plt.grid()
# plt.savefig('./save_img_/r_err_num_gpu_only_rigid.pdf')

# plt.figure(figsize=(2,1.6))
plt.subplot(3,4,8)
plt.boxplot(t_err,flierprops=dict(markersize=1))
plt.title('Outlier ratio (90%)')
plt.gca().set_xticklabels(labels=[f'{round(x/1000)}k' for x in num_list],rotation=45)
# plt.xlabel('Number of input')
# plt.ylabel('trans. error')
plt.grid()
# plt.savefig('./save_img_/t_err_num_gpu_only_rigid.pdf')

plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.savefig('./save_img/simu_all_rigid.pdf')












