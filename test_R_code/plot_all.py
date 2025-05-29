import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import matplotlib
plt.style.use(['science','no-latex','ieee'])
matplotlib.rcParams['font.family']='sans'
plt.figure(figsize=(10,3.8))


##-----------------------------------------------------
tim=np.loadtxt('./save_data/tim_high_outlier_gpu_only_R.txt')
err=np.loadtxt('./save_data/err_high_outlier_gpu_only_R.txt')

rate_len=10
rate_list=np.arange(rate_len)*0.01+0.9

plt.subplot(2,4,6)
plt.boxplot(tim,showfliers=False)
plt.title('N=$10^5$')
plt.xlabel('Outlier ratio')
# plt.ylabel('Run time (ms)')
plt.gca().ticklabel_format(style='sci',axis='y',scilimits=[-1,2])
plt.xticks(np.arange(rate_len)[0::2]+2,[f'{int(x*100)}%' for x in rate_list[1::2]])
plt.grid()

plt.subplot(2,4,2)
plt.boxplot(err,flierprops=dict(markersize=1))
plt.xticks(np.arange(rate_len)[0::2]+2,[f'{int(x*100)}%' for x in rate_list[1::2]])
plt.title('N=$10^5$')
plt.xlabel('Outlier ratio')
# plt.ylabel('Rotation error (deg)')
plt.grid()

##-----------------------------------------------------
rate_len=10
rate_list=np.arange(rate_len)*0.1

tim=np.loadtxt('./save_data/tim_outlier_gpu_only_R.txt')
err=np.loadtxt('./save_data/err_outlier_gpu_only_R.txt')

plt.subplot(2,4,5)
plt.boxplot(tim,showfliers=False)
plt.xticks(np.arange(rate_len)[0::2]+2,[f'{int(x*100)}%' for x in rate_list[1::2]])
plt.title('N=$10^5$')
plt.xlabel('Outlier ratio')
plt.ylabel('Run time (ms)')
plt.gca().ticklabel_format(style='sci',axis='y',scilimits=[-1,2])
plt.grid()



plt.subplot(2,4,1)
plt.boxplot(err,flierprops=dict(markersize=1))
plt.xticks(np.arange(rate_len)[0::2]+2,[f'{int(x*100)}%' for x in rate_list[1::2]])
plt.title('N=$10^5$')
plt.xlabel('Outlier ratio')
plt.ylabel('Rotation error (deg)')
plt.grid()

##-------------------------------------------------------------

num=100000
num_len=10
num_list=(np.arange(num_len)+1)*num
tim=np.loadtxt('./save_data/tim_num_gpu_only_R.txt')
err=np.loadtxt('./save_data/err_num_gpu_only_R.txt')
plt.subplot(2,4,7)
plt.boxplot(tim,showfliers=False)
plt.xticks(np.arange(num_len)[0::2]+2,labels=[f'{round(x/100000)}' for x in num_list[1::2]])
plt.title('Outlier ratio (99%)')
plt.xlabel(r'Number of input ($\times10^5$)')
# plt.ylabel('Run time (ms)')
# plt.gca().ticklabel_format(style='sci',axis='y',scilimits=[-1,2])
# plt.gca().set_xticklabels(labels=[f'{int(x/1000)}K' for x in num_list],rotation=30)
plt.grid()



plt.subplot(2,4,3)
plt.boxplot(err,flierprops=dict(markersize=1))
plt.title('Outlier ratio (99%)')
plt.xticks(np.arange(num_len)[0::2]+2,labels=[f'{int(x/100000)}' for x in num_list[1::2]])
plt.xlabel(r'Number of input ($\times10^5$)')
# plt.ylabel('Rotation error (deg)')
plt.grid()


####--------------------------------------------------------------------------------
noise_len=10
noise_list=np.arange(noise_len)*0.01+0.01
tim=np.loadtxt('./save_data/tim_noise_gpu_only_R.txt')
err=np.loadtxt('./save_data/err_noise_gpu_only_R.txt')
plt.subplot(2,4,8)
plt.boxplot(tim,showfliers=False)
# plt.gca().set_xticklabels(labels=[f'{x:0.3f}' for x in noise_list],rotation=30)
plt.xticks(np.arange(noise_len)[0::2]+2,labels=[f'{x:0.2f}' for x in noise_list[1::2]])
plt.title('N=$10^5$')
plt.xlabel('Noise level')
# plt.ylabel('Run time (ms)')
plt.gca().ticklabel_format(style='sci',axis='y',scilimits=[-1,2])
plt.grid()

plt.subplot(2,4,4)
plt.boxplot(err,flierprops=dict(markersize=1))
# plt.gca().set_xticklabels(labels=[f'{x:0.3f}' for x in noise_list],rotation=30)
plt.xticks(np.arange(noise_len)[0::2]+2,labels=[f'{x:0.2f}' for x in noise_list[1::2]])
plt.title('N=$10^5$')
plt.xlabel('Noise level')
# plt.ylabel('Rotation error (deg)')
plt.grid()


###----------------------------------------------------
plt.tight_layout(pad=0)
plt.savefig('./save_img/syn_rotation_all.pdf')