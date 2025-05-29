from rot_voting_cupy import *
import scienceplots
plt.style.use(['science','ieee'])

# import tikzplotlib
import numpy as np 
import matplotlib.pyplot as plt 
#============这里是很高的outlier=============================

num=100000

noise_level=0.01

rate_len=10
rate_list=np.arange(rate_len)*0.01+0.9

repeat_time=50

tim=np.zeros((repeat_time,rate_len))
err=np.zeros((repeat_time,rate_len))

for ii in range(repeat_time):
    for jj in range(rate_len):

        [tim[ii,jj],err[ii,jj]]=test_rot_voting(num,rate_list[jj],noise_level)
        print([ii,jj])

np.savetxt('./save_data/tim_high_outlier_gpu_only_R.txt',tim)
np.savetxt('./save_data/err_high_outlier_gpu_only_R.txt',err)


tim=np.loadtxt('./save_data/tim_high_outlier_gpu_only_R.txt')
err=np.loadtxt('./save_data/err_high_outlier_gpu_only_R.txt')

plt.figure(figsize=(2,1.6))
plt.boxplot(tim,tick_labels=[f'{x:0.2f}' for x in rate_list],showfliers=False)
plt.title('N=100K')
plt.xlabel('Outlier ratio')
plt.ylabel('Run time (ms)')
# plt.gca().ticklabel_format(style='sci',axis='y',scilimits=[-1,2],rotation=30)
plt.gca().set_xticklabels(labels=[f'{int(x*100)}%' for x in rate_list],rotation=45)
plt.grid()
plt.savefig('./save_img/tim_high_outlier_gpu_only_R.pdf')
# tikzplotlib.save("./save_img_tikz/tim_high_outlier_gpu_only_R.tex")


plt.figure(figsize=(2,1.6))
plt.boxplot(err,tick_labels=[f'{int(x*100)}%' for x in rate_list],sym='.')
plt.gca().set_xticklabels(labels=[f'{int(x*100)}%' for x in rate_list],rotation=45)
plt.title('N=100K')
plt.xlabel('Outlier ratio')
plt.ylabel('Rotation error (deg)')
plt.grid()
plt.savefig('./save_img/err_high_outlier_gpu_only_R.pdf')
# tikzplotlib.save("./save_img_tikz/err_high_outlier_gpu_only_R.tex")


# # ##============这里是一般的outlier=============================================

num=100000
blocks=360
each_num=360
noise_level=0.01

rate_len=10
rate_list=np.arange(rate_len)*0.1

repeat_time=50

tim=np.zeros((repeat_time,rate_len))
err=np.zeros((repeat_time,rate_len))

for ii in range(repeat_time):
    for jj in range(rate_len):

        [tim[ii,jj],err[ii,jj]]=test_rot_voting(num,rate_list[jj],noise_level)
        print([ii,jj])

np.savetxt('./save_data/tim_outlier_gpu_only_R.txt',tim)
np.savetxt('./save_data/err_outlier_gpu_only_R.txt',err)


tim=np.loadtxt('./save_data/tim_outlier_gpu_only_R.txt')
err=np.loadtxt('./save_data/err_outlier_gpu_only_R.txt')

plt.figure(figsize=(2,1.6))
plt.boxplot(tim,tick_labels=[f'{int(x*100)}\%' for x in rate_list],showfliers=False)
plt.title('N=100K')
plt.xlabel('Outlier ratio')
plt.ylabel('Run time (ms)')
plt.gca().set_xticklabels(labels=[f'{int(x*100)}\%' for x in rate_list],rotation=45)
plt.grid()
plt.savefig('./save_img/tim_outlier_gpu_only_R.pdf')
# tikzplotlib.save("./save_img_tikz/tim_outlier_gpu_only_R.tex")


plt.figure(figsize=(2,1.6))
plt.boxplot(err,tick_labels=[f'{int(x*100)}\%' for x in rate_list],sym='.')
plt.gca().set_xticklabels(labels=[f'{int(x*100)}\%' for x in rate_list],rotation=45)
plt.title('N=100K')
plt.xlabel('Outlier ratio')
plt.ylabel('Rotation error (deg)')
plt.grid()
plt.savefig('./save_img/err_outlier_gpu_only_R.pdf')
# tikzplotlib.save("./save_img_tikz/err_outlier_gpu_only_R.tex")

#==============这是测数量============================



num=100000
noise_level=0.01

num_len=10
num_list=(np.arange(num_len)+1)*num

repeat_time=200

tim=np.zeros((repeat_time,num_len))
err=np.zeros((repeat_time,num_len))

rate=0.99

for ii in range(repeat_time):
    for jj in range(num_len):

        [tim[ii,jj],err[ii,jj]]=test_rot_voting(num_list[jj],rate,noise_level)
        print([ii,jj])

np.savetxt('./save_data/tim_num_gpu_only_R.txt',tim)
np.savetxt('./save_data/err_num_gpu_only_R.txt',err)


tim=np.loadtxt('./save_data/tim_num_gpu_only_R.txt')
err=np.loadtxt('./save_data/err_num_gpu_only_R.txt')

plt.figure(figsize=(2,1.6))

plt.boxplot(tim,tick_labels=[f'{int(x)}' for x in num_list],showfliers=False)
plt.title('Outlier ratio (99%)')
plt.xlabel(r'Number of input($\times 10^5$)')
plt.ylabel('Run time (ms)')
plt.gca().set_xticklabels(labels=[f'{int(x/100000)}' for x in num_list])
plt.grid()
plt.savefig('./save_img/tim_num_gpu_only_R.pdf')
# tikzplotlib.save("./save_img_tikz/tim_num_gpu_only_R.tex")


plt.figure(figsize=(2,1.6))
plt.boxplot(err,sym='.')
plt.title('Outlier ratio (99%)')
plt.gca().set_xticklabels(labels=[f'{int(x/100000)}' for x in num_list])
plt.xlabel(r'Number of input($\times 10^5$)')
plt.ylabel('Rotation error (deg)')
plt.grid()
plt.savefig('./save_img/err_num_gpu_only_R.pdf')
# tikzplotlib.save("./save_img_tikz/err_num_gpu_only_R.tex")


# ######============这里是noise =============================

num=100000


noise_len=10
noise_list=np.arange(noise_len)*0.01+0.01
outlier_rate=0.9

repeat_time=50

tim=np.zeros((repeat_time,noise_len))
err=np.zeros((repeat_time,noise_len))

for ii in range(repeat_time):
    for jj in range(noise_len):

        [tim[ii,jj],err[ii,jj]]=test_rot_voting(num,outlier_rate,noise_list[jj])
        print([ii,jj])

np.savetxt('./save_data/tim_noise_gpu_only_R.txt',tim)
np.savetxt('./save_data/err_noise_gpu_only_R.txt',err)


tim=np.loadtxt('./save_data/tim_noise_gpu_only_R.txt')
err=np.loadtxt('./save_data/err_noise_gpu_only_R.txt')

plt.figure(figsize=(2,1.6))
plt.boxplot(tim,showfliers=False)
plt.gca().set_xticklabels(labels=[f'{x:.2f}' for x in noise_list],rotation=45)
plt.title('N=100K')
plt.xlabel('Noise level')
plt.ylabel('Run time (ms)')
# plt.gca().ticklabel_format(style='sci',axis='y',scilimits=[-1,2])
plt.grid()
plt.savefig('./save_img/tim_noise_gpu_only_R.pdf')
# tikzplotlib.save("./save_img_tikz/tim_noise_gpu_only_R.tex")


plt.figure(figsize=(2,1.6))
plt.boxplot(err,sym='.')
plt.gca().set_xticklabels(labels=[f'{x:.2f}' for x in noise_list],rotation=45)
plt.title('N=100K')
plt.xlabel('Noise level')
plt.ylabel('Rotation error (deg)')
plt.grid()
plt.savefig('./save_img/err_noise_gpu_only_R.pdf')
# tikzplotlib.save("./save_img_tikz/err_noise_gpu_only_R.tex")











