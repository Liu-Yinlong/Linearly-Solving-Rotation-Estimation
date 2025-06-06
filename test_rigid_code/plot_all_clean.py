import numpy as np 
import matplotlib.pyplot as plt 
from scipy.spatial.transform import Rotation
import scipy.linalg
import  time
import cupy as cp 
from cupyx.scipy.ndimage import maximum_filter
import scienceplots
np.set_printoptions(precision=3)  # 浮点数保留3位小数
plt.style.use(['ieee','science','no-latex'])
plt.rcParams['font.family']='Sans'

num_model=np.arange(8)+2
num=200

fig=plt.figure(figsize=(4.2,1.8))

flierprops=dict(marker='o',  markersize=1)
boxprops = dict(linestyle='-', linewidth=1)

tim=np.loadtxt(f'./results/tim-{num}.txt')
ave_err_r=np.loadtxt(f'./results/r_err-{num}.txt')
ave_err_t=np.loadtxt(f'./results/t_err-{num}.txt')

tim_refine=np.loadtxt(f'./results/tim-{num}_refine.txt')
ave_err_r_refine=np.loadtxt(f'./results/r_err-{num}_refine.txt')
ave_err_t_refine=np.loadtxt(f'./results/t_err-{num}_refine.txt')


tim_5k=np.loadtxt(f'./results/tim-{num}_ransac_5k.txt')
ave_err_r_5k=np.loadtxt(f'./results/r_err-{num}_ransac_5k.txt')
ave_err_t_5k=np.loadtxt(f'./results/t_err-{num}_ransac_5k.txt')

tim_10k=np.loadtxt(f'./results/tim-{num}_ransac_10k.txt')
ave_err_r_10k=np.loadtxt(f'./results/r_err-{num}_ransac_10k.txt')
ave_err_t_10k=np.loadtxt(f'./results/t_err-{num}_ransac_10k.txt')


positions_1=np.arange(len(num_model))+2-0.3
positions_2=np.arange(len(num_model))+2
positions_3=np.arange(len(num_model))+2+0.3

positions=np.arange(len(num_model))+2

print(positions)

plt.subplot(121)


plt.bar(positions_3,np.median(tim_refine,axis=0),width=0.3,color='royalblue',zorder=2)
plt.bar(positions_1,np.median(tim_5k,axis=0),width=0.3,color='orange',zorder=2)
plt.bar(positions_2,np.median(tim_10k,axis=0),width=0.3,color='green',zorder=2)

plt.xlabel(r'# model number')
plt.title('Median time (ms)')
# plt.ylim([0,2500])
plt.grid()
plt.xticks(ticks=positions,labels=positions)




plt.subplot(122)
s1=np.logical_and((ave_err_r_refine<2),(ave_err_t_refine)<0.02)
s2=np.logical_and((ave_err_r_5k<2),(ave_err_t_5k)<0.02)
s3=np.logical_and((ave_err_r_10k<2),(ave_err_t_10k)<0.02)


plt.bar(positions_1,np.sum(s2,axis=0),width=0.3,color='orange',label='RANSAC_5K',zorder=2)
plt.bar(positions_2,np.sum(s3,axis=0),width=0.3,color='green',label='RANSAC_10K',zorder=2)
plt.bar(positions_3,np.sum(s1,axis=0),width=0.3,color='royalblue',label='Ours',zorder=2)

plt.xticks(ticks=positions,labels=positions)
plt.title(r'Success rates (%)')
plt.xlabel(r'# model number')
plt.grid()
fig.legend(loc='outside upper center',ncols=3,frameon=True,bbox_to_anchor=(0.5, 1.2))


plt.tight_layout(pad=0.1,h_pad=0.015,w_pad=0.5)
plt.savefig('multi-rigid-all-clean.pdf')
    











