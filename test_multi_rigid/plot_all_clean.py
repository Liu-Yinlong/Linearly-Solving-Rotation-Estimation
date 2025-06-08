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
plt.rcParams['font.size']=7
import matplotlib
colors=matplotlib.colormaps['tab10']

num_model=np.arange(8)+2
num=200

fig=plt.figure(figsize=(8,2))

flierprops=dict(marker='.',  markersize=0.5,fillstyle='full',markerfacecolor=None)
boxprops = dict(linestyle='-', linewidth=0.5,color='lime')



tim_refine=np.loadtxt(f'./results/tim-{num}_refine.txt')
ave_err_r_refine=np.loadtxt(f'./results/r_err-{num}_refine.txt')
ave_err_t_refine=np.loadtxt(f'./results/t_err-{num}_refine.txt')


tim_5k=np.loadtxt(f'./results/tim-{num}_ransac_5k.txt')
ave_err_r_5k=np.loadtxt(f'./results/r_err-{num}_ransac_5k.txt')
ave_err_t_5k=np.loadtxt(f'./results/t_err-{num}_ransac_5k.txt')

tim_10k=np.loadtxt(f'./results/tim-{num}_ransac_10k.txt')
ave_err_r_10k=np.loadtxt(f'./results/r_err-{num}_ransac_10k.txt')
ave_err_t_10k=np.loadtxt(f'./results/t_err-{num}_ransac_10k.txt')

tim_multi_10k=np.loadtxt(f'./results/tim-{num}_multi_ransac_10k.txt')
ave_err_r_multi_10k=np.loadtxt(f'./results/r_err-{num}_multi_ransac_10k.txt')
ave_err_t_multi_10k=np.loadtxt(f'./results/t_err-{num}_multi_ransac_10k.txt')

tim_multi_5k=np.loadtxt(f'./results/tim-{num}_multi_ransac_5k.txt')
ave_err_r_multi_5k=np.loadtxt(f'./results/r_err-{num}_multi_ransac_5k.txt')
ave_err_t_multi_5k=np.loadtxt(f'./results/t_err-{num}_multi_ransac_5k.txt')


positions_1=np.arange(len(num_model))+2-0.36
positions_2=np.arange(len(num_model))+2-0.18
positions_3=np.arange(len(num_model))+2
positions_4=np.arange(len(num_model))+2+0.18
positions_5=np.arange(len(num_model))+2+0.36

positions=np.arange(len(num_model))+2

print(positions)

plt.subplot(131)


plt.bar(positions_1,np.median(tim_5k,axis=0),width=0.18,color=colors(1),zorder=2)
plt.bar(positions_2,np.median(tim_multi_5k,axis=0),width=0.18,color=colors(2),zorder=2)
plt.bar(positions_3,np.median(tim_10k,axis=0),width=0.18,color=colors(3),zorder=2)
plt.bar(positions_4,np.median(tim_multi_10k,axis=0),width=0.18,color=colors(4),zorder=2)

plt.bar(positions_5,np.median(tim_refine,axis=0),width=0.18,color=colors(5),zorder=2)

plt.xlabel(r'# model number')
plt.title('Median time (ms)')
plt.xlim([positions[0]-0.5,positions[-1]+0.5])
plt.grid(axis='y')
for p in positions[0:-1]:
    plt.axvline(p+0.5,ls='--',color='grey',lw=0.1)
plt.xticks(ticks=positions,labels=positions)




plt.subplot(132)

bp1=plt.boxplot(ave_err_r_5k,positions=positions_1,widths=0.15,
                zorder=2,patch_artist=True,flierprops=flierprops,boxprops=boxprops)
for f in bp1['boxes']:
    f.set_color(colors(1))
f.set_label('Sequential-RANSAC-5k')
bp1=plt.boxplot(ave_err_r_multi_5k,positions=positions_2,widths=0.15,zorder=2,patch_artist=True,flierprops=flierprops)
for f in bp1['boxes']:
    f.set_color(colors(2))
f.set_label('Multi-RANSAC-5k')
bp1=plt.boxplot(ave_err_r_10k,positions=positions_3,widths=0.15,zorder=2,patch_artist=True,flierprops=flierprops)
for f in bp1['boxes']:
    f.set_color(colors(3))
f.set_label('Sequential-RANSAC-10k')
bp1=plt.boxplot(ave_err_r_multi_10k,positions=positions_4,widths=0.15,zorder=2,patch_artist=True,flierprops=flierprops)
for f in bp1['boxes']:
    f.set_color(colors(4))
f.set_label('Multi-RANSAC-10k')
bp1=plt.boxplot(ave_err_r_refine,positions=positions_5,widths=0.15,zorder=2,patch_artist=True,flierprops=flierprops)
for f in bp1['boxes']:
    f.set_color(colors(5))
f.set_label('Ours')
plt.yscale('log')
plt.xticks(ticks=positions,labels=positions)
plt.title(r'Average rotation error (deg)')
plt.xlabel(r'# model number')
plt.xlim([positions[0]-0.5,positions[-1]+0.5])
plt.grid(axis='y')
for p in positions[0:-1]:
    plt.axvline(p+0.5,ls='--',color='grey',lw=0.1)


plt.subplot(133)

bp1=plt.boxplot(ave_err_t_5k,positions=positions_1,widths=0.15,zorder=2,patch_artist=True,flierprops=flierprops)
for f in bp1['boxes']:
    f.set_color(colors(1))
# f.set_label('Sequential-RANSAC-1k')
bp1=plt.boxplot(ave_err_t_multi_5k,positions=positions_2,widths=0.15,zorder=2,patch_artist=True,flierprops=flierprops)
for f in bp1['boxes']:
    f.set_color(colors(2))
# f.set_label('Multi-RANSAC-1k')
bp1=plt.boxplot(ave_err_t_10k,positions=positions_3,widths=0.15,zorder=2,patch_artist=True,flierprops=flierprops)
for f in bp1['boxes']:
    f.set_color(colors(3))
# f.set_label('Sequential-RANSAC-5k')
bp1=plt.boxplot(ave_err_t_multi_10k,positions=positions_4,widths=0.15,zorder=2,patch_artist=True,flierprops=flierprops)
for f in bp1['boxes']:
    f.set_color(colors(4))
# f.set_label('Multi-RANSAC-5k')
bp1=plt.boxplot(ave_err_t_refine,positions=positions_5,widths=0.15,zorder=2,patch_artist=True,flierprops=flierprops)
for f in bp1['boxes']:
    f.set_color(colors(5))
# f.set_label('Ours')
plt.yscale('log')
plt.xticks(ticks=positions,labels=positions)
plt.title(r'Average Translation error')
plt.xlabel(r'# model number')
plt.xlim([positions[0]-0.5,positions[-1]+0.5])
plt.grid(axis='y')
for p in positions[0:-1]:
    plt.axvline(p+0.5,ls='--',color='grey',lw=0.1)



fig.legend(loc='outside upper center',ncols=5,frameon=True,bbox_to_anchor=(0.5, 1.15))




plt.tight_layout(pad=0.1,h_pad=0.015,w_pad=0.5)
plt.savefig('multi-rigid-all-clean.pdf')
    











