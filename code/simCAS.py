"""
# Authors: Chen Li, Xiaoyang Chen
# File name: utils.py
# Description: 
# Version: 1.0.0
"""

import numpy as np
from scipy.stats import multivariate_normal
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import pandas as pd
from sklearn.neighbors import KernelDensity
import os
from scipy import sparse
import scipy.io as sio
import scanpy as sc
from Bio import Phylo
from io import StringIO
import logging
from scipy.optimize import fsolve
import random
import threading
import scipy.stats as stats
from sklearn.mixture import GaussianMixture as GMM
from scipy.stats import logser

from scipy.special import rel_entr
from statsmodels.discrete.count_model import (ZeroInflatedNegativeBinomialP, ZeroInflatedPoisson,
                                              ZeroInflatedGeneralizedPoisson)
import statsmodels.api as sm
from scipy.stats import nbinom
from scipy.special import expit
import anndata
import scipy

k_dict,pi_dict={},{}

def cal_cell_var(adata):
    """
    calculate statistics cell variance and return a 1D array
    """
    if isinstance(adata.X,scipy.sparse._csr.csr_matrix):
        return np.array(np.var(adata.X.toarray(),axis=1)).ravel()
    else:
        return np.array(np.var(adata.X,axis=1)).ravel()
    
def cal_peak_var(adata):
    """
    calculate statistics peak variance and return a 1D array
    """
    if isinstance(adata.X,scipy.sparse._csr.csr_matrix):
        return np.array(np.var(adata.X.toarray(),axis=0)).ravel()
    else:
        return np.array(np.var(adata.X,axis=0)).ravel()
    
def cal_peak_nozero(adata):
    """
    calculate statistics peak non-zeros and return a 1D array
    """
    X=adata.X.copy()
    X[X>0]=1
    sparsity=np.sum(X,axis=0)
    return np.array(sparsity).ravel()

# library size
def cal_lib(adata):
    """
    calculate statistics library size and return a 1D array
    """
    return np.array(np.sum(adata.X,axis=1)).ravel()

def cal_pm(adata):
    """
    calculate statistics peak mean and return a 1D array
    """
    return (np.array(np.sum(adata.X,axis=0))/adata.X.shape[0]).ravel()

def cal_pl(adata):
    """
    calculate statistics peak length and return a 1D array
    """
    start=np.array([int(i.split('_')[1]) for i in adata.var.index])
    end=np.array([int(i.split('_')[2]) for i in adata.var.index])
    return (end-start).ravel()

def cal_nozero(adata):
    """
    calculate statistics cell non-zeros and return a 1D array
    """
    X=adata.X.copy()
    X[X>0]=1
    sparsity=np.sum(X,axis=1)
    return np.array(sparsity).ravel()

def cal_peak_count(adata):
    """
    calculate statistics peak summation and return a 1D array
    """
    return np.array(np.sum(adata.X,axis=0)).ravel()

def cal_spa(adata):
    """
    calculate statistics cell sparsity and return a 1D array
    """
    X=adata.X.copy()
    X[X>0]=1
    sparsity=np.sum(X,axis=1)/X.shape[1]
    return np.array(sparsity).ravel()

def cal_peak_spa(adata):
    """
    calculate statistics peak sparsity and return a 1D array
    """
    X=adata.X.copy()
    X[X>0]=1
    sparsity=np.sum(X,axis=0)/X.shape[0]
    return np.array(sparsity).ravel()

def Activation(X,method='sigmod',K=4,A=1):
    """
    Transform the negative values of parameter matrix to be positive.
    
    Parameter
    ----------
    
    Return
    ----------
    """
    if method=='exp_linear':
        if K==None:K=4
        exp_num=K
        k=np.exp(exp_num)
        X_act=X.copy()
        X_act[X_act>=exp_num]=k*X_act[X_act>=exp_num]+np.exp(exp_num)-exp_num*np.exp(exp_num)
        X_act[X_act<exp_num]=np.exp(X_act[X_act<exp_num])
        return X_act
    elif method=='sigmod':
        if K==None:
            K=2
            A=1
        return A/(1+K**(-1*X))
        
    else:
        raise ValueError('wrong activation method!')
        

def fix_seed(seed):
    """
    Seed all necessary random number generators.
    
    Parameter
    ----------
    
    Return
    ----------
    """
    if seed is None:
        seed = random.randint(1, 10000)
    # torch.set_num_threads(1)  # Suggested for issues with deadlocks, etc.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def Bernoulli_pm_correction(X_peak,param_pm): # X_peak:等待矫正的矩阵  param_pm:对应的采样得到的peak_mean
    # peak mean correction
    peak_p_list=[]
    for i in range(0,X_peak.shape[0]):
        peak_p=X_peak[i,:] # 单个peak对应的所有cell的值
        peak_mean=np.mean(peak_p) # 当前矩阵的peak mean
        peak_mean_ex=np.exp(param_pm[i])-1 # 期望的peak mean
        
        # 若期望的peak_mean都是0
        if peak_mean_ex==0 or peak_mean==0:
            peak_p_list.append(peak_p*peak_mean_ex)
            continue
        
        if np.max(peak_p)/peak_mean*peak_mean_ex>1:
            peak_p_sort=np.sort(peak_p)
            idx=len(peak_p_sort)-1
            while(1):
                weight=(len(peak_p)*peak_mean_ex+idx-len(peak_p))/(np.sum(peak_p_sort[0:idx])+1e-8)
                if peak_p_sort[idx-1]*weight<=1:
                    # print(idx)
                    break

                for idx_2 in range(idx,-1,-1):  # 找到*weight<1 的idx
                    if peak_p_sort[idx_2-1]*weight<=1:
                        # print(idx_2)
                        break
                    # 如果实在没有idx能够使得值*weight<1,此时就会一直循环，需要及时跳出循环
                    if idx_2<=1:
                        break
                idx=idx_2
                if idx_2<=1:
                    weight=(len(peak_p)*peak_mean_ex+idx-len(peak_p))/(np.sum(peak_p_sort[0:idx])+1e-8)
                    break
            peak_p=peak_p*weight
            peak_p[peak_p>1]=1
        else:
            peak_p=peak_p/peak_mean*peak_mean_ex
        peak_p_list.append(peak_p)
    peak_p_matrix=np.vstack(peak_p_list)
    return peak_p_matrix

def Bernoulli_lib_correction(X_peak,param_lib):
    peak_p_list=[]
    for i in range(X_peak.shape[1]):
        peak_p=X_peak[:,i]
        lib_size=np.sum(peak_p)
        lib_size_ex=np.exp(param_lib[i])-1
        
        # 若期望的library_size都是0
        if lib_size_ex==0 or lib_size==0:
            peak_p_list.append((peak_p*lib_size_ex).reshape(-1,1))
            continue

        if np.max(peak_p)/lib_size*lib_size_ex>1:
            peak_p_sort=np.sort(peak_p)
            idx=len(peak_p_sort)-1
            while(1):
                weight=(lib_size_ex+idx-len(peak_p))/(np.sum(peak_p_sort[0:idx])+1e-8)
                if peak_p_sort[idx-1]*weight<=1:
                    break
                for idx_2 in range(idx,-1,-1):
                    if peak_p_sort[idx_2-1]*weight<=1:
                        # print(idx_2)
                        break
                    # 如果实在没有idx能够使得值*weight<1,此时就会一直循环，需要及时跳出循环
                    if idx_2<=1:
                        break
                idx=idx_2
                if idx_2<=1:
                    weight=(len(peak_p)*peak_mean_ex+idx-len(peak_p))/(np.sum(peak_p_sort[0:idx])+1e-8)
                    break
            # 防止出现<1的部分全都是0
            if np.sum(peak_p_sort[0:idx])==0:
                peak_p[peak_p>1]=1
            else:
                peak_p=peak_p*weight
                peak_p[peak_p>1]=1
        else:
            peak_p=peak_p/lib_size*lib_size_ex
        peak_p_list.append(peak_p.reshape(-1,1))
    peak_p_matrix=np.hstack(peak_p_list)
    return peak_p_matrix

def Get_Effect(n_peak,n_cell_total,len_cell_embed,rand_seed,zero_prob,zero_set,effect_mean,effect_sd):
    # 生成peak effect和library size effect
    # np.random.seed(rand_seed)
    peak_effect=np.random.normal(effect_mean,effect_sd,(n_peak,len_cell_embed))
    lib_size_effect=np.random.normal(effect_mean,effect_sd,(1,len_cell_embed))
    
    # 対生成的effect vevtor进行置零
    if zero_set=='by_row':
        # 对于每个peak的effect进行相同概率的置零
        def set_zero(a,zero_prob=0.5):
            a[np.random.choice(len(a),replace=False,size=int(len(a)*zero_prob))]=0
            return a
        peak_effect=np.apply_along_axis(set_zero,1,peak_effect,zero_prob=zero_prob)

    if zero_set=='all':
        # 对于所有index选择进行置零
        indices = np.random.choice(peak_effect.shape[1]*peak_effect.shape[0], replace=False, size=int(peak_effect.shape[1]*peak_effect.shape[0]*zero_prob))
        peak_effect[np.unravel_index(indices, peak_effect.shape)] = 0 
        
    return peak_effect,lib_size_effect

def Get_Single_Embedding(n_cell_total,embed_mean_same,embed_sd_same,
                 n_embed_diff,n_embed_same):
    embed=np.random.normal(embed_mean_same,embed_sd_same,(n_embed_same+n_embed_diff,n_cell_total))
    index=['embedding_'+str(m+1) for m in range(n_embed_same+n_embed_diff)]
    columns=['single cluster' for m in  range(n_cell_total)]
    df=pd.DataFrame(embed,columns=columns,index=index)
    
    return df,columns

def Get_Discrete_Embedding(pops_name,min_popsize,tree_text,
                 n_cell_total,pops_size,
                 embed_mean_same,embed_sd_same,
                  embed_mean_diff,embed_sd_diff,
                 n_embed_diff,n_embed_same,rand_seed,min_pop):
    # np.random.seed(rand_seed)
    n_pop=len(pops_name)
    if(n_cell_total<min_popsize*n_pop):
        raise ValueError("The size of the smallest population is too big for the total number of cells")

    if not pops_size:
        if min_pop:# 若设定了最小pop的size，则其他pop将原来的细胞数目平均分配
            pop_size=np.floor((n_cell_total-min_popsize)/(len(pops_name)-1))
            left_over=n_cell_total-min_popsize-pop_size*(len(pops_name)-1)
            pop_name_size={} #每个pop对应的size
            for name in pops_name:
                if name==min_pop:
                    pop_name_size[name]=min_popsize
                else:
                    pop_name_size[name]=pop_size
            pop_name_size[pops_name[pops_name.index(min_pop)-1]]+=left_over
        else:# 未设置最小pop，直接将每个pop的cell数目均分
            pop_size=np.floor((n_cell_total)/(len(pops_name)))
            left_over=n_cell_total-pop_size*(len(pops_name))
            pop_name_size={}
            for name in pops_name:
                pop_name_size[name]=pop_size
            pop_name_size[pops_name[0]]+=left_over

    else:# 若直接对每个pop赋予size
        pop_name_size={}
        for (i,name) in enumerate(pops_name):
            pop_name_size[name]=pops_size[i]
    # 将float转化为int 
    for key,value in pop_name_size.items():
        pop_name_size[key]=int(value)

    #--------生成不同pop之间的协方差矩阵，这里需要在你的python环境中使用R包ape
    ape = importr('ape')
    phyla=ape.read_tree(text=tree_text)
    corr_matrix=np.array(ape.vcv_phylo(phyla,cor=True))

    #--------生成embed
        
    embed_same,embed_diff=[],[]
    #生成差异embedding特征对应的均值，保证不同的pop之间的相关性
    embed_diff_mean_mv = multivariate_normal.rvs(mean=[embed_mean_diff]*n_pop, cov=corr_matrix, size=n_embed_diff)
    for (j,name) in enumerate(pops_name):
        #生成每个pop对应的非差异embed部分
        embed_same_pop=np.random.normal(embed_mean_same,embed_sd_same,(n_embed_same,pop_name_size[name]))

        #生成每个pop对应的差异embed部分
        embed_diff_pop=[]
        for k in range(n_embed_diff):
            embed=np.random.normal(embed_diff_mean_mv[k,j],embed_sd_diff,(pop_name_size[name],))
            embed_diff_pop.append(embed)
        embed_diff_pop=np.vstack(embed_diff_pop)

        # 对每个pop差异/非差异embed进行汇总
        embed_same.append(embed_same_pop) # n_embed_same*pop_size
        embed_diff.append(embed_diff_pop) # n_embed_diff*pop_size

    # embed_param: len_cell_embed*n_cell_total
    embed_same=np.hstack(embed_same)
    embed_diff=np.hstack(embed_diff)
    embed_param=np.vstack([embed_same,embed_diff])

    columns=np.hstack([[name]*pop_name_size[name] for name in pops_name])
    index=['same_embedding_'+str(m+1) for m in range(n_embed_same)]+['diff_embedding_'+str(m+1) for m in range(n_embed_diff)]
    df=pd.DataFrame(embed_param,columns=columns,index=index)

    
    return df,columns


def Generate_Tree_Sd(branches,root,depth=0,anchor=0,rand_seed=0):# depth就是到根节点的深度;一个递归函数,用来获取细胞在每个branch上的位置以及enbedding
    # np.random.seed(rand_seed)
    
    start_nodes=[i.split('-')[0] for i in branches]
    
    df=pd.DataFrame({'branches':[],'cell_places':[],'embeddings':[]})
    for i in range(len(start_nodes)): 
        if root==start_nodes[i]:# 该节点对应的所有branch
            branch=branches[i]
            start,end,branch_len,n_cells=branch.split('-')[0],\
                    branch.split('-')[1],float(branch.split('-')[2]),int(branch.split('-')[3])
            interval=branch_len/(n_cells-1)#获取interval
            cell_places=[depth+interval*i for i in range(n_cells-1)]+[depth+branch_len]#以interval为间隔获取cell在branch上的位置
            
            # 获取单维所有细胞的embedding
            embeddings=np.array([0]+list(np.cumsum(np.random.normal(0,np.sqrt(interval),(n_cells-1)))))+anchor
            
            df_=pd.DataFrame({'branches':[branch]*len(cell_places),'cell_places':cell_places,'embeddings':embeddings})
            df=pd.concat([df,df_,Generate_Tree_Sd(branches,end,depth+branch_len,anchor=embeddings[-1])],axis=0)
    return df


def Get_Continuous_Embedding(tree_text,n_cell_total,
                 embed_mean_same,embed_sd_same,
                  embed_mean_diff,embed_sd_diff,
                 n_embed_diff,n_embed_same,rand_seed):
    # np.random.seed(rand_seed)
    # 构建tree
    tree = Phylo.read(StringIO(tree_text), "newick")
    
    # 获取不同的branch，形式为‘X-X-length’
    clades = [i for i in tree.find_clades()]
    branch_clades=[i for i in clades if i.branch_length]
    branches=[tree.get_path(i)[-2:] for i in branch_clades]
    branches=[branches[i][0].name+'-'+branches[i][1].name+'-'+str(branch_clades[i].branch_length) for i in range(len(branches))]
    
    # 获取所有branch的长度
    total_branch_len=sum([float(i.split('-')[2]) for i in branches])
    
    
    # 获取每个branch上的细胞数目（按照branch长度进行均分）
    n_branches_cell=[]
    for i in range(len(branches)):
        branch_len=float(branches[i].split('-')[2])
        n_cells=np.floor(n_cell_total*(branch_len/total_branch_len))
        n_branches_cell.append(n_cells)

    # 将偏置加到数目最多的分支上
    n_branches_cell[n_branches_cell.index(max(n_branches_cell))]=n_branches_cell[n_branches_cell.index(max(n_branches_cell))]+n_cell_total-sum(n_branches_cell)
    n_branches_cell=[int(i) for i in n_branches_cell]
    
    # 将细胞数目加入branch，最终branch格式：A-B-1.0-200
    branches=[branches[i]+'-'+str(n_branches_cell[i]) for i in range(len(branches))]
    
    # 获取root名字
    root=clades[1].name
    
    # 生成continuous的embedding
    embed_same=np.random.normal(embed_mean_same,embed_sd_same,(n_embed_same,n_cell_total))
    embed_diff=[]
    for i in range(n_embed_diff):
        df_continuous=Generate_Tree_Sd(branches,root,depth=0,anchor=embed_mean_diff,rand_seed=rand_seed+i)
        embed_diff.append(np.array(df_continuous['embeddings']))
    embed_diff=np.vstack(embed_diff)
    # print(embed_same.shape,embed_diff.shape)
    # print(branches)
    embed=np.vstack([embed_same,embed_diff])

    
    # 加上columns和index
    columns=list(df_continuous['branches'])
    index=['same_embedding_'+str(m+1) for m in range(n_embed_same)]+['diff_embedding_'+str(m+1) for m in range(n_embed_diff)]
    df=pd.DataFrame(embed,columns=columns,index=index)
    
    return df,columns


def zip_correction(i,simu_param_lib_i,lambdas_i,lambdas_sum_i,simu_param_nozero_i,n_peak):
    global k_dict,pi_dict
    # print(i)
    # if i%200==0:print(i)
    def solve_function(unsolved_value):
        k,pi=unsolved_value[0],unsolved_value[1]
        return [
            k*(1-pi)-simu_param_lib_i/(lambdas_sum_i),
            n_peak*pi+(1-pi)*np.sum(np.exp(-lambdas_i*k))-(n_peak-simu_param_nozero_i)
        ]

    solved=fsolve(solve_function,[3,0.5],maxfev=2000)
    k,pi=solved[0],solved[1]
    simu1=k*(1-pi)*(lambdas_sum_i)
    real1=simu_param_lib_i
    if abs(simu1-real1)/real1>0.1:
        solved=fsolve(solve_function,[20,0.5],maxfev=2000)
    k,pi=solved[0],solved[1]

    k_dict[i]=solved[0]
    pi_dict[i]=solved[1]

class zip_correction_thread(threading.Thread):
    def __init__(self,i,simu_param_lib_i,lambdas_i,lambdas_sum_i,simu_param_nozero_i,n_peak):
        super(zip_correction_thread, self).__init__()
        self.i  = i
        self.simu_param_lib_i  = simu_param_lib_i
        self.lambdas_i  = lambdas_i
        self.lambdas_sum_i  = lambdas_sum_i
        self.simu_param_nozero_i  = simu_param_nozero_i
        self.n_peak  = n_peak
        

    def run(self):
        zip_correction(self.i,self.simu_param_lib_i,self.lambdas_i,self.lambdas_sum_i,self.simu_param_nozero_i,self.n_peak)

        
def Get_Tree_Counts(peak_mean,lib_size,nozero,n_peak,n_cell_total,rand_seed,peak_effect,lib_size_effect,
                    embeds_peak,embeds_lib,correct_iter,distribution='Bernoulli',
                    activation='exp',bw_pm=1e-4,bw_lib=0.05,bw_nozero=0.05,real_param=True,two_embeds=True,K=None,A=None):
    # np.random.seed(rand_seed)
    if distribution=='Bernoulli' and np.max(np.exp(peak_mean)-1)>1:
        raise ValueError('you data may not be Bernoulli distribution!')
    
    if real_param: #如果直接使用真实参数，peak mean直接按照真实参数来，lib size抽样
        param_pm=np.sort(peak_mean,axis=0).ravel()
        param_lib=np.sort(np.random.choice(lib_size,size=n_cell_total),axis=0).ravel()
        param_nozero=np.sort(np.random.choice(nozero,size=n_cell_total),axis=0).ravel()
    else:
        # kde
        kde_pm = KernelDensity(kernel='gaussian', bandwidth=bw_pm).fit(peak_mean.reshape(-1,1))
        kde_lib = KernelDensity(kernel='gaussian', bandwidth=bw_lib).fit(lib_size.reshape(-1,1))
        kde_nozero = KernelDensity(kernel='gaussian', bandwidth=bw_nozero).fit(nozero.reshape(-1,1))

        # 从kde中采样并进行排序（从小到大）
        param_pm=kde_pm.sample(n_peak,random_state=rand_seed)
        param_lib=kde_lib.sample(n_cell_total,random_state=rand_seed)
        param_nozero=kde_nozero.sample(n_cell_total,random_state=rand_seed)

        param_pm=np.sort(param_pm,axis=0).ravel()
        param_lib=np.sort(param_lib,axis=0).ravel()
        param_nozero=np.sort(param_nozero,axis=0).ravel()



#         estimation_dis='one_logser' # 'NB'/'one_logser'/'gamma'/'zero_logser'
        
#         print('the estimation method is ',estimation_dis)
        
#         if estimation_dis=='gamma':
#             peak_mean_real = np.exp(peak_mean)-1
#             peak_mean_sqrt = np.sqrt(peak_mean_real)

#             fit_alpha, fit_loc, fit_beta = stats.gamma.fit(peak_mean_sqrt,floc=np.min(peak_mean_sqrt)-0.001)
#             peak_mean_sqrt_sample = stats.gamma.rvs(a=fit_alpha, loc=fit_loc, scale=fit_beta, size=n_peak, random_state=rand_seed)
#             param_pm = np.sort(peak_mean_sqrt_sample)
#             param_pm = np.log(param_pm**2+1)
#         elif estimation_dis=='zero_logser':
#             peak_count_simu=zero_logser(peak_count)
#             param_pm=np.log(peak_count_simu/n_cell_total+1)
#             param_pm=np.sort(param_pm)
#         elif estimation_dis=='one_logser':
#             peak_count_simu=one_logser(peak_count)
#             param_pm=np.log(peak_count_simu/n_cell_total+1)
#             param_pm=np.sort(param_pm)
#         elif estimation_dis=='zero_NB':
#             peak_count_simu=zero_NB(peak_count)
#             param_pm=np.log(peak_count_simu/n_cell_total+1)
#             param_pm=np.sort(param_pm)
#         elif estimation_dis=='NB':
#             peak_count_simu=NB(peak_count)
#             param_pm=np.log(peak_count_simu/n_cell_total+1)
#             param_pm=np.sort(param_pm)
#         elif estimation_dis=='ZIP':
#             peak_count_simu=ZIP(peak_count)
#             param_pm=np.log(peak_count_simu/n_cell_total+1)
#             param_pm=np.sort(param_pm)
            
#         elif estimation_dis=='ZINB':
#             peak_count_simu=ZINB(peak_count)
#             param_pm=np.log(peak_count_simu/n_cell_total+1)
#             param_pm=np.sort(param_pm)
            
#         else:
#             raise ValueError('wrong estimation distribution!')
        
#         #n,random_state = 2,2022
#         gmm_lz = GMM(2, random_state=rand_seed)
#         gmm_lz.fit(lib_size_log.reshape(-1,1))
#         # [sample[0] for sample in gmm.sample(1000)]
#         lib_size_log_sample = gmm_lz.sample(n_cell_total)[0].reshape(-1)
#         param_lib = np.sort(lib_size_log_sample)
        
#         non_zero_real = np.exp(nozero)-1
#         non_zero_log = np.log(non_zero_real)
#         gmm_nz = GMM(2, random_state=rand_seed)
#         gmm_nz.fit(non_zero_log.reshape(-1,1))
#         # [sample[0] for sample in gmm.sample(1000)]
#         non_zero_log_sample = gmm_nz.sample(n_cell_total)[0].reshape(-1)
#         param_nozero = np.log(np.exp(np.sort(non_zero_log_sample))+1)

    # 从模拟矩阵的参数顺序对应到采样的真实参数
    X_peak=np.dot(peak_effect,embeds_peak)# peak*cell
    X_peak=Activation(X_peak,method=activation,K=K,A=A) # 防止出现负值
    rank=np.arange(len(X_peak))[np.mean(X_peak,axis=1).argsort().argsort()]
    param_pm=param_pm[rank]

    if two_embeds:
        X_lib=np.dot(lib_size_effect,embeds_lib).ravel()
    else:
        X_lib=np.dot(lib_size_effect,embeds_peak).ravel()
    rank = np.arange(len(X_lib))[X_lib.argsort().argsort()]
    param_lib=param_lib[rank]
    param_nozero=param_nozero[rank]

    # 对参数进行修正
    # X_peak维度是peak*cell
    simu_param_peak=X_peak
    if distribution=='Poisson':
        for i in range(correct_iter):
            # print('correct_iter '+str(i+1))
            simu_param_peak=simu_param_peak/(np.sum(simu_param_peak,axis=1).reshape(-1,1))*((np.exp(param_pm)-1).reshape(-1,1))*simu_param_peak.shape[1]
            simu_param_peak=simu_param_peak/np.sum(simu_param_peak,axis=0).reshape(1,-1)*((np.exp(param_lib)-1).reshape(1,-1))
            
        simu_param_lib=np.exp(param_lib)-1
        simu_param_nozero=np.exp(param_nozero)-1
        #--------使用poisson分布生成ATAC
        lambdas=simu_param_peak
        # lambdas=simu_param_peak*(simu_param_lib.reshape(1,-1))
        
        # 对sparsity进行修正
        lambdas_sum=np.sum(lambdas,axis=0)
        
#         print("**********start ZIP correction...**********")
#         k_list,pi_list=[],[]
#         # 求解每个cell中lambda扩大的倍数和置零的比例
#         for i in range(n_cell_total):
#             iter_=i
#             # print(i)
#             def solve_function(unsolved_value):
#                 k,pi=unsolved_value[0],unsolved_value[1]
#                 return [
#                     k*(1-pi)-simu_param_lib[iter_]/(lambdas_sum[iter_]),
#                     n_peak*pi+(1-pi)*np.sum(np.exp(-lambdas[:,iter_]*k))-(n_peak-simu_param_nozero[iter_])
#                 ]

#             solved=fsolve(solve_function,[3,0.5],maxfev=2000)
#             k,pi=solved[0],solved[1]
#             simu1=k*(1-pi)*(lambdas_sum[iter_])
#             real1=simu_param_lib[iter_]
#             if abs(simu1-real1)/real1>0.1:
#                 print('=================================')
#                 print(i)
#                 print(simu1,real1)
#                 # print('=================================')
#                 solved=fsolve(solve_function,[20,0.5],maxfev=2000)
#             simu1=solved[0]*(1-solved[1])*(lambdas_sum[iter_])
#             real1=simu_param_lib[iter_]
#             if abs(simu1-real1)/real1>0.1:
#                 print(i)
#                 print(simu1,real1)
#                 print("=================================")
                
#             k_list.append(solved[0])
#             pi_list.append(solved[1])
#         # 对每个cell的lambda置零并扩大相应倍数
#         for i in range(n_cell_total):
#             if k_list[i]==3 or k_list[i]==20 or pi_list[i]<0:
#                 continue
#             a=lambdas[:,i]*k_list[i]
#             # print(i)
#             # print(k_list[i],pi_list[i])
#             # print("=============================")
#             # b=atac_counts[:,i]
#             a[np.random.choice(n_peak,replace=False,size=int(pi_list[i]*n_peak))]=0
#             lambdas[:,i]=a
#         print("**********ZIP correction finished!**********")
            
        print("**********start ZIP correction...**********")
        batch_size = 1000 # 并行数目，全局字典
        global k_dict,pi_dict
        for i in range(0,n_cell_total,batch_size):
            if i+batch_size<=n_cell_total:
                my_thread = [zip_correction_thread(j,simu_param_lib[j],lambdas[:,j],lambdas_sum[j],simu_param_nozero[j],n_peak) for j in range(i, i+batch_size)]
            else:
                my_thread = [zip_correction_thread(j,simu_param_lib[j],lambdas[:,j],lambdas_sum[j],simu_param_nozero[j],n_peak) for j in range(i, n_cell_total)]
            for thread_ in my_thread:
                thread_.start()
            for thread_ in my_thread:
                thread_.join()
        # 对每个cell的lambda置零并扩大相应倍数
        for i in range(n_cell_total):
            if k_dict[i]==3 or k_dict[i]==20 or pi_dict[i]<0 or k_dict[i]<0:
                continue
            a=lambdas[:,i]*k_dict[i]
            # b=atac_counts[:,i]
            a[np.random.choice(n_peak,replace=False,size=int(pi_dict[i]*n_peak))]=0
            lambdas[:,i]=a
            
        print("**********ZIP correction finished!**********")
            
        atac_counts=np.random.poisson(lambdas, lambdas.shape)
    elif distribution=='Bernoulli':
        for i in range(correct_iter):
            # print('correct_iter '+str(i+1))
            simu_param_peak=Bernoulli_pm_correction(simu_param_peak,param_pm)
            simu_param_peak=Bernoulli_lib_correction(simu_param_peak,param_lib)
        atac_counts=np.random.binomial(1,p=simu_param_peak,size=simu_param_peak.shape)
    
    return atac_counts


   
def kl_div(peak_count,peak_count_simu):
    # -------- K-L散度
    peak_count_combine=np.concatenate((peak_count,peak_count_simu))
    value=np.sort(np.unique(peak_count_combine))
    value_count_ori,value_count_simu=[],[]
    for value_ in value:
        value_count_ori.append(len(np.where(peak_count==value_)[0]))
        value_count_simu.append(len(np.where(peak_count_simu==value_)[0]))

    value_count_ori=np.array(value_count_ori)
    value_count_ori=value_count_ori/sum(value_count_ori)
    value_count_simu=np.array(value_count_simu)
    value_count_simu=value_count_simu/sum(value_count_simu)
    
    epsilon = 0.00001
    value_count_ori+=epsilon
    value_count_simu+=epsilon
    
    # print('KL divergence:',sum(rel_entr(value_count_ori, value_count_simu)))
    return sum(rel_entr(value_count_ori, value_count_simu))
    
def zero_logser(peak_count):
    peak_count_new=np.delete(peak_count,np.where(peak_count == 0))
    zero_prob_=len(np.where(peak_count == 0)[0])/len(peak_count)
    def solve_function(unsolved_value):
        p=unsolved_value[0]
        return [
            -1*p/(np.log(1-p)*(1-p))-np.mean(peak_count_new)
        ]

    solved=fsolve(solve_function,[0.995],maxfev=2000)
    p=solved[0]
    # print(-1*p/(np.log(1-p)*(1-p)),np.mean(peak_count_new))
    peak_count_simu=logser.rvs(p,size=len(peak_count))*\
        stats.bernoulli.rvs(p = 1-zero_prob_, size = len(peak_count)) 
    
    return peak_count_simu

def one_logser(peak_count):
    zero_prob_=len(np.where(peak_count == 0)[0])/len(peak_count)
    one_prob=len(np.where(peak_count == 1)[0])/len(peak_count)
    peak_count_new=np.delete(peak_count,np.where(peak_count == 0))
    peak_count_new=np.delete(peak_count_new,np.where(peak_count_new == 1))-1
    # 固定0、1的概率
    idx_all=range(len(peak_count))
    idx_zero=np.random.choice(idx_all,replace=False,size=int(len(peak_count)*(zero_prob_)))
    idx_one=np.random.choice(np.delete(idx_all,idx_zero),replace=False,size=int(len(peak_count)*(one_prob)))

    def solve_function(unsolved_value):
        p=unsolved_value[0]
        return [
            -1*p/(np.log(1-p)*(1-p))-np.mean(peak_count_new)
        ]

    solved=fsolve(solve_function,[0.995],maxfev=2000)
    p=solved[0]
    # print(-1*p/(np.log(1-p)*(1-p)),np.mean(peak_count_new))

    peak_count_simu=logser.rvs(p,size=len(peak_count))+1
    peak_count_simu[idx_zero]=0
    peak_count_simu[idx_one]=1
    
    return peak_count_simu

def ZINB(peak_count):
    model_zinb = ZeroInflatedNegativeBinomialP(peak_count, np.ones_like(peak_count), p=1)
    res_zinb = model_zinb.fit(method='bfgs', maxiter=5000, maxfun=5000)
    mu = np.exp(res_zinb.params[1])
    alpha = res_zinb.params[2]
    pi = expit(res_zinb.params[0])

    p=1/(1+alpha)
    n=mu*p/(1-p)

    peak_count_simu=(nbinom.rvs(n,p,size=len(peak_count)))*\
        stats.bernoulli.rvs(p = 1-pi, size = len(peak_count))
    
    return peak_count_simu

def zero_NB(peak_count):
    zero_prob=len(np.where(peak_count == 0)[0])/len(peak_count)
    peak_count_new=np.delete(peak_count,np.where(peak_count == 0))
    res=sm.NegativeBinomial(peak_count_new-1, np.ones_like(peak_count_new)).fit(start_params=[1,1])
    mu=np.exp(res.params[0])
    p=1/(1+mu*res.params[1])
    n=mu*p/(1-p)

    peak_count_simu=(nbinom.rvs(n,p,size=len(peak_count))+1)*\
        stats.bernoulli.rvs(p = 1-zero_prob, size = len(peak_count))
    
    return peak_count_simu
    
def NB(peak_count):
    res=sm.NegativeBinomial(peak_count, np.ones_like(peak_count)).fit(start_params=[1,1])
    mu=np.exp(res.params[0])
    p=1/(1+mu*res.params[1])
    n=mu*p/(1-p)

    peak_count_simu=nbinom.rvs(n,p,size=len(peak_count))
    
    return peak_count_simu

def ZIP(peak_count):
    zip_model = ZeroInflatedPoisson(endog = peak_count, exog= np.ones_like(peak_count)) 
    zip_res = zip_model.fit()
    mu=zip_res.params[1]
    pi = expit(zip_res.params[0])
    peak_count_simu = stats.bernoulli.rvs(p = 1-pi, size = len(peak_count))*\
            stats.poisson.rvs(mu = mu, size = len(peak_count))
    
    return peak_count_simu
 
    
def Get_Celltype_Counts(adata_part,two_embeds,embed_mean_same,embed_sd_same,len_cell_embed,effect_mean,effect_sd,
                 n_embed_diff,n_embed_same,correct_iter=10,lib_simu='real',n_cell_total=None,
                       distribution='Poisson',activation='sigmod'
                       ,bw_pm=1e-4,bw_lib=0.05,bw_nozero=0.05,rand_seed=0,zero_prob=0.5,zero_set='all',K=None,A=None,stat_estimation='one_logser',cell_scale=1):# 如果lib_simu为‘estimate’则需要提供对应的n_cell_total
    
    # np.random.seed(rand_seed)
    # 计算真实参数
    peak_mean=np.log(cal_pm(adata_part)+1)
    lib_size=np.log(cal_lib(adata_part)+1)
    nozero=np.log(cal_nozero(adata_part)+1)
    peak_count=cal_peak_count(adata_part)
    
    if distribution=='Bernoulli' and np.max(np.exp(peak_mean)-1)>1:
        raise ValueError('you data may not be Bernoulli distribution!')
    
    n_peak         =len(peak_mean)
    n_cell_total   =len(lib_size) #总共的细胞数目
    if lib_simu=='real':
        # param_lib=lib_size
        param_pm=np.sort(peak_mean,axis=0).ravel()
        param_lib=np.sort(np.random.choice(lib_size,size=n_cell_total),axis=0).ravel()
        param_nozero=np.sort(np.random.choice(nozero,size=n_cell_total),axis=0).ravel()
    elif lib_simu=='estimate':
        # kde_lib = KernelDensity(kernel='gaussian', bandwidth=bw_lib).fit(lib_size.reshape(-1,1))
        # param_lib=kde_lib.sample(n_cell_total,random_state=rand_seed)
        # param_lib=np.sort(param_lib)
        
        estimation_dis=stat_estimation # 'NB'/'one_logser'/'gamma'/'zero_logser'
        
        # print('the estimation method is ',estimation_dis)
        
        if estimation_dis=='gamma':
            peak_mean_real = np.exp(peak_mean)-1
            peak_mean_sqrt = np.sqrt(peak_mean_real)

            fit_alpha, fit_loc, fit_beta = stats.gamma.fit(peak_mean_sqrt,floc=np.min(peak_mean_sqrt)-0.001)
            peak_mean_sqrt_sample = stats.gamma.rvs(a=fit_alpha, loc=fit_loc, scale=fit_beta, size=n_peak, random_state=rand_seed)
            param_pm = np.sort(peak_mean_sqrt_sample)
            param_pm = np.log(param_pm**2+1)
        elif estimation_dis=='zero_logser':
            peak_count_simu=zero_logser(peak_count)
            param_pm=np.log(peak_count_simu/n_cell_total+1)
            param_pm=np.sort(param_pm)
        elif estimation_dis=='one_logser':
            peak_count_simu=one_logser(peak_count)
            param_pm=np.log(peak_count_simu/n_cell_total+1)
            param_pm=np.sort(param_pm)
        elif estimation_dis=='zero_NB':
            peak_count_simu=zero_NB(peak_count)
            param_pm=np.log(peak_count_simu/n_cell_total+1)
            param_pm=np.sort(param_pm)
        elif estimation_dis=='NB':
            peak_count_simu=NB(peak_count)
            param_pm=np.log(peak_count_simu/n_cell_total+1)
            param_pm=np.sort(param_pm)
        elif estimation_dis=='ZIP':
            peak_count_simu=ZIP(peak_count)
            param_pm=np.log(peak_count_simu/n_cell_total+1)
            param_pm=np.sort(param_pm)
            
        elif estimation_dis=='ZINB':
            peak_count_simu=ZINB(peak_count)
            param_pm=np.log(peak_count_simu/n_cell_total+1)
            param_pm=np.sort(param_pm)
            
        else:
            raise ValueError('wrong estimation distribution!')
            
        n_cell_total   =int(len(lib_size)*cell_scale)
        
        lib_size_real = np.exp(lib_size)-1
        lib_size_log = np.log(lib_size_real)
        
        #n,random_state = 2,2022
        gmm_lz = GMM(2, random_state=rand_seed)
        gmm_lz.fit(lib_size_log.reshape(-1,1))
        # [sample[0] for sample in gmm.sample(1000)]
        lib_size_log_sample = gmm_lz.sample(n_cell_total)[0].reshape(-1)
        param_lib = np.sort(lib_size_log_sample)
        
        non_zero_real = np.exp(nozero)-1
        non_zero_log = np.log(non_zero_real)
        gmm_nz = GMM(2, random_state=rand_seed)
        gmm_nz.fit(non_zero_log.reshape(-1,1))
        # [sample[0] for sample in gmm.sample(1000)]
        non_zero_log_sample = gmm_nz.sample(n_cell_total)[0].reshape(-1)
        param_nozero = np.log(np.exp(np.sort(non_zero_log_sample))+1)

    param_pm=param_pm[peak_mean.argsort().argsort()]
    # param_pm=np.sort(peak_mean)
    # origin_peak=np.arange(len(peak_mean))[peak_mean.argsort()]#记录实际peak的位置，保证最后输出的与输入peak含义一致

    # 生成effect和embedding
    peak_effect,lib_size_effect=Get_Effect(n_peak,n_cell_total,
                    len_cell_embed,rand_seed,zero_prob,zero_set,effect_mean,effect_sd)

    # if simu_type=='single':
    embeds_param={}
    embeds_param['peak'],meta=Get_Single_Embedding(n_cell_total,embed_mean_same,embed_sd_same,
                 n_embed_diff,n_embed_same)
    embeds_param['lib_size'],meta=Get_Single_Embedding(n_cell_total,embed_mean_same,embed_sd_same,
                 n_embed_diff,n_embed_same)


    # 从模拟矩阵的参数顺序对应到采样的真实参数
    X_peak=np.dot(peak_effect,embeds_param['peak'].values)# peak*cell
    X_peak=Activation(X_peak,method=activation,K=K,A=A)
    # rank=np.arange(len(X_peak))[np.mean(X_peak,axis=1).argsort().argsort()]
    # param_pm=param_pm[rank]
    # origin_peak=origin_peak[rank]

    if two_embeds:
        X_lib=np.dot(lib_size_effect,embeds_param['lib_size'].values).ravel()
    else:
        X_lib=np.dot(lib_size_effect,embeds_param['peak'].values).ravel()
    rank = np.arange(len(X_lib))[X_lib.argsort().argsort()]
    param_lib=param_lib[rank]
    param_nozero=param_nozero[rank]
    

    # 对参数进行修正
    # X_peak维度是peak*cell
    simu_param_peak=X_peak
    if distribution=='Poisson':
        for i in range(correct_iter):
            # print('correct_iter '+str(i+1))
            simu_param_peak=simu_param_peak/(np.sum(simu_param_peak,axis=0).reshape(1,-1)+1e-8)*((np.exp(param_lib)-1).reshape(1,-1))   # 分母加一个很小的数防止nan
            simu_param_peak=simu_param_peak/(np.sum(simu_param_peak,axis=1).reshape(-1,1)+1e-8)*((np.exp(param_pm)-1).reshape(-1,1))*simu_param_peak.shape[1]
            # simu_param_peak=simu_param_peak/(np.sum(simu_param_peak,axis=0).reshape(1,-1)+1e-8)*((np.exp(param_lib)-1).reshape(1,-1))   # 分母加一个很小的数防止nan
            
        simu_param_lib=np.exp(param_lib)-1
        simu_param_nozero=np.exp(param_nozero)-1
        simu_param_pm=np.exp(param_pm)-1
        #--------使用poisson分布生成ATAC
        lambdas=simu_param_peak
        # lambdas=lambdas[origin_peak.argsort(),:] #保证peak与输入peak一致
 
        atac_counts=np.random.poisson(lambdas, lambdas.shape)
    elif distribution=='Bernoulli':
        for i in range(correct_iter):
            # print('correct_iter '+str(i+1))
            simu_param_peak=Bernoulli_pm_correction(simu_param_peak,param_pm)
            simu_param_peak=Bernoulli_lib_correction(simu_param_peak,param_lib)
        atac_counts=np.random.binomial(1,p=simu_param_peak,size=simu_param_peak.shape)
        
        lambdas,simu_param_nozero,simu_param_lib,simu_param_pm=None,None,None,None
    
    return atac_counts,embeds_param['peak'].values,embeds_param['lib_size'].values,lambdas,simu_param_nozero,simu_param_lib,simu_param_pm

def simCAS_generate(peak_mean=None,lib_size=None,nozero=None,n_peak=1e5,n_cell_total=1500,rand_seed=2022,zero_prob=0.5,zero_set='all',effect_mean=0,effect_sd=1,
                   min_popsize=300,min_pop=None,tree_text=None,pops_name=None,pops_size=None,
                   embed_mean_same=1,embed_sd_same=0.5,embed_mean_diff=1,embed_sd_diff=0.5,
                   len_cell_embed=12,n_embed_diff=10,n_embed_same=2,simu_type='discrete',correct_iter=2,activation='exp_linear',
                   two_embeds=True,adata_dir=None,lib_simu='estimate',distribution='Poisson',bw_pm=1e-4,bw_lib=0.05,bw_nozero=0.05,real_param=False,K=None,A=None,stat_estimation='one_logser',cell_scale=1.0):
    """
    generate scCAS data with three modes: pseudo-cell-type mode, discrete mode, continuous mode.
    
    Parameter
    ----------
    peak_mean: 1D numpy array, default=None
        Real peak mean of scCAS data. Used for statistical estimation.
    lib_size: 1D numpy array, default=None
        Real library size of scCAS data. Used for statistical estimation.
    nozero: 1D numpy array, default=None
        Real non-zeros of scCAS data. Used for statistical estimation.
    n_peak: int, default=1e5
        Number of peaks in the synthetic data.
    n_cell_total: int, default=1500
        Number of simulating cells.
    rand_seed: int, default=2022
        Random seed for generation.
    zero_prob: float, default=0.5
        The probability of zeros in PEM.
    zero_set: str, default='all'
        How to set the PEM values to zero.
        1.'all': set the PEM values to zero with probability zero_prob for the whole PEM.
        2.'by_row': set the PEM values to zero with probability zero_prob for each row (peak) of PEM.
    effect_mean: float, default=0.0
        Mean of the Gaussian distribution, from which the PEM values are sampled.
    effect_sd: float, default=1.0
        Standard deviation of the Gaussian distribution, from which the PEM values are sampled.
    min_popsize: int, default=300
        The cell number of the minimal population set in the discrete mode. The number should be less than n_cell_total.
    min_pop: str, default=None
        The name of the minimal population. This should be contained pops_name
    tree_text: str, default=None
        A string of Newick format. In discrete mode this is used to define the covariance matrix of different populations. In continuous mode this is used to provide the differentiation trajectory.
    pops_name: list, default=None
        A list of defined names of populations.
    pops_size: list, default=None
        The number of cells of corresponding populations in the pops_name.
    embed_mean_same: float, default=1.0
        Mean of the Gaussian distritbution, from which the homogeneous CEM values are sampled.
    embed_sd_same: float, default=0.5
        Standard deviation of the Gaussian distritbution, from which the homogeneous CEM values are sampled.
    embed_mean_diff: float, default=1.0
        Mean of the Gaussian distritbution, with which the heterogeneous CEM values are generated.
    embed_sd_diff: float, default=0.5
        Standard deviation of the Gaussian distritbution, with which the heterogeneous CEM values are generated.
    len_cell_embed: int, default=12
        The number of the total embedding dimensions.
    n_embed_diff: int, default=10
        The number of the heterogeneous embedding dimensions.
    simu_type: str, default='cell_type'
        1.'cell_type': Pseudo-cell-type mode. Simulate data resembling real data.
        2.'discrete': Simulate cells with discrete populations.
        3.'continuous': Simulate cells with continuous trajectories.
    correct_iter: int, default=2
        The iterations of correction.
    activation: str, default='exp_linear'
        Choose the activation function to convert parameter matrix values to positive.
        1.'exp_linear': Used in the discrete mode or continuous mode.
        2.'sigmod': Used in the pseudo-cell-type mode.
        3.'exp': Used for biological batch effect generation.
    adata_dir: str, default=None
        The directory of real scCAS anndata.
    distribution: str, default='Poisson'
        Choose the distribution of data.
        1.'Poisson': for count data.
        2.'Bernoulli': for binary data.
    bw_pm: float, default=1e-4
        Band width of the kernel density estimation for peak mean.
    bw_lib: float, default=0.05
        Band width of the kernel density estimation for library size.
    bw_nozero: float, default=0.05
        Band width of the kernel density estimation for non-zero.
    K: float, default=None
        Adjust the slope of the activation function.
    A: float, default=None
        Adjust the slope of the activation function.
    stat_estimation: str, default='one_logser'
        Different discrete distributions to fit peak summation.
        1. 'zero_logser': a variant of logarithmic distribution.
        2. 'one_logser': a variant of logarithmic distribution.
        3. 'ZIP' : zero-inflated Poisson distribution.
        4. 'NB': Negative Binomial distribution.
        4. 'zero_NB': a variant of NB distribution.
        5. 'ZINB': zero-inflated Negative Binomial distribution. 
    cell_scale: float, default=1.0
        when conduct pseudo-celltype simulating mode, cell_scale is the magnification of the original cell number.
    
    
    Return
    ----------
    adata_final: anndata
        The simulated scCAS data with anndata format. The cell type information is in the observation.
    """
    n_embed_same   =len_cell_embed-n_embed_diff
    fix_seed(rand_seed)
    # 生成effect和embedding
    print("**********start generate effect vector...**********")
    peak_effect,lib_size_effect=Get_Effect(n_peak,n_cell_total,
                    len_cell_embed,rand_seed,zero_prob,zero_set,effect_mean,effect_sd)
    print("**********generate effect finished!**********")



    print("**********start generate cell embedding...**********")
    print("simulation type is {0}".format(simu_type))
    if simu_type=='discrete':
        # 重复两次获得两个矩阵，后续使用参数two_embeds决定是用两个矩阵还是用一个
        embeds_peak,meta=Get_Discrete_Embedding(pops_name,min_popsize,tree_text,
                     n_cell_total,pops_size,
                     embed_mean_same,embed_sd_same,
                      embed_mean_diff,embed_sd_diff,
                     n_embed_diff,n_embed_same,rand_seed,min_pop)
        embeds_lib,meta=Get_Discrete_Embedding(pops_name,min_popsize,tree_text,
                     n_cell_total,pops_size,
                     embed_mean_same,embed_sd_same,
                      embed_mean_diff,embed_sd_diff,
                     n_embed_diff,n_embed_same,rand_seed+1,min_pop)
        embeds_peak,embeds_lib=embeds_peak.values,embeds_lib.values
        print("**********generate cell embedding finished**********")
        # 获得count
        atac_counts=Get_Tree_Counts(peak_mean,lib_size,nozero,n_peak,n_cell_total,rand_seed,peak_effect,lib_size_effect,
                        embeds_peak,embeds_lib,correct_iter,distribution,activation,bw_pm,bw_lib,bw_nozero,
                                    real_param,two_embeds,K,A)
        print("**********generate counts finshed!**********")

    elif simu_type=='continuous':
        embeds_param={}
        embeds_peak,meta=Get_Continuous_Embedding(tree_text,n_cell_total,
                     embed_mean_same,embed_sd_same,
                      embed_mean_diff,embed_sd_diff,
                     n_embed_diff,n_embed_same,rand_seed)
        embeds_lib,meta=Get_Continuous_Embedding(tree_text,n_cell_total,
                     embed_mean_same,embed_sd_same,
                      embed_mean_diff,embed_sd_diff,
                     n_embed_diff,n_embed_same,rand_seed+1)
        embeds_peak,embeds_lib=embeds_peak.values,embeds_lib.values
        print("**********generate cell embedding finished**********")

        print("**********start generate counts...**********")
        atac_counts=Get_Tree_Counts(peak_mean,lib_size,nozero,n_peak,n_cell_total,rand_seed,peak_effect,lib_size_effect,
                        embeds_peak,embeds_lib,correct_iter,distribution,activation,bw_pm,bw_lib,bw_nozero,
                                   real_param,K,A)
        print("**********generate counts finshed!**********")

    elif simu_type=='single':
        embeds_param={}
        embeds_peak,meta=Get_Single_Embedding(n_cell_total,embed_mean_same,embed_sd_same,
                     n_embed_diff,n_embed_same)
        embeds_lib,meta=Get_Single_Embedding(n_cell_total,embed_mean_same,embed_sd_same,
                     n_embed_diff,n_embed_same)
        embeds_peak,embeds_lib=embeds_peak.values,embeds_lib.values
        print("**********generate cell embedding finished!**********")


        print("**********start generate counts...**********")
        atac_counts=Get_Tree_Counts(peak_mean,lib_size,nozero,n_peak,n_cell_total,rand_seed,
                        embeds_peak,embeds_lib,correct_iter,distribution,activation,bw_pm,bw_lib,bw_nozero,K,A)
        print("**********generate counts finshed!**********")

    elif simu_type=='cell_type':
        adata=sc.read_h5ad(adata_dir)
        counts_list,celltype_list,embed_peak_list,embed_lib_list=[],[],[],[]
        lambdas_list,simu_param_nozero_list,simu_param_lib_list,simu_param_pm_list=[],[],[],[]#新加的list用来重新对lambdas进行spasity的修正
        celltypes=np.unique(adata.obs.celltype)
        for i in range(len(celltypes)):
        # 可以分为直接从真实数据中进行采样或是从核密度估计中采样特定细胞数目，先做直接从真实数据中采样的结果
            # print(celltypes[i])
            print("simulating cell type: {}...".format(celltypes[i]))
            adata_part=adata[adata.obs.celltype==celltypes[i],:]

            # 对每个celltype单独进行仿真
            counts,embed_peak,embed_lib,lambdas,simu_param_nozero,simu_param_lib,simu_param_pm=Get_Celltype_Counts(adata_part,two_embeds,
                                                embed_mean_same,embed_sd_same,len_cell_embed,effect_mean,effect_sd,
                         n_embed_diff,n_embed_same,correct_iter,lib_simu=lib_simu,n_cell_total=None,
                                            distribution=distribution,activation=activation,
                        bw_pm=bw_pm,bw_lib=bw_lib,bw_nozero=bw_nozero,rand_seed=rand_seed,zero_prob=zero_prob,zero_set=zero_set,K=K,A=A,stat_estimation=stat_estimation,cell_scale=cell_scale) # peak*cell

            counts_list.append(counts)
            embed_peak_list.append(embed_peak)
            embed_lib_list.append(embed_lib)
            celltype_list.append([celltypes[i]]*counts.shape[1])
            lambdas_list.append(lambdas)
            simu_param_nozero_list.append(simu_param_nozero)
            simu_param_lib_list.append(simu_param_lib)
            simu_param_pm_list.append(simu_param_pm)

        if distribution=='Poisson':
            # atac_counts=np.hstack(counts_list)
            meta=np.hstack(celltype_list)
            embeds_peak=np.hstack(embed_peak_list)
            embeds_lib=np.hstack(embed_lib_list)
            #对整体lambdas进行sparsity修正
            lambdas=np.hstack(lambdas_list)
            simu_param_nozero=np.hstack(simu_param_nozero_list)
            simu_param_lib=np.hstack(simu_param_lib_list)
            simu_param_pm=peak_mean

            lambdas_sum=np.sum(lambdas,axis=0)


#             n_cell_total=len(simu_param_lib)
#             print("**********start ZIP correction...**********")
#             k_list,pi_list=[],[]
#             # 求解每个cell中lambda扩大的倍数和置零的比例
#             for i in range(n_cell_total):
#                 iter_=i
#                 # print(i)
#                 def solve_function(unsolved_value):
#                     k,pi=unsolved_value[0],unsolved_value[1]
#                     return [
#                         k*(1-pi)-simu_param_lib[iter_]/(lambdas_sum[iter_]),
#                         n_peak*pi+(1-pi)*np.sum(np.exp(-lambdas[:,iter_]*k))-(n_peak-simu_param_nozero[iter_])
#                     ]

#                 solved=fsolve(solve_function,[3,0.5],maxfev=2000)
#                 k,pi=solved[0],solved[1]
#                 simu1=k*(1-pi)*(lambdas_sum[iter_])
#                 real1=simu_param_lib[iter_]
#                 if abs(simu1-real1)/real1>0.1:
#                     print('=================================')
#                     print(i)
#                     print(simu1,real1)
#                     solved=fsolve(solve_function,[20,0.5],maxfev=2000)
#                 k,pi=solved[0],solved[1]
#                 simu1=k*(1-pi)*(lambdas_sum[iter_])
#                 real1=simu_param_lib[iter_]
#                 if abs(simu1-real1)/real1>0.1:
#                     print(i)
#                     print(simu1,real1)
#                     print('=================================')
#                 k_list.append(solved[0])
#                 pi_list.append(solved[1])
#             # 对每个cell的lambda置零并扩大相应倍数
#             for i in range(n_cell_total):
#                 if k_list[i]==3 or k_list[i]==20 or pi_list[i]<0 or k_list[i]<0:
#                     continue
#                 a=lambdas[:,i]*k_list[i]
#                 # b=atac_counts[:,i]
#                 a[np.random.choice(n_peak,replace=False,size=int(pi_list[i]*n_peak))]=0
#                 lambdas[:,i]=a
#             print("**********ZIP correction finished!**********")

            n_cell_total=len(simu_param_lib)
            print("**********start ZIP correction...**********")
            batch_size = 1000 # 并行数目，全局字典
            global k_dict,pi_dict
            for i in range(0,n_cell_total,batch_size):
                if i+batch_size<=n_cell_total:
                    my_thread = [zip_correction_thread(j,simu_param_lib[j],lambdas[:,j],lambdas_sum[j],simu_param_nozero[j],n_peak) for j in range(i, i+batch_size)]
                else:
                    my_thread = [zip_correction_thread(j,simu_param_lib[j],lambdas[:,j],lambdas_sum[j],simu_param_nozero[j],n_peak) for j in range(i, n_cell_total)]
                for thread_ in my_thread:
                    thread_.start()
                for thread_ in my_thread:
                    thread_.join()
            # 对每个cell的lambda置零并扩大相应倍数
            for i in range(n_cell_total):
                if k_dict[i]==3 or k_dict[i]==20 or pi_dict[i]<0 or k_dict[i]<0:
                    continue
                a=lambdas[:,i]*k_dict[i]
                # b=atac_counts[:,i]
                a[np.random.choice(n_peak,replace=False,size=int(pi_dict[i]*n_peak))]=0
                lambdas[:,i]=a

            print("**********ZIP correction finished!**********")

            # # spasity矫正完之后再来一轮peak mean和library size的矫正，保证都符合实际
            # lambdas_copy=lambdas.copy()
            # lambdas_copy=lambdas_copy/(np.sum(lambdas_copy,axis=1).reshape(-1,1)+1e-8)*(simu_param_pm.reshape(-1,1))*lambdas_copy.shape[1]
            # lambdas_copy=lambdas_copy/(np.sum(lambdas_copy,axis=0).reshape(1,-1)+1e-8)*(simu_param_lib.reshape(1,-1))

            atac_counts=np.random.poisson(lambdas, lambdas.shape)

        elif distribution=='Bernoulli':
            atac_counts=np.hstack(counts_list)
            meta=np.hstack(celltype_list)
            embeds_peak=np.hstack(embed_peak_list)
            embeds_lib=np.hstack(embed_lib_list)

        else:
            raise ValueError('wrong distribution input!')

        print("**********generate counts finshed!**********")

    else:
        raise ValueError('wrong simulation type!')
        
    adata_final=anndata.AnnData(X=scipy.sparse.csr_matrix(atac_counts.T))
    adata_final.obs['celltype']=meta
    return adata_final



if __name__ == '__main__':
    
    import argparse
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    import umap
    from sklearn.decomposition import PCA
    import episcanpy.api as epi
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="simCAS simulation")
    parser.add_argument('--adata_dir', type=str, default=None)
    parser.add_argument('--simu_type', type=str, default='cell_type') # cell_type/discrete/continuous
    parser.add_argument('--n_peak', type=int, default=-1)
    parser.add_argument('--n_cell_total', type=int, default=1500)
    parser.add_argument('--zero_prob', type=float, default=0.5)
    parser.add_argument('--zero_set', type=str, default='all') # 'all'/'by_row'
    parser.add_argument('--effect_mean', type=float, default=0.0)
    parser.add_argument('--effect_sd', type=float, default=1.0)
    parser.add_argument('--min_popsize', type=int, default=300)
    parser.add_argument('--min_pop', type=str, default=None)
    parser.add_argument('--tree_text', type=str, default=None)
    parser.add_argument('--pops_name', nargs='+',type=str, help="a list of names of cell populations",default=None)
    parser.add_argument('--pops_size', nargs='+',type=int, help="a list of numbers of cell populations",default=None)
    parser.add_argument('--embed_mean_same', type=float, default=1.0)
    parser.add_argument('--embed_sd_same', type=float, default=0.5)
    parser.add_argument('--embed_mean_diff', type=float, default=1.0)
    parser.add_argument('--embed_sd_diff', type=float, default=0.5)
    parser.add_argument('--len_cell_embed', type=int, default=12)
    parser.add_argument('--n_embed_diff', type=int, default=10)
    parser.add_argument('--correct_iter', type=int, default=2)
    parser.add_argument('--activation', type=str, default='exp_linear') # exp_linear/sigmod/exp
    parser.add_argument('--distribution', type=str, default='Poisson') # Poisson/Bernoulli
    parser.add_argument('--bw_pm', type=float, default=1e-4)
    parser.add_argument('--bw_lib', type=float, default=0.05)
    parser.add_argument('--bw_nozero', type=float, default=0.05)
    parser.add_argument('--K', type=float, default=None)
    parser.add_argument('--A', type=float, default=None)
    parser.add_argument('--stat_estimation', type=str, default='one_logser') # one_logser/zero_logser/ZIP/NB/zero_NB/ZINB
    parser.add_argument('--rand_seed', type=int, default=2022)
    parser.add_argument('--cell_scale', type=float, default=1.0)

    args = parser.parse_args()

    if args.simu_type=='cell_type':
        adata = sc.read(args.adata_dir)
        args.n_peak = adata.shape[1]

        adata_final=simCAS_generate(
            adata_dir=args.adata_dir,
            n_peak=args.n_peak,
            n_cell_total=None,
            rand_seed=args.rand_seed,
            zero_prob=args.zero_prob,
            zero_set=args.zero_set,
            effect_mean=args.effect_mean,
            effect_sd=args.effect_sd,
            min_popsize=args.min_popsize,
            min_pop=args.min_pop,
            tree_text=args.tree_text,
            pops_name=args.pops_name,
            pops_size=args.pops_size,
            embed_mean_same=args.embed_mean_same,
            embed_sd_same=args.embed_sd_same,
            embed_mean_diff=args.embed_mean_diff,
            embed_sd_diff=args.embed_sd_diff,
            len_cell_embed=args.len_cell_embed,
            n_embed_diff=args.n_embed_diff,
            simu_type=args.simu_type,
            correct_iter=args.correct_iter,
            activation=args.activation,
            distribution=args.distribution,
            bw_pm=args.bw_pm,
            bw_lib=args.bw_lib,
            bw_nozero=args.bw_nozero,
            K=args.K,
            A=args.A,
            stat_estimation=args.stat_estimation,
            cell_scale=args.cell_scale)

    elif args.simu_type=='discrete':
        if args.adata_dir!=None:
            adata=sc.read(resultdir+'adata_forsimulation.h5ad')
            peak_mean=np.log(cal_pm(adata)+1)
            lib_size=np.log(cal_lib(adata)+1)
            nozero=np.log(cal_nozero(adata)+1)
        else:
            print('use statistics of default peak-by-cell matrix')
            peak_mean=pd.read_csv('../data/peak_mean_log.csv',index_col=0)
            lib_size=pd.read_csv('../data/library_size_log.csv',index_col=0)
            nozero=pd.read_csv('../data/nozero_log.csv',index_col=0)

            peak_mean=np.array(peak_mean['peak mean'])
            lib_size=np.array(lib_size['library size'])
            nozero=np.array(nozero['nozero'])

        if args.n_peak==-1:
            args.n_peak=len(peak_mean)

        adata_final=simCAS_generate(
            peak_mean=peak_mean,
            lib_size=lib_size,
            nozero=nozero,
            adata_dir=args.adata_dir,
            n_peak=args.n_peak,
            n_cell_total=args.n_cell_total,
            rand_seed=args.rand_seed,
            zero_prob=args.zero_prob,
            zero_set=args.zero_set,
            effect_mean=args.effect_mean,
            effect_sd=args.effect_sd,
            min_popsize=args.min_popsize,
            min_pop=args.min_pop,
            tree_text=args.tree_text,
            pops_name=args.pops_name,
            pops_size=args.pops_size,
            embed_mean_same=args.embed_mean_same,
            embed_sd_same=args.embed_sd_same,
            embed_mean_diff=args.embed_mean_diff,
            embed_sd_diff=args.embed_sd_diff,
            len_cell_embed=args.len_cell_embed,
            n_embed_diff=args.n_embed_diff,
            simu_type=args.simu_type,
            correct_iter=args.correct_iter,
            activation=args.activation,
            distribution=args.distribution,
            bw_pm=args.bw_pm,
            bw_lib=args.bw_lib,
            bw_nozero=args.bw_nozero,
            K=args.K,
            A=args.A,
            stat_estimation=args.stat_estimation,
            cell_scale=args.cell_scale)

    elif args.simu_type=='continuous':
        if args.adata_dir!=None:
            adata=sc.read(resultdir+'adata_forsimulation.h5ad')
            peak_mean=np.log(cal_pm(adata)+1)
            lib_size=np.log(cal_lib(adata)+1)
            nozero=np.log(cal_nozero(adata)+1)
        else:
            print('use statistics of default peak-by-cell matrix')
            peak_mean=pd.read_csv('../data/peak_mean_log.csv',index_col=0)
            lib_size=pd.read_csv('../data/library_size_log.csv',index_col=0)
            nozero=pd.read_csv('../data/nozero_log.csv',index_col=0)

            peak_mean=np.array(peak_mean['peak mean'])
            lib_size=np.array(lib_size['library size'])
            nozero=np.array(nozero['nozero'])

        if args.n_peak==-1:
            args.n_peak=len(peak_mean)

        adata_final=simCAS_generate(
            peak_mean=peak_mean,
            lib_size=lib_size,
            nozero=nozero,
            adata_dir=args.adata_dir,
            n_peak=args.n_peak,
            n_cell_total=args.n_cell_total,
            rand_seed=args.rand_seed,
            zero_prob=args.zero_prob,
            zero_set=args.zero_set,
            effect_mean=args.effect_mean,
            effect_sd=args.effect_sd,
            min_popsize=args.min_popsize,
            min_pop=args.min_pop,
            tree_text=args.tree_text,
            pops_name=args.pops_name,
            pops_size=args.pops_size,
            embed_mean_same=args.embed_mean_same,
            embed_sd_same=args.embed_sd_same,
            embed_mean_diff=args.embed_mean_diff,
            embed_sd_diff=args.embed_sd_diff,
            len_cell_embed=args.len_cell_embed,
            n_embed_diff=args.n_embed_diff,
            simu_type=args.simu_type,
            correct_iter=args.correct_iter,
            activation=args.activation,
            distribution=args.distribution,
            bw_pm=args.bw_pm,
            bw_lib=args.bw_lib,
            bw_nozero=args.bw_nozero,
            K=args.K,
            A=args.A,
            stat_estimation=args.stat_estimation,
            cell_scale=args.cell_scale)
    else:
        raise ValueError('Wrong simu_type input!')

    print("the simulated adata shape is:",adata_final.shape)
    adata_final.write('../data/adata_final.h5ad')
