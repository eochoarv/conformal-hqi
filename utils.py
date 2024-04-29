#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 09:54:19 2023

@author: eochoa
"""

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.interpolate import interp1d
from collections import defaultdict

def conform_pred(cal_smx, val_smx, cal_labels, n, alpha, val_labels=None, cond=0):
    
    # 1: get conformal scores. n = calib_Y.shape[0]
    cal_scores = 1-cal_smx[np.arange(n),cal_labels]
    # 2: get adjusted quantile
    q_level = np.ceil((n+1)*(1-alpha))/n
    if alpha > 1 / (1+n):
        qhat = np.quantile(cal_scores, q_level, interpolation='higher')
    else:
        qhat = 1
    n_list = [cal_scores[cal_labels==i].shape[0] for i in range(cal_smx.shape[1])]
    # 3: get prediction sets
    prediction_sets = val_smx >= (1-qhat) # 3: form prediction sets
    
    if cond == 1:
        # 4: if class conditional
        q_level_cond = [np.ceil((n+1)*(1-alpha))/n for n in n_list]
        qhat_cond = np.array([1 if alpha < 1 / (1+n_list[i]) else 
                              np.quantile(cal_scores[cal_labels==i], q_level_cond[i], interpolation='higher') 
                              if i in np.unique(cal_labels) else 0 for i in range(cal_smx.shape[1])])
        prediction_sets_cond = (val_smx >= (1-qhat_cond))  
        if val_labels is None: 
            return prediction_sets_cond, qhat_cond
        # Calculate empirical coverage
        empirical_coverage = prediction_sets_cond[np.arange(prediction_sets_cond.shape[0]),val_labels].mean()
        return (prediction_sets_cond, empirical_coverage), qhat_cond
        
    else:
        if val_labels is None: 
            return prediction_sets, qhat
        # Calculate empirical coverage
        empirical_coverage = prediction_sets[np.arange(prediction_sets.shape[0]),val_labels].mean()
        return (prediction_sets, empirical_coverage), qhat
    
# Pearson correlation coefficient 
def smx_pcc(X_train, X_test, labels_train):
    
    n_ref = X_train.shape[0]
    df_R = np.corrcoef(X_train,  X_test)[:n_ref, n_ref:]
    
    #df_R = 1-df_R
    #df_R = np.exp(-df_R/.1)
    
    df_R = pd.DataFrame(df_R)
    df_R['polymer_ref'] = labels_train
    smx = (df_R.melt(id_vars=['polymer_ref']).
             groupby(['variable','polymer_ref']).
             max().
             reset_index().
             pivot(index='variable', columns='polymer_ref', values='value').
             to_numpy())
    
    return smx

# nearest neighbor
def smx_nn(X_train, X_test, labels_train, tau=10):
    
    
    D = pairwise_distances(X_train, X_test)
    df_D = pd.DataFrame(D)
    df_D['polymer_ref'] = labels_train
    smx = (df_D.melt(id_vars=['polymer_ref']).
                 groupby(['variable','polymer_ref']).
                 min().
                 reset_index().
                 pivot(index='variable', columns='polymer_ref', values='value').
                 to_numpy())
    smx_1 = (smx / smx.min(1)[:,None])
    smx_2 = (smx / np.sort(smx, 1)[:,1][:,None])
    smx_1[smx_1 == 1] = smx_2[smx_1 == 1]
    smx = np.exp(-smx_1/tau)

    return smx
 
# function to split data into calibration and teste sets
def split_cal_test(smx, labels_test, cal_prop=0.5, seed=0):
    
    n = int(smx.shape[0] * cal_prop) # number of calibration points
    idx = np.array([1] * n + [0] * (smx.shape[0]-n)) > 0
    np.random.seed(seed)
    np.random.shuffle(idx)
    cal_smx, val_smx = smx[idx,:], smx[~idx,:]
    cal_labels, val_labels = labels_test[idx], labels_test[~idx]

    return cal_smx, val_smx, cal_labels, val_labels, idx

# function to run the numerical simulations of the paper
def experiment(smx, label_test, alphas, N, adaptive=0, cal_prop=0.8, initial_seed=0, filter_label=None):

    res_dict_hqi = defaultdict(list) 
    cov_by_label_hqi = defaultdict(list)
    cov_by_set_size_hqi = defaultdict(list)
    set_size_by_label_hqi = defaultdict(list)
    qhat_dict = defaultdict(list)
    
    unique_labels = np.unique(label_test)
    
    for alpha in alphas:
        cov = []
        set_size = []
        #qhat_list = np.zeros(n_sim)
        zero_sets = []
        one_sets = []
        n_sim = 0
        for i in range(N):
            
            cal_smx, val_smx, cal_labels, val_labels, _ = split_cal_test(smx, label_test, cal_prop, initial_seed+i)
            
            if filter_label is not None:
                val_smx = val_smx[val_labels == filter_label, :]
                val_labels = val_labels[val_labels == filter_label]
        
            if adaptive == 1 and not (np.array([(cal_labels == l).sum() for l in unique_labels]) > np.ceil((1-alpha) /alpha)).all():
                continue
            
            n_sim = n_sim + 1
            n = cal_smx.shape[0]
            (prediction_sets, empirical_coverage), qhat = conform_pred(cal_smx, val_smx, cal_labels, n, alpha, val_labels, cond=adaptive)
            
            cov.append(empirical_coverage)
            set_size.append(prediction_sets.sum(1).mean())
            zero_sets.append((prediction_sets.sum(1) == 0).mean())
            one_sets.append((prediction_sets.sum(1) == 1).mean())
            #qhat_list[i] = qhat
            
            cov_by_label_hqi[alpha].append(
            pd.DataFrame({'label':val_labels, 
                          'coverage':prediction_sets[range(val_smx.shape[0]),
                                                     val_labels]}).groupby('label').mean()* 100
        )
            cov_by_set_size_hqi[alpha].append(
            pd.DataFrame({'set_size':prediction_sets.sum(1), 
                          'coverage':prediction_sets[range(val_smx.shape[0]),
                                                     val_labels]}).groupby('set_size').mean()* 100
        )
            set_size_by_label_hqi[alpha].append(
            pd.DataFrame({'label':val_labels, 
                          'set_size':prediction_sets.sum(1)}).groupby('label').mean()
        )
            
        res_dict_hqi['n_sim'].append(n_sim)
        res_dict_hqi['alpha'].append(alpha)    
        res_dict_hqi['cov_mean'].append(np.mean(cov))
        res_dict_hqi['set_size_mean'].append(np.mean(set_size))
        
        res_dict_hqi['cov_sd'].append(np.std(cov))
        res_dict_hqi['set_size_sd'].append(np.std(set_size))
        #qhat_dict['set_size_sd'].append(qhat_list)
        res_dict_hqi['zero_sets'].append(np.mean(zero_sets))
        res_dict_hqi['one_sets'].append(np.mean(one_sets))
        
    res_df_hqi = pd.DataFrame(res_dict_hqi)
    
    return res_df_hqi, (cov_by_label_hqi, cov_by_set_size_hqi, set_size_by_label_hqi)
    

def interpol_spect(df_spectr, min_wl=0, max_wl=3500, n=1400, id_spetr='id', labeled=False):
    
    spectr_list = df_spectr[id_spetr].drop_duplicates()
    df_spectr_min = df_spectr.groupby(id_spetr).min()
    df_spectr_max = df_spectr.groupby(id_spetr).max()
    all_wl = np.linspace(min_wl, max_wl, num=n, endpoint=True)

    df_spectr_inter = pd.DataFrame()
    

    for fn in spectr_list:
        df_spectr_fn = df_spectr.query(id_spetr + ' == @fn').reset_index(drop=True)
        #df_lenght = df_spectr_fn.shape[0]    
        min_wl = df_spectr_min.loc[fn, 'wavelength']
        max_wl = df_spectr_max.loc[fn, 'wavelength']
        
        if labeled:
            polymer_fn = df_spectr_fn['polymer'][0]
            label_fn = df_spectr_fn['label'][0]
        else:
            polymer_fn = ''
            label_fn = ''
    
        x = df_spectr_fn.loc[:, 'wavelength']
        y = df_spectr_fn.loc[:, 'intensity']
        f = interp1d(x, y)
        #x_new = np.linspace(min_wl, max_wl, num=df_lenght, endpoint=True)
        x_new = all_wl[np.logical_and(all_wl >= min_wl, all_wl <= max_wl)]
        x_out = all_wl[np.logical_or(all_wl < min_wl, all_wl > max_wl)]
        
        df_spectr_inter = pd.concat([df_spectr_inter, 
                                  pd.DataFrame({'wavelength':np.concatenate([x_new, x_out]), 
                                                'intensity':np.concatenate([f(x_new), np.zeros(len(x_out))]) , 
                                                'polymer':[polymer_fn] * n ,
                                                'label':[label_fn] * n ,
                                                id_spetr:[fn] * n }).sort_values('wavelength')])
        
    return df_spectr_inter.reset_index(drop=True)

def hom_data(reference_library, calibration_set, test_set):    
    # Interpolate calibration set and test set to make sure they are compatible with reference library wavenumbers
    
    wl_list = np.float64(reference_library.columns[3:])
    calibration_set_l = calibration_set.melt(id_vars=['id', 'polymer', 'label'],
                               var_name='wavelength',
                               value_name='intensity')
    calibration_set_l['wavelength'] = np.float64(calibration_set_l['wavelength'])

    calibration_set_l = interpol_spect(calibration_set_l, 
                                min_wl=wl_list.min(), 
                                max_wl=wl_list.max(), n=wl_list.shape[0], id_spetr='id', labeled=True)

    calibration_set = calibration_set_l.pivot(index=['id', 'polymer', 'label'], 
                                columns='wavelength', 
                                values='intensity').reset_index()

    if 'polymer' in test_set.columns:
        test_set_l = test_set.melt(id_vars=['id', 'polymer', 'label'],
                                   var_name='wavelength',
                                   value_name='intensity')

        test_set_l['wavelength'] = np.float64(test_set_l['wavelength'])

        test_set_l = interpol_spect(test_set_l, 
                                    min_wl=wl_list.min(), 
                                    max_wl=wl_list.max(), n=wl_list.shape[0], id_spetr='id',labeled=True)

        test_set = test_set_l.pivot(index=['id', 'polymer', 'label'], 
                                    columns='wavelength', 
                                    values='intensity').reset_index()

    else:
        test_set_l = test_set.melt(id_vars=['id'],
                                   var_name='wavelength',
                                   value_name='intensity')

        test_set_l['wavelength'] = np.float64(test_set_l['wavelength'])

        test_set_l = interpol_spect(test_set_l, 
                                    min_wl=wl_list.min(), 
                                    max_wl=wl_list.max(), n=wl_list.shape[0], id_spetr='id')

        test_set = test_set_l.pivot(index=['id', 'polymer', 'label'], 
                                    columns='wavelength', 
                                    values='intensity').reset_index()
    return calibration_set, test_set
