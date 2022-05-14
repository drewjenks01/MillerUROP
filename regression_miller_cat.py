# -*- coding: utf-8 -*-
#%%

#import cProfile
from contextlib import contextmanager
import sys, os
import neural_analysis.plots as plotting
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import sem
import pickle
from tqdm import tqdm
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind
"""
Created on Mon Sep 20 16:57:30 2021

@author: drewj
"""
#used for profiling the code
# pr = cProfile.Profile()
# pr.enable()

#to suppress outputs
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


#########################################
#######       DEFINITIONS        ########
#######         SETTINGS         ########
#########################################

#which data to plot
plot_sem=True  #sem data
plot_pooled=False   #categorical data
plot_bounds=False
test_sig=False

#all different possible valid areas
areas={'PFC', 'Cd', 'CA1','CA2','CA3','CA4','Hpc','DG'}

#boolean to determine if Hpc components grouped together or seperate
Hpc_together=True

#valid Hpc areas
if Hpc_together:
    Hpcs={'CA1','CA2','CA3','CA4','Hpc','DG'}
    sub_dict={'Hpc':[],'PFC':[],'Cd':[]}
else:
    Hpcs={'Hpc'}
    sub_dict={'Hpc':[],'PFC':[],'Cd':[],'CA1':[],'CA2':[],'CA3':[],'CA4':[]}


#set design method: pooled, allcombs, nested
design_method='allterms'

#diff tasks based in design method
if design_method=='pooled':
    tasks=  ['Quadrant', 'Object', 'Expected_response', 'Interaction']

elif design_method=='nested':
    tasks=['quadrant_category', 'Quadrant_stimulus', 'object_category', 'Object_stimulus', 'Expected_response', 'Interaction']

elif design_method=='allcombs':
    tasks=['quadrant_category', 'Quadrant_scrambled', 'object_category', 'Object_scrambled', 'Expected_response', 'Interaction']

elif design_method=='allterms':
    tasks=['quadrant_category','quadrant_withinCategory1','quadrant_withinCategory2','object_category','object_withinCategory1',
    'object_withinCategory2','response','interaction_0&4','interaction_0&5','interaction_1&3','interaction_1&4','interaction_1&5',
    'interaction_2&2','interaction_2&4','interaction_2&5']



#CAT areas
quad_ob=['quadrant_category','Qaudrant_scrambled','object_category','Object_scrambled']


interaction_terms = [term for term in tasks if 'interaction' in term]
# print(interaction_terms)
quad_total_terms= ['quadrant_category', 'quadrant_withinCategory1', 'quadrant_withinCategory2']
obj_total_terms=['object_category', 'object_withinCategory1', 'object_withinCategory2']
# other_terms=['quadrant_category','object_category','response']
other_terms=['quadrant_category','object_category','response']
#sem_tasks=[ 'quadrant_category','quadrant_total','object_category','object_total','response','interaction']
sem_tasks=['interaction', 'quadrant_category','quadrant_total','object_category','object_total','response']
paper_tasks=['object_total','quadrant_total','interaction','response']




#load data and cat data
with open("cat_dict.pkl", "rb") as fp:
    cat_dict = pickle.load(fp)

with open("sem_data.pkl", "rb") as fg:
    data = pickle.load(fg)

with open("timepoints.pkl", "rb") as tp:
    timepoints = pickle.load(tp)


with open("sig_data.pkl", "rb") as fm:
        sig_data = pickle.load(fm)

with open("onesamp.pkl", "rb") as fl:
    onesamp = pickle.load(fl)

with open("ind_test.pkl", "rb") as fn:
    ind_test = pickle.load(fn)


#########################################
#######    Categorical Funcs      #######
#########################################

def compute_total_sensory(betweenCategory, *withinCategory):
    """
    Computes total and senory information of Brincat et al PNAS 2018 from individual-term neural information
    """
    n_within_terms = len(withinCategory)
    
    # Total information (PEV) for all terms = upper-bound for category information
    total = betweenCategory.copy()
    for term in range(n_within_terms):
        total += withinCategory[term]

    # "Sensory" information (PEV) = average of all terms ~ lower-bound for category information
    sensory = total / (n_within_terms + 1)

    return total, sensory

    
def compute_category_index(betweenCategory, total, sensory, timepoints=None, window=None):
    """
    Computes category index of Brincat et al PNAS 2018 from between-category, total, and sensory neural information
    """
    # Compute index at each timepoints
    if window is None:
        category_index = (betweenCategory - sensory) / (total - sensory)

    # Compute index pooled over given time window (as in PNAS paper)
    else:
        tbool   = (timepoints >= window[0]) & (timepoints <= window[1])
        num     = (betweenCategory - sensory).sum(axis=0)
        num=num[tbool]
        denom   = (total - sensory).sum(axis=0)
        denom=denom[tbool]
        category_index = num / denom

    return category_index




    
#########################################
#######    Data Manipulation      #######
#########################################

#########################################
########        SEM             #########
#########################################
if plot_sem:

    #dictionary with means for each timepoint: key=area, val=list of means
    mean_dict={}

    for task in sem_tasks:
        mean_dict[task]=pickle.loads(pickle.dumps(sub_dict))

    #print(data[-0.7].keys())


    
    #for each area
    for ar in mean_dict['interaction']:

        #for each timepoint
        for val in data:
            
            int_arr=[]
            
            
            
            for term in interaction_terms:

               if data[val][term][ar]!=[]:

                    #np array holding all data for interaction terms
                    #int_arr=[arr[ar] for arr in data[val] if arr in interaction_terms]
                    int_arr.append(data[val][term][ar])
            

            int_arr=np.array(int_arr)

            #summ terms together --> 1d array
            int_arr=np.sum(int_arr,axis=0)

            avg=np.mean(int_arr)
            mean_dict['interaction'][ar].append(avg)

    
    for ar in mean_dict['quadrant_total']:
        #for each timepoint
        for val in data:
            
            quad_arr=[]


            for term in quad_total_terms:

                if data[val][term][ar]!=[]:

                    #np array holding all data for interaction terms
                    #quad_arr=[arr[ar] for arr in data[val] if arr in quad_total_terms]
                    quad_arr.append(data[val][term][ar])

            # quad_arr=np.array(quad_arr)
            #print('int_arr(8d):',quad_arr.shape)

            quad_arr=np.array(quad_arr)
            #summ terms together --> 1d array
            quad_arr=np.sum(quad_arr,axis=0)

            avg=np.mean(quad_arr)
            mean_dict['quadrant_total'][ar].append(avg)

    for ar in mean_dict['object_total']:
        #for each timepoint
        for val in data:
            
            obj_arr=[]

            for term in obj_total_terms:

                if data[val][term][ar]!=[]:

                    #np array holding all data for interaction terms
                   
                    obj_arr.append(data[val][term][ar])

            obj_arr=np.array(obj_arr)
            # print('obj_arr(8d):',obj_arr.shape)

            #summ terms together --> 1d array
            obj_arr=np.sum(obj_arr,axis=0)

            avg=np.mean(obj_arr)
            mean_dict['object_total'][ar].append(avg)
           
           


    for task in other_terms:

        for ar in mean_dict[task]:

            for val in data:

                if data[val][task][ar]!=[]:

                    #if not empty, find avg of values (val of summed spikes, divided by number of contributing neurons in area)
                    avg=sum(data[val][task][ar])/len(data[val][task][ar])
                    

                    mean_dict[task][ar].append(avg)

    #dic to hold standard error of means data
    sem_dict={}

    for task in sem_tasks:
        sem_dict[task]=pickle.loads(pickle.dumps(sub_dict))

    #loop through timepoints in data
    for d in data:


        #loop through tasks
        for t in data[d]:

            if t not in interaction_terms and t not in obj_total_terms and t not in quad_total_terms:

                #loop through areas
                for a in data[d][t]:

                    if data[d][t][a]!=[]:


                        #find standard error for spec area and timepoint
                        sem_dict[t][a].append(sem(data[d][t][a]))

            elif t =='interaction_0&4':
                int_dic={'Hpc':[],'PFC':[],'Cd':[]}
                #loop through areas
                for a in data[d][t]:

                    

                    for term in interaction_terms:
                        int_dic[a].append(data[d][term][a])


                    int_dic[a]=np.array(int_dic[a])


                for k in int_dic:
                    int_dic[k]=np.sum(int_dic[k],axis=0)
                    int_dic[k]=int_dic[k].tolist()


                    sem_dict['interaction'][k].append(sem(int_dic[k]))
                  #  print( len(sem_dict['interaction'][k]))

            # elif 'interaction' in t:
            #     int_dic={'Hpc':[],'PFC':[],'Cd':[]}
            #     #loop through areas
            #     for a in data[d][t]:

            #         if data[d][t][a]!=[]:


            #             #find standard error for spec area and timepoint
            #             sem_dict[t][a].append(sem(data[d][t][a]))




            elif t =='quadrant_withinCategory1':
                q_dic={'Hpc':[],'PFC':[],'Cd':[]}
                #loop through areas
                for a in data[d][t]:

                    

                    for term in quad_total_terms:
                        q_dic[a].append(data[d][term][a])


                    q_dic[a]=np.array(q_dic[a])


                for k in q_dic:
                    q_dic[k]=np.sum(q_dic[k],axis=0)
                    q_dic[k]=q_dic[k].tolist()


                    sem_dict['quadrant_total'][k].append(sem(q_dic[k]))


            elif t =='object_withinCategory1':
                o_dic={'Hpc':[],'PFC':[],'Cd':[]}
                #loop through areas
                for a in data[d][t]:

                    

                    for term in obj_total_terms:
                        o_dic[a].append(data[d][term][a])


                    o_dic[a]=np.array(o_dic[a])


                for k in o_dic:
                    o_dic[k]=np.sum(o_dic[k],axis=0)
                    o_dic[k]=o_dic[k].tolist()


                    sem_dict['object_total'][k].append(sem(o_dic[k]))


            elif t =='object_category':
                    #loop through areas
                for a in data[d][t]:

                    if data[d][t][a]!=[]:


                        #find standard error for spec area and timepoint
                        sem_dict[t][a].append(sem(data[d][t][a]))



            elif t =='quadrant_category':
                 #loop through areas
                for a in data[d][t]:

                    if data[d][t][a]!=[]:


                        #find standard error for spec area and timepoint
                        sem_dict[t][a].append(sem(data[d][t][a]))





    #boolean for plotting
    plot_shadow= True


    if plot_shadow:

    
        #loop through tasks
        for task in paper_tasks:
            #ylim[1] - j*0.02*(ylim[1] - ylim[0])

            if task=='object_total':
                ylim=(-0.05,2)
            elif task=='quadrant_total':
                ylim=(-0.05,6.5)
            elif task=='interaction':
                ylim=(-0.05,1)

            elif task=='response':
                ylim=(-0.05,1.4)


            plt.figure(figsize=(16,12))

            #Hpc
            if mean_dict[task]['Hpc']:
                plotting.plot_line_with_error_fill(timepoints,mean_dict[task]['Hpc'],err= sem_dict[task]['Hpc'], events=[0.0,0.5],color=['green'])
                hpc_sig=onesamp[task]['Hpc']
                hpc_sig = np.array(hpc_sig,dtype=np.double)
                hpc_sig[ hpc_sig==0 ] = np.nan

                plt.plot(timepoints,hpc_sig*(ylim[1] - 3*0.02*(ylim[1] - ylim[0])),'g*',markersize=5)




            #Hpc Seperate if setting TRUE
            if not Hpc_together:

                #CA1
                if mean_dict[task]['CA1']:
                    plotting.plot_line_with_error_fill(timepoints,mean_dict[task]['CA1'],err= sem_dict[task]['CA1'], events=[0.0,0.5],color=['purple'])
                    

                #CA2
                if mean_dict[task]['CA2']:
                    plotting.plot_line_with_error_fill(timepoints,mean_dict[task]['CA2'],err= sem_dict[task]['CA2'], events=[0.0,0.5],color=['lime'])

                #CA3
                if mean_dict[task]['CA3']:
                    plotting.plot_line_with_error_fill(timepoints,mean_dict[task]['CA3'],err= sem_dict[task]['CA3'], events=[0.0,0.5],color=['orange'])

                #CA4
                if mean_dict[task]['CA4']:
                    plotting.plot_line_with_error_fill(timepoints,mean_dict[task]['CA4'],err= sem_dict[task]['CA4'], events=[0.0,0.5],color=['cyan'])


            #Cd
            if mean_dict[task]['Cd']:
                plotting.plot_line_with_error_fill(timepoints,mean_dict[task]['Cd'],err= sem_dict[task]['Cd'], events=[0.0,0.5],color=['blue'])
                cd_sig=onesamp[task]['Cd']
                cd_sig = np.array(cd_sig,dtype=np.double)
                cd_sig[ cd_sig==0 ] = np.nan

                plt.plot(timepoints,cd_sig*(ylim[1] - 1*0.02*(ylim[1] - ylim[0])),'b*',markersize=5)
                
            #PFC
            if mean_dict[task]['PFC']:
                plotting.plot_line_with_error_fill(timepoints,mean_dict[task]['PFC'],err= sem_dict[task]['PFC'], events=[0.0,0.5],color=['red'])
                pfc_sig=onesamp[task]['PFC']
                pfc_sig = np.array(pfc_sig,dtype=np.double)
                pfc_sig[ pfc_sig==0 ] = np.nan

                plt.plot(timepoints,pfc_sig*(ylim[1] - 2*0.02*(ylim[1] - ylim[0])),'r*',markersize=5)
            
            
            
            #plot 2nd sig test
            #('Hpc','PFC'),('Hpc','Cd'),('PFC','Cd')
            ind_test[task]['HP'] = np.array(ind_test[task]['HP'],dtype=np.double)
            ind_test[task]['HP'][ ind_test[task]['HP']==0 ] = np.nan

            ind_test[task]['HC'] = np.array(ind_test[task]['HC'],dtype=np.double)
            ind_test[task]['HC'][ ind_test[task]['HC']==0 ] = np.nan

            ind_test[task]['PC'] = np.array(ind_test[task]['PC'],dtype=np.double)
            ind_test[task]['PC'][ ind_test[task]['PC']==0 ] = np.nan
            
            
            plt.plot(timepoints,ind_test[task]['HP']*(ylim[1] - 4*0.02*(ylim[1] - ylim[0])),'ko',markersize=5)
            plt.plot(timepoints,ind_test[task]['HC']*(ylim[1] - 5*0.02*(ylim[1] - ylim[0])),'ko',markersize=5)
            plt.plot(timepoints,ind_test[task]['PC']*(ylim[1] - 6*0.02*(ylim[1] - ylim[0])),'ko',markersize=5)

            

                
            # plt.ylabel('PEV')
            # plt.xlabel('Time (s)')
            Cd_col = mpatches.Patch(color='blue', label='Cd',lw=0.5)
            PFC_col = mpatches.Patch(color='red', label='PFC',lw=0.5)
            Hpc_col = mpatches.Patch(color='green', label='Hpc',lw=0.5)
            if not Hpc_together:
                CA1_col = mpatches.Patch(color='purple', label='CA1',lw=0.5)
                CA2_col = mpatches.Patch(color='lime',  label='CA2',lw=0.5)
                CA3_col = mpatches.Patch(color='orange', label='CA3',lw=0.5)
                CA4_col = mpatches.Patch(color='cyan', label='CA4',lw=0.5)
                plt.legend(handles=[Cd_col,PFC_col,Hpc_col,CA1_col,CA2_col,CA3_col,CA4_col],loc='upper right')
            else:
                plt.legend(handles=[Cd_col,PFC_col,Hpc_col])
            #plt.suptitle(task)
            plt.autoscale(enable=True) 
            plt.ylim(ylim[0],ylim[1])
            plt.xlim((-0.5,1.4))
            plt.savefig('/Users/drewj/Documents/millerUROP/Regr/'+design_method+' sem/'+task+'_regress'+'.png',facecolor='white')
            plt.show()


#########################################
########        Bounds             ######
#########################################
if plot_bounds:

    #dictionary with means for each timepoint: key=area, val=list of means
    mean_dict={}

    for task in sem_tasks:
        mean_dict[task]=pickle.loads(pickle.dumps(sub_dict))

    #print(data[-0.7].keys())


    
    #for each area
    for ar in mean_dict['interaction']:

        #for each timepoint
        for val in data:
            
            int_arr=[]
            
            
            
            for term in interaction_terms:

               if data[val][term][ar]!=[]:

                    #np array holding all data for interaction terms
                    #int_arr=[arr[ar] for arr in data[val] if arr in interaction_terms]
                    int_arr.append(data[val][term][ar])
            

            int_arr=np.array(int_arr)

            #summ terms together --> 1d array
            int_arr=np.sum(int_arr,axis=0)

            avg=np.mean(int_arr)
            mean_dict['interaction'][ar].append(avg)

    
    for ar in mean_dict['quadrant_total']:
        #for each timepoint
        for val in data:
            
            quad_arr=[]


            for term in quad_total_terms:

                if data[val][term][ar]!=[]:

                    #np array holding all data for interaction terms
                    #quad_arr=[arr[ar] for arr in data[val] if arr in quad_total_terms]
                    quad_arr.append(data[val][term][ar])

            # quad_arr=np.array(quad_arr)
            #print('int_arr(8d):',quad_arr.shape)

            quad_arr=np.array(quad_arr)
            #summ terms together --> 1d array
            quad_arr=np.sum(quad_arr,axis=0)

            avg=np.mean(quad_arr)
            mean_dict['quadrant_total'][ar].append(avg)

    for ar in mean_dict['object_total']:
        #for each timepoint
        for val in data:
            
            obj_arr=[]

            for term in obj_total_terms:

                if data[val][term][ar]!=[]:

                    #np array holding all data for interaction terms
                   
                    obj_arr.append(data[val][term][ar])

            obj_arr=np.array(obj_arr)
            # print('obj_arr(8d):',obj_arr.shape)

            #summ terms together --> 1d array
            obj_arr=np.sum(obj_arr,axis=0)

            avg=np.mean(obj_arr)
            mean_dict['object_total'][ar].append(avg)
           
           


    for task in other_terms:

        for ar in mean_dict[task]:

            for val in data:

                if data[val][task][ar]!=[]:

                    #if not empty, find avg of values (val of summed spikes, divided by number of contributing neurons in area)
                    avg=sum(data[val][task][ar])/len(data[val][task][ar])
                    

                    mean_dict[task][ar].append(avg)


    #extract task and area data and pass as np array
    hpc_qc=np.array(cat_dict[tasks[0]]['Hpc'])
    hpc_qwc1=np.array(cat_dict[tasks[1]]['Hpc'])
    hpc_qwc2=np.array(cat_dict[tasks[2]]['Hpc'])
    hpc_oc=np.array(cat_dict[tasks[3]]['Hpc'])
    hpc_owc1=np.array(cat_dict[tasks[4]]['Hpc'])
    hpc_owc2=np.array(cat_dict[tasks[5]]['Hpc'])


    pfc_qc=np.array(cat_dict[tasks[0]]['PFC'])
    pfc_qwc1=np.array(cat_dict[tasks[1]]['PFC'])
    pfc_qwc2=np.array(cat_dict[tasks[2]]['PFC'])
    pfc_oc=np.array(cat_dict[tasks[3]]['PFC'])
    pfc_owc1=np.array(cat_dict[tasks[4]]['PFC'])
    pfc_owc2=np.array(cat_dict[tasks[5]]['PFC'])

    cd_qc=np.array(cat_dict[tasks[0]]['Cd'])
    cd_qwc1=np.array(cat_dict[tasks[1]]['Cd'])
    cd_qwc2=np.array(cat_dict[tasks[2]]['Cd'])
    cd_oc=np.array(cat_dict[tasks[3]]['Cd'])
    cd_owc1=np.array(cat_dict[tasks[4]]['Cd'])
    cd_owc2=np.array(cat_dict[tasks[5]]['Cd'])
        


     #calc total and sensory info
    hpc_quad_total,hpc_quad_sens=compute_total_sensory(hpc_qc,hpc_qwc1,hpc_qwc2)
    hpc_obj_total,hpc_obj_sens=compute_total_sensory(hpc_oc,hpc_owc1,hpc_owc2)
    hpc_int_total,hpc_int_sens=compute_total_sensory(np.array(cat_dict['response']['Hpc']),*[np.array(cat_dict[term]['Hpc']) for term in interaction_terms if 'interaction' in term])


    pfc_quad_total,pfc_quad_sens=compute_total_sensory(pfc_qc,pfc_qwc1,pfc_qwc2)
    pfc_obj_total,pfc_obj_sens=compute_total_sensory(pfc_oc,pfc_owc1,pfc_owc2)
    pfc_int_total,pfc_int_sens=compute_total_sensory(np.array(cat_dict['response']['PFC']),*[np.array(cat_dict[term]['PFC']) for term in interaction_terms if 'interaction' in term])

    cd_quad_total,cd_quad_sens=compute_total_sensory(cd_qc,cd_qwc1,cd_qwc2)
    cd_obj_total,cd_obj_sens=compute_total_sensory(cd_oc,cd_owc1,cd_owc2)
    cd_int_total,cd_int_sens=compute_total_sensory(np.array(cat_dict['response']['Cd']),*[np.array(cat_dict[term]['Cd']) for term in interaction_terms if 'interaction' in term])


    #plot

    #hpc
    plt.figure()
    plt.plot(timepoints,mean_dict['quadrant_category']['Hpc'])    #normal pev
    plt.plot(timepoints,np.nanmean(hpc_quad_total,axis=0)) #upper bound
    plt.plot(timepoints,np.nanmean(hpc_quad_sens,axis=0)) #lower bound
    plt.xlim((-0.5,1.4))
    plt.title("HPC Quad Index")
    plt.legend(['Mean','Upper','Lower'])
    plt.savefig('/Users/drewj/Documents/millerUROP/Regr/bounds/hpc_quadcat'+'.png',facecolor='white')
    plt.show()

    plt.figure()
    plt.plot(timepoints,mean_dict['object_category']['Hpc'])    #normal pev
    plt.plot(timepoints,np.nanmean(hpc_obj_total,axis=0)) #upper bound
    plt.plot(timepoints,np.nanmean(hpc_obj_sens,axis=0)) #lower bound
    plt.title("HPC Object Index")
    plt.legend(['Mean','Upper','Lower'])
    plt.xlim([0.0,1.5])
    plt.savefig('/Users/drewj/Documents/millerUROP/Regr/bounds/hpc_objcat'+'.png',facecolor='white')
    plt.show()

    plt.figure()
    plt.plot(timepoints,mean_dict['response']['Hpc'])    #normal pev
    plt.plot(timepoints,np.nanmean(hpc_int_total,axis=0)) #upper bound
    plt.plot(timepoints,np.nanmean(hpc_int_sens,axis=0)) #lower bound
    plt.title("HPC Int Index")
    plt.legend(['Mean','Upper','Lower'])
    plt.xlim([0.0,1.5])
    plt.savefig('/Users/drewj/Documents/millerUROP/Regr/bounds/hpc_intcat'+'.png',facecolor='white')
    plt.show()
    

    #pfc
    plt.figure()
    plt.plot(timepoints,mean_dict['quadrant_category']['PFC'])    #normal pev
    plt.plot(timepoints,np.nanmean(pfc_quad_total,axis=0)) #upper bound
    plt.plot(timepoints,np.nanmean(pfc_quad_sens,axis=0)) #lower bound
    plt.title("PFC Quad Index")
    plt.legend(['Mean','Upper','Lower'])
    plt.xlim([0.0,1.5])
    plt.savefig('/Users/drewj/Documents/millerUROP/Regr/bounds/pfc_quadcat'+'.png',facecolor='white')
    plt.show()

    plt.figure()
    plt.plot(timepoints,mean_dict['object_category']['PFC'])    #normal pev
    plt.plot(timepoints,np.nanmean(pfc_obj_total,axis=0)) #upper bound
    plt.plot(timepoints,np.nanmean(pfc_obj_sens,axis=0)) #lower bound
    plt.title("PFC Object Index")
    plt.legend(['Mean','Upper','Lower'])
    plt.xlim([0.0,1.5])
    plt.savefig('/Users/drewj/Documents/millerUROP/Regr/bounds/pfc_objcat'+'.png',facecolor='white')
    plt.show()

    plt.figure()
    plt.plot(timepoints,mean_dict['response']['PFC'])    #normal pev
    plt.plot(timepoints,np.nanmean(pfc_int_total,axis=0)) #upper bound
    plt.plot(timepoints,np.nanmean(pfc_int_sens,axis=0)) #lower bound
    plt.title("PFC Int Index")
    plt.legend(['Mean','Upper','Lower'])
    plt.xlim([0.0,1.5])
    plt.savefig('/Users/drewj/Documents/millerUROP/Regr/bounds/pfc_intcat'+'.png',facecolor='white')
    plt.show()

    #cd
    plt.figure()
    plt.plot(timepoints,mean_dict['quadrant_category']['Cd'])    #normal pev
    plt.plot(timepoints,np.nanmean(cd_quad_total,axis=0)) #upper bound
    plt.plot(timepoints,np.nanmean(cd_quad_sens,axis=0)) #lower bound
    plt.title("Cd Quad Index")
    plt.legend(['Mean','Upper','Lower'])
    plt.xlim([0.0,1.5])
    plt.savefig('/Users/drewj/Documents/millerUROP/Regr/bounds/cd_quadcat'+'.png',facecolor='white')
    plt.show()

    plt.figure()
    plt.plot(timepoints,mean_dict['object_category']['Cd'])    #normal pev
    plt.plot(timepoints,np.nanmean(cd_obj_total,axis=0)) #upper bound
    plt.plot(timepoints,np.nanmean(cd_obj_sens,axis=0)) #lower bound
    plt.title("Cd Object Index")
    plt.legend(['Mean','Upper','Lower'])
    plt.xlim([0.0,1.5])
    plt.savefig('/Users/drewj/Documents/millerUROP/Regr/bounds/cd_objcat'+'.png',facecolor='white')
    plt.show()

    plt.figure()
    plt.plot(timepoints,mean_dict['response']['Cd'])    #normal pev
    plt.plot(timepoints,np.nanmean(cd_int_total,axis=0)) #upper bound
    plt.plot(timepoints,np.nanmean(cd_int_sens,axis=0)) #lower bound
    plt.title("Cd Int Index")
    plt.legend(['Mean','Upper','Lower'])
    plt.xlim([0.0,1.5])
    plt.savefig('/Users/drewj/Documents/millerUROP/Regr/bounds/cd_intcat'+'.png',facecolor='white')
    plt.show()

    


#########################################
########        Pooled          #########
#########################################

if plot_pooled:

   #different epochs to look at
    transient_epoch = (0,0.25)      # (start,end) of transient-response analysis epoch
    sustained_epoch = (0.25,1.5)    # (start,end) of sustained-response analysis epoch  
    windows=[transient_epoch,sustained_epoch]
    

    #extract task and area data and pass as np array
    hpc_qc=np.array(cat_dict[tasks[0]]['Hpc'])
    hpc_qwc1=np.array(cat_dict[tasks[1]]['Hpc'])
    hpc_qwc2=np.array(cat_dict[tasks[2]]['Hpc'])
    hpc_oc=np.array(cat_dict[tasks[3]]['Hpc'])
    hpc_owc1=np.array(cat_dict[tasks[4]]['Hpc'])
    hpc_owc2=np.array(cat_dict[tasks[5]]['Hpc'])


    pfc_qc=np.array(cat_dict[tasks[0]]['PFC'])
    pfc_qwc1=np.array(cat_dict[tasks[1]]['PFC'])
    pfc_qwc2=np.array(cat_dict[tasks[2]]['PFC'])
    pfc_oc=np.array(cat_dict[tasks[3]]['PFC'])
    pfc_owc1=np.array(cat_dict[tasks[4]]['PFC'])
    pfc_owc2=np.array(cat_dict[tasks[5]]['PFC'])

    cd_qc=np.array(cat_dict[tasks[0]]['Cd'])
    cd_qwc1=np.array(cat_dict[tasks[1]]['Cd'])
    cd_qwc2=np.array(cat_dict[tasks[2]]['Cd'])
    cd_oc=np.array(cat_dict[tasks[3]]['Cd'])
    cd_owc1=np.array(cat_dict[tasks[4]]['Cd'])
    cd_owc2=np.array(cat_dict[tasks[5]]['Cd'])

     #calc total and sensory info
    hpc_quad_total,hpc_quad_sens=compute_total_sensory(hpc_qc,hpc_qwc1,hpc_qwc2)
    hpc_obj_total,hpc_obj_sens=compute_total_sensory(hpc_oc,hpc_owc1,hpc_owc2)
    hpc_int_total,hpc_int_sens=compute_total_sensory(np.array(cat_dict['response']['Hpc']),*[np.array(cat_dict[term]['Hpc']) for term in interaction_terms if 'interaction' in term])


    pfc_quad_total,pfc_quad_sens=compute_total_sensory(pfc_qc,pfc_qwc1,pfc_qwc2)
    pfc_obj_total,pfc_obj_sens=compute_total_sensory(pfc_oc,pfc_owc1,pfc_owc2)
    pfc_int_total,pfc_int_sens=compute_total_sensory(np.array(cat_dict['response']['PFC']),*[np.array(cat_dict[term]['PFC']) for term in interaction_terms if 'interaction' in term])

    cd_quad_total,cd_quad_sens=compute_total_sensory(cd_qc,cd_qwc1,cd_qwc2)
    cd_obj_total,cd_obj_sens=compute_total_sensory(cd_oc,cd_owc1,cd_owc2)
    cd_int_total,cd_int_sens=compute_total_sensory(np.array(cat_dict['response']['Cd']),*[np.array(cat_dict[term]['Cd']) for term in interaction_terms if 'interaction' in term])





    #comepute categoricality
   # print(hpc_qc.shape,hpc_quad_sens.shape,hpc_quad_total.shape)
    hpc_quad_tran=compute_category_index(hpc_qc,hpc_quad_total,hpc_quad_sens,timepoints=timepoints,window=transient_epoch)
    hpc_quad_sus=compute_category_index(hpc_qc,hpc_quad_total,hpc_quad_sens,timepoints=timepoints,window=sustained_epoch)


    pfc_quad_tran=compute_category_index(pfc_qc,pfc_quad_total,pfc_quad_sens,timepoints=timepoints,window=transient_epoch)
    pfc_quad_sus=compute_category_index(pfc_qc,pfc_quad_total,pfc_quad_sens,timepoints=timepoints,window=sustained_epoch)

    cd_quad_tran=compute_category_index(cd_qc,cd_quad_total,cd_quad_sens,timepoints=timepoints,window=transient_epoch)
    cd_quad_sus=compute_category_index(cd_qc,cd_quad_total,cd_quad_sens,timepoints=timepoints,window=sustained_epoch)

    hpc_obj_tran=compute_category_index(hpc_oc,hpc_obj_total,hpc_obj_sens,timepoints=timepoints,window=transient_epoch)
    hpc_obj_sus=compute_category_index(hpc_oc,hpc_obj_total,hpc_obj_sens,timepoints=timepoints,window=sustained_epoch)

    pfc_obj_tran=compute_category_index(pfc_oc,pfc_obj_total,pfc_obj_sens,timepoints=timepoints,window=transient_epoch)
    pfc_obj_sus=compute_category_index(pfc_oc,pfc_obj_total,pfc_obj_sens,timepoints=timepoints,window=sustained_epoch)

    cd_obj_tran=compute_category_index(cd_oc,cd_obj_total,cd_obj_sens,timepoints=timepoints,window=transient_epoch)
    cd_obj_sus=compute_category_index(cd_oc,cd_obj_total,cd_obj_sens,timepoints=timepoints,window=sustained_epoch)

    hpc_int_tran=compute_category_index(np.array(cat_dict['response']['Hpc']),hpc_int_total,hpc_int_sens,timepoints=timepoints,window=transient_epoch)
    hpc_int_sus=compute_category_index(np.array(cat_dict['response']['Hpc']),hpc_int_total,hpc_int_sens,timepoints=timepoints,window=sustained_epoch)

    pfc_int_tran=compute_category_index(np.array(cat_dict['response']['PFC']),pfc_int_total,pfc_int_sens,timepoints=timepoints,window=transient_epoch)
    pfc_int_sus=compute_category_index(np.array(cat_dict['response']['PFC']),pfc_int_total,pfc_int_sens,timepoints=timepoints,window=sustained_epoch)

    cd_int_tran=compute_category_index(np.array(cat_dict['response']['Cd']),cd_int_total,cd_int_sens,timepoints=timepoints,window=transient_epoch)
    cd_int_sus=compute_category_index(np.array(cat_dict['response']['Cd']),cd_int_total,cd_int_sens,timepoints=timepoints,window=sustained_epoch)
    

    #plot each area for each time window

    #quad
    figure,axis=plt.subplots(1,2)
    
    #transient window
    axis[0].bar(0, np.nanmean(hpc_quad_tran,axis=0))
    axis[0].bar(1, np.nanmean(pfc_quad_tran,axis=0))
    axis[0].bar(2,np.nanmean(cd_quad_tran,axis=0))
    axis[0].title.set_text('Pooled Quad Index (transient)')

    #sustained window
    axis[1].bar(0, np.nanmean(hpc_quad_sus,axis=0))
    axis[1].bar(1, np.nanmean(pfc_quad_sus,axis=0))
    axis[1].bar(2,np.nanmean(cd_quad_sus,axis=0))
    axis[1].title.set_text('Pooled Quad Index (sustained)')

    axis[0].legend(['HPC','PFC','Cd'])
    axis[1].legend(['HPC','PFC','Cd'])
    # plt.xlabel("Average Pulse")
    # plt.ylabel("Calorie Burnage")
    figure.tight_layout()
    plt.savefig('/Users/drewj/Documents/millerUROP/Regr/pooled/pooled_quad_index'+'.png',facecolor='white')
    
    plt.show()

    #obj
    figure,axis=plt.subplots(1,2)
    
    #transient window
    axis[0].bar(0, np.nanmean(hpc_obj_tran,axis=0))
    axis[0].bar(1, np.nanmean(pfc_obj_tran,axis=0))
    axis[0].bar(2,np.nanmean(cd_obj_tran,axis=0))
    axis[0].title.set_text('Pooled Obj Index (transient)')

    #sustained window
    axis[1].bar(0, np.nanmean(hpc_obj_sus,axis=0))
    axis[1].bar(1, np.nanmean(pfc_obj_sus,axis=0))
    axis[1].bar(2,np.nanmean(cd_obj_sus,axis=0))
    axis[1].title.set_text('Pooled Obj Index (sustained)')

    axis[0].legend(['HPC','PFC','Cd'])
    axis[1].legend(['HPC','PFC','Cd'])
    # plt.xlabel("Average Pulse")
    # plt.ylabel("Calorie Burnage")
    figure.tight_layout()
    plt.savefig('/Users/drewj/Documents/millerUROP/Regr/pooled/pooled_obj_index'+'.png',facecolor='white')
    plt.show()


     #int
    figure,axis=plt.subplots(1,2)
    
    #transient window
    axis[0].bar(0, np.nanmean(hpc_int_tran,axis=0))
    axis[0].bar(1, np.nanmean(pfc_int_tran,axis=0))
    axis[0].bar(2,np.nanmean(cd_int_tran,axis=0))
    axis[0].title.set_text('Pooled Int Index (transient)')

    #sustained window
    axis[1].bar(0, np.nanmean(hpc_int_sus,axis=0))
    axis[1].bar(1, np.nanmean(pfc_int_sus,axis=0))
    axis[1].bar(2,np.nanmean(cd_int_sus,axis=0))
    axis[1].title.set_text('Pooled Int Index (sustained)')

    axis[0].legend(['HPC','PFC','Cd'])
    axis[1].legend(['HPC','PFC','Cd'])
    # plt.xlabel("Average Pulse")
    # plt.ylabel("Calorie Burnage")
    figure.tight_layout()
    plt.savefig('/Users/drewj/Documents/millerUROP/Regr/pooled/pooled_int_index'+'.png',facecolor='white')


#########################################
########        Sig             #########
#########################################

if test_sig:

    print('1111')

    run_sig_data=False
    save_onesamp=True
    save_ind=True

    # The scipy functions for testing that I was talking about are:
    # scipy.stats.ttest_1samp -- to test PEV for each area and task factor for significant difference from 0
    # scipy.stats.ttest_ind -- to test for differences in PEV between areas, for each task factor

    # The interface is pretty similar to what we've been working with -- 
    # just input the data (PEV across all timepoints and units in a given area) 
    # and the axis corresponding to observations (here, the units) and it should return 
    # an array of "p-values" for each time point.  We will want to convert that into binary "significance" values by calling significant, say, any timepoint with p < 0.05.

    sig_tasks=['quadrant_category','quadrant_total','object_category','object_total','interaction','response']



    if run_sig_data:
        

        #loop through data array and combine tasks for totals
        sig_data={}
     
        for task in sig_tasks:
            sig_data[task]=pickle.loads(pickle.dumps(sub_dict))

        #print(data[-0.7].keys())

        print('2')

        
        #for each area
        for ar in sig_data['interaction']:

            #for each timepoint
            for val in data:
                
                int_arr=[]
                
                
                
                for term in interaction_terms:

                    if data[val][term][ar]!=[]:

                            #np array holding all data for interaction terms
                            #int_arr=[arr[ar] for arr in data[val] if arr in interaction_terms]
                            int_arr.append(data[val][term][ar])
                           # print(int_arr)
                

                int_arr=np.array(int_arr)

                #summ terms together --> 1d array
                int_arr=np.sum(int_arr,axis=0)

                
                sig_data['interaction'][ar].append(int_arr)

        print('1')

        
        for ar in sig_data['quadrant_total']:
            #for each timepoint
            for val in data:
                
                quad_arr=[]


                for term in quad_total_terms:

                    if data[val][term][ar]!=[]:

                        #np array holding all data for interaction terms
                        #quad_arr=[arr[ar] for arr in data[val] if arr in quad_total_terms]
                        quad_arr.append(data[val][term][ar])

                # quad_arr=np.array(quad_arr)
                #print('int_arr(8d):',quad_arr.shape)

                quad_arr=np.array(quad_arr)
                #summ terms together --> 1d array
                quad_arr=np.sum(quad_arr,axis=0)

                #avg=np.mean(quad_arr)
                sig_data['quadrant_total'][ar].append(quad_arr)

        for ar in sig_data['object_total']:
            #for each timepoint
            for val in data:
                
                obj_arr=[]

                for term in obj_total_terms:

                    if data[val][term][ar]!=[]:

                        #np array holding all data for interaction terms
                    
                        obj_arr.append(data[val][term][ar])

                obj_arr=np.array(obj_arr)
                # print('obj_arr(8d):',obj_arr.shape)

                #summ terms together --> 1d array
                obj_arr=np.sum(obj_arr,axis=0)

                #avg=np.mean(obj_arr)
                sig_data['object_total'][ar].append(obj_arr)


        for task in ['quadrant_category','object_category','response']:

            for ar in sig_data[task]:
                for val in data:

                    if data[val][task][ar]!=[]:
                        sig_data[task][ar].append(data[val][task][ar])



        #print(sig_data)
        with open('sig_data.pkl', 'wb') as o:
            # Pickle dictionary using protocol 0.
            pickle.dump(sig_data, o)


    #Test PEV for each area and task for sig difference from 0

    #print(sig_data)

    if save_onesamp:
        onesamp={}
        #for timept in tqdm(sig_data):
        # print('1')
        
        for tas in sig_data:
            onesamp[tas]={}
        # print('2')
            onesamp[tas]={'Hpc':[],'PFC':[],'Cd':[]}

            for ar in sig_data[tas]:
                #print('3')
                #print(sig_data[timept][tas][ar])
                #print(sig_data[timept][tas][ar].shape)
                #print(sig_data[tas][ar])
                sig=ttest_1samp(sig_data[tas][ar],popmean=0,axis=1)[1]

                #print(sig)

                for s in sig:

                    #print(sig)
                    if s<0.01:
                        onesamp[tas][ar].append(1)
                    else:
                        onesamp[tas][ar].append(0)


        with open('onesamp.pkl', 'wb') as o:
                    # Pickle dictionary using protocol 0.
                    pickle.dump(onesamp, o)




    #print number of sig units for each task x area
    for task in onesamp:
       # print('1')
        for ar in onesamp[task]:
            sig_tot=0
            tot=0
            for unit in onesamp[task][ar]:
                #print(unit)
                if unit==1:
                    sig_tot+=1

                tot+=1


            print(task+', '+ar+': ',sig_tot,' sig timepoints, ',tot,' total timepoints')

    



    #2ND TEST: scipy.stats.ttest_ind -- to test for differences in PEV between areas, for each task factor
    area_combs=['HP','HC','PC']
    if save_ind:
        ind={}
        for task in sig_data:
            ind[task]={'HP':[],'HC':[],'PC':[]}
            
            test=[ttest_ind(sig_data[task]['Hpc'],sig_data[task]['PFC'],axis=1),
            ttest_ind(sig_data[task]['Hpc'],sig_data[task]['Cd'],axis=1),
            ttest_ind(sig_data[task]['PFC'],sig_data[task]['Cd'],axis=1)]
            print(test)

            for t in range(len(test)):
                #print(test[t])
                for i in range(len(test[t][1])):
                    if test[t][1][i]<0.01:

                        ind[task][area_combs[t]].append(1)

                    else:
                        ind[task][area_combs[t]].append(0)



        with open('ind_test.pkl', 'wb') as o:
                        # Pickle dictionary using protocol 0.
                        pickle.dump(ind, o)

            
        






# pr.disable()
# pr.print_stats(sort='time')



#17.30 sem
#18.14 cat


 # %%

# %%
