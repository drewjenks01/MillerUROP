
import cProfile
from os import walk
from contextlib import contextmanager
import sys, os
import neural_analysis.spikes as spk
from neural_analysis.info import neural_info
from neural_analysis.matIO import loadmat
from neural_analysis.utils import unsorted_unique
import numpy as np
from tqdm import tqdm
import pickle
import xarray as xr

#%%
"""

Calculates PEV for each neuron in each file.

Creates cat_dict and/or data and saves to pickle files to use in analysis.

"""

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

#store the data to be used faster for plotting
save_sem=True
save_cat=True
save_timepts=False


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

#which kind of method being used
#if T, bin used, else density
bin_meth=False


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

quad_tasks=['quadrant_category','Qaudrant_scrambled']
obj_tasks=['object_category','Object_scrambled']

#CAT data dict
#data categorized by task and area, but not combined in any way
cat_dict={}
for task in tasks:
    cat_dict[task]=pickle.loads(pickle.dumps(sub_dict))



#########################################
#######       Design Matrix        ######
#########################################

def set_design_matrix(trial_info, design='allTerms'):
    """
    Sets up design matrix for regression analysis for contextAssoc dataset

    ARGS
    trial_info  (n_trials,n_cols) DataFrame | Dict w/ (n_trials,) values.
                Table-like data structure with per-trial task and behavioral information

    design      String. Determines exact type of design matrix. Default: 'allTerms'
                'nested'    : Stimulus terms are set to be nested within categorical factor
                'allcombs   : Terms with all combinations of stimulus conditions are included
                'allTerms'  : 'allCombs' design, but returns PEV of each individual term (no pooling of same-type terms)
                'pooled'    : All quadrant, all object terms are set so their explained variance is pooled together
    """
    design = design.lower()
    assert design in ['nested','allcombs','allterms','pooled'], \
        ValueError("Unsupported value '%s' set for design" % design)

    n_trials = len(trial_info['sample'])
    n_terms = 16

    design_matrix = np.zeros((n_trials,n_terms))
    design_labels = np.zeros((n_terms,), dtype='object')

    # Quadrant "bewteen-category" information (reflects quadrants that have same response mapping in task -- UR&LL vs UL&LR)
    design_labels[0] = 'quadrant_category'
    design_matrix[:,0] = ((trial_info['quadrant'] == 'upper-right') | (trial_info['quadrant'] == 'lower-left')).astype(float) - \
                         ((trial_info['quadrant'] == 'upper-left')  | (trial_info['quadrant'] == 'lower-right')).astype(float)
    if design == 'nested':
        # Quadrant stimulus information (reflects differences *btwn* quadrants with same response mapping -- URvsLL, ULvsLR)
        design_labels[1:2+1] = 'quadrant_stimulus'
        design_matrix[:,1] = (trial_info['quadrant'] == 'upper-right').astype(float) - (trial_info['quadrant'] == 'lower-left').astype(float)
        design_matrix[:,2] = (trial_info['quadrant'] == 'upper-left').astype(float) - (trial_info['quadrant'] == 'lower-right').astype(float)
    else:
        # Quadrant "within-category" stimulus information
        # All other (non-task-relevant) paired groupings of quadrants (UR&UL vs LL&LR [upper vs lower], UR&LR vs UL&LL [right vs left])
        design_matrix[:,1] = ((trial_info['quadrant'] == 'upper-right') | (trial_info['quadrant'] == 'upper-left')).astype(float) - \
                             ((trial_info['quadrant'] == 'lower-left')  | (trial_info['quadrant'] == 'lower-right')).astype(float)
        design_matrix[:,2] = ((trial_info['quadrant'] == 'upper-right') | (trial_info['quadrant'] == 'lower-right')).astype(float) - \
                             ((trial_info['quadrant'] == 'upper-left')  | (trial_info['quadrant'] == 'lower-left')).astype(float)
        if design == 'allterms':
            design_labels[1]        = 'quadrant_withinCategory1'
            design_labels[2]        = 'quadrant_withinCategory2'
        elif design == 'allcCombs':
            design_labels[1:2+1]    = 'quadrant_withinCategory'

    # Sample object "between-category" information (reflects objects that have same response mapping in task -- 2&3 vs 1&4)
    design_labels[3] = 'object_category'
    design_matrix[:,3] = ((trial_info['sample'] == 2) | (trial_info['sample'] == 3)).astype(float) - \
                         ((trial_info['sample'] == 1) | (trial_info['sample'] == 4)).astype(float)
    if design == 'nested':
        # Sample object stimulus information (reflects differences *btwn* objects with same response mapping -- 2vs3, 1vs4)
        design_labels[4:5+1] = 'object_stimulus'
        design_matrix[:,4] = (trial_info['sample'] == 2).astype(float) - (trial_info['sample'] == 3).astype(float)
        design_matrix[:,5] = (trial_info['sample'] == 1).astype(float) - (trial_info['sample'] == 4).astype(float)
    else:
        # Sample object "within-category" stimulus information
        # All other (non-task-relevant) paired groupings of sample objects (1&2 vs 3&4, 1&3 vs 2&4)
        design_matrix[:,4] = ((trial_info['sample'] == 1) | (trial_info['sample'] == 2)).astype(float) - \
                             ((trial_info['sample'] == 3) | (trial_info['sample'] == 4)).astype(float)
        design_matrix[:,5] = ((trial_info['sample'] == 1) | (trial_info['sample'] == 3)).astype(float) - \
                             ((trial_info['sample'] == 2) | (trial_info['sample'] == 4)).astype(float)
        if design == 'allterms':
            design_labels[4]        = 'object_withinCategory1'
            design_labels[5]        = 'object_withinCategory2'
        elif design == 'allcombs':
            design_labels[4:5+1]    = 'object_withinCategory'
            
    # For 'pooled' design, set labels so all quadrant, all object terms are pooled together
    if design == 'pooled':
        design_labels[0:2+1] = 'quadrant'
        design_labels[3:5+1] = 'object'

    # Expected response direction (left vs right) = interaction btwn terms 0 & 3
    design_labels[6] = 'response'
    design_matrix[:,6] = (trial_info['expectedResponse'] == 'right').astype(float) - (trial_info['expectedResponse'] == 'left').astype(float)
    assert np.array_equal(design_matrix[:,6], design_matrix[:,0]*design_matrix[:,3])

    design_matrix[:,7]  = design_matrix[:,0] * design_matrix[:,4]   # Interaction btwn terms 0 & 4
    design_matrix[:,8]  = design_matrix[:,0] * design_matrix[:,5]   # Interaction btwn terms 0 & 5
    design_matrix[:,9]  = design_matrix[:,1] * design_matrix[:,3]   # Interaction btwn terms 1 & 3
    design_matrix[:,10] = design_matrix[:,1] * design_matrix[:,4]   # Interaction btwn terms 1 & 4
    design_matrix[:,11] = design_matrix[:,1] * design_matrix[:,5]   # Interaction btwn terms 1 & 5
    design_matrix[:,12] = design_matrix[:,2] * design_matrix[:,3]   # Interaction btwn terms 2 & 3
    design_matrix[:,13] = design_matrix[:,2] * design_matrix[:,4]   # Interaction btwn terms 2 & 4
    design_matrix[:,14] = design_matrix[:,2] * design_matrix[:,5]   # Interaction btwn terms 2 & 5
    if design == 'allterms':
        design_labels[7]    = 'interaction_0&4'
        design_labels[8]    = 'interaction_0&5'
        design_labels[9]    = 'interaction_1&3'
        design_labels[10]   = 'interaction_1&4'
        design_labels[11]   = 'interaction_1&5'
        design_labels[12]   = 'interaction_2&2'
        design_labels[13]   = 'interaction_2&4'
        design_labels[14]   = 'interaction_2&5'        
    else:
        design_labels[7:-1] = 'interaction'
        
    # Constant (aka intercept) term -- all 1's
    design_labels[15] = 'intercept'
    design_matrix[:,15] = np.ones((n_trials,))

    return design_matrix, design_labels

    

#########################################
#####   Load Data From File      ########
#########################################

#list of all files
filenames = next(walk('/Users/drewj/Documents/millerUROP/millerdata/'), (None, None, []))[2]  # [] if no file
filenames=filenames[1:]         #took out contextAss.mat file


#########################################
#######        Main Loop        #########
#########################################

#area count
if Hpc_together:
    area_count={'Hpc':0,'PFC':0,'Cd':0}
else:
    area_count={'Hpc':0,'PFC':0,'Cd':0,'CA1':0,'CA2':0,'CA3':0,'CA4':0}

#dict for timepoint data
data={}

#var to check if first file
first=True

#loop through each file
for file in tqdm(filenames):

    #file
    filename = r'/Users/drewj/Documents/millerUROP/millerdata/'+file

    #load data, get spike times, trial info, unit info
    with suppress_stdout():
        trial_infor,spike_times, unit_info = loadmat(filename, variables=['trialInfo','spikeTimes','unitInfo'])

   # print(trial_infor['quadrant'],len(trial_infor['quadrant']))

    with suppress_stdout():

        #use rate() to find spike rate and rate bins
        if bin_meth:

            spike_rate, rate_bins = spk.rate(spike_times, method='bin', lims=[-0.7,1.7], width=50e-3, step=10e-3)
            timepoints = np.mean(rate_bins,axis=1)    #represents timepoints

        else:

            spike_rate, rate_bins = spk.rate(spike_times, method='density', lims=[-0.7,1.7], width=20e-3,buffer=0,kernel='gaussian')
            timepoints= rate_bins   #represents timepoints  
            if save_timepts:
                with open('timepoints.pkl', 'wb') as o:
                    # Pickle dictionary using protocol 0.
                    pickle.dump(timepoints, o)
                break


  
    #set up data dict if first file only
    if first:
        if design_method=='pooled':
            for val in timepoints:
                    data[val]={'Quadrant':pickle.loads(pickle.dumps(sub_dict)),'Object':pickle.loads(pickle.dumps(sub_dict)),
                    'Expected_response':pickle.loads(pickle.dumps(sub_dict)),'Interaction':pickle.loads(pickle.dumps(sub_dict))}

        elif design_method=='nested':
            for val in timepoints:
                    data[val]={'quadrant_category':pickle.loads(pickle.dumps(sub_dict)),'Quadrant_stimulus':pickle.loads(pickle.dumps(sub_dict)),
                    'object_category':pickle.loads(pickle.dumps(sub_dict)),'Object_stimulus':pickle.loads(pickle.dumps(sub_dict)),
                    'Expected_response':pickle.loads(pickle.dumps(sub_dict)),'Interactions':pickle.loads(pickle.dumps(sub_dict))}

        elif design_method=='allcombs':
            for val in timepoints:
                    data[val]={'quadrant_category':pickle.loads(pickle.dumps(sub_dict)),'Quadrant_scrambled':pickle.loads(pickle.dumps(sub_dict)),
                    'object_category':pickle.loads(pickle.dumps(sub_dict)),'Object_scrambled':pickle.loads(pickle.dumps(sub_dict)),
                    'Expected_response':pickle.loads(pickle.dumps(sub_dict)),'Interaction':pickle.loads(pickle.dumps(sub_dict))}

        elif design_method=='allterms':
            for val in timepoints:
                    data[val]={'quadrant_category':pickle.loads(pickle.dumps(sub_dict)),'quadrant_withinCategory1':pickle.loads(pickle.dumps(sub_dict)),
                    'quadrant_withinCategory2':pickle.loads(pickle.dumps(sub_dict)) ,'object_category':pickle.loads(pickle.dumps(sub_dict)),
                    'object_withinCategory1':pickle.loads(pickle.dumps(sub_dict)),'object_withinCategory2':pickle.loads(pickle.dumps(sub_dict)),
                    'response':pickle.loads(pickle.dumps(sub_dict)),'interaction_0&4':pickle.loads(pickle.dumps(sub_dict)),
                    'interaction_0&5':pickle.loads(pickle.dumps(sub_dict)),'interaction_1&3':pickle.loads(pickle.dumps(sub_dict)),
                    'interaction_1&4':pickle.loads(pickle.dumps(sub_dict)),'interaction_1&5':pickle.loads(pickle.dumps(sub_dict)),
                    'interaction_2&2':pickle.loads(pickle.dumps(sub_dict)),'interaction_2&4':pickle.loads(pickle.dumps(sub_dict)),'interaction_2&5':pickle.loads(pickle.dumps(sub_dict))}
        first=False

    #use neural_info to calculate pev

    with suppress_stdout():
        ### Compute multi-factor (multiple regression) information analysis ###

        if design_method=='pooled':
            design_pooled, labels_pooled = set_design_matrix(trial_infor, design='pooled')
            terms_pooled = unsorted_unique(labels_pooled)
            pev= neural_info(design_pooled, spike_rate, axis=0, method='pev',
                                    model='regress', col_terms=labels_pooled)
        
        elif design_method=='allcombs':
            design_allcombs, labels_allcombs = set_design_matrix(trial_infor, design='allcombs')
            terms_allcombs = unsorted_unique(labels_allcombs)
            pev= neural_info(design_allcombs, spike_rate, axis=0, method='pev',
                                    model='regress', col_terms=labels_allcombs)

        elif design_method=='allterms':
            design, labels = set_design_matrix(trial_infor, design='allTerms')
            pev = neural_info(design, spike_rate, axis=0, method='pev',
                            model='regress', col_terms=labels)

            # # Step 1a: Put pev into xarray DataArray (labeled ndarray) to make it easier to do further analysis on 
            # pev = xr.DataArray(pev, dims=['term','channel','time'],
            #                 coords={'term':labels[:-1], 'channel':np.arange(pev.shape[1]), 'time':timepoints})

            # ## Step 2: Pool PEV (information) into summary variables of interest (total/sensory, all interactions)
            # labels_pooled = ['quadrant_category','quadrant_total','quadrant_sensory',
            #                 'object_category','object_total','object_sensory',
            #                 'response','interaction']
            # pev_pooled = xr.DataArray(np.empty((len(labels_pooled),pev.shape[1],pev.shape[2])),
            #                         dims=['term','channel','time'],
            #                         coords={'term':labels_pooled, 'channel':np.arange(pev.shape[1]), 'time':timepoints})
            # # These variables are just copied as-is
            # pev_pooled.loc[['quadrant_category','object_category','response'],...] = \
            #     pev.loc[['quadrant_category','object_category','response'],...]


        
    #########################################
    ###   Split pev up by area & timept   ###
    #########################################

    #########################################
    ########        SEM             #########
    #########################################
    
    if save_sem:
        #loop through areas to get area counts
        for i in range(len(unit_info['area'])):
            
            #area
            area=unit_info['area'][i]
                
            #if a valid area
            if area in areas:

                #check if Hpc
                if area in Hpcs:

                    #update count
                    area_count['Hpc']+=1


                    #loop through tasks
                    for t in range(len(tasks)):
                        task=tasks[t]
                        pev_task=pev[t][i]

                        #loop through timepoints
                        for r in range(len(timepoints)):
                            #add spike data based on timpoint, task, and area
                            data[timepoints[r]][task]['Hpc'].append(pev_task[r])


                #do same but for other areas
                else:
                    #update count
                    area_count[area]+=1
                    

                    #loop through tasks
                    for t in range(len(tasks)):
                        task=tasks[t]
                        pev_task=pev[t][i]

                        #loop through timepoints
                        for r in range(len(timepoints)):
                            #add spike data based on timpoint, task, and area
                            data[timepoints[r]][task][area].append(pev_task[r])


    #########################################
    ########        CAT             #########
    #########################################
    
    #see if any neurons should be removed, if so, just skip the area from pot_area
    if save_cat:


        for i in range(len(unit_info['area'])):
            
            #area
            area=unit_info['area'][i]
                
            #if a valid area
            if area in areas:

                #check if Hpc
                if area in Hpcs:

                    #update count
                    if not save_sem:
                        area_count['Hpc']+=1


                    #loop through tasks
                    for t in range(len(tasks)):
                        task=tasks[t]

                        pev_task=pev[t][i]

                        #add neuron to specific task and area slot in cat dict as np array
                        cat_dict[task]['Hpc'].append(np.array(pev_task))

                        


                #do same but for other areas
                else:
                    #update count
                    if not save_sem:
                        area_count['Hpc']+=1
                    

                    #loop through tasks
                    for t in range(len(tasks)):
                        task=tasks[t]
                        pev_task=pev[t][i]

                        #add neuron to specific task and area slot in cat dict as np array
                        cat_dict[task][area].append(np.array(pev_task))

       

#sem
if save_sem:
    with open('sem_data.pkl', 'wb') as output:
        # Pickle dictionary using protocol 0.
        pickle.dump(data, output)


#cat
if save_cat:
    with open('cat_dict.pkl', 'wb') as out:
        # Pickle dictionary using protocol 0.
        pickle.dump(cat_dict, out)

if save_timepts:
    with open('timepoints.pkl', 'wb') as o:
        # Pickle dictionary using protocol 0.
        pickle.dump(timepoints, o)

# %%
