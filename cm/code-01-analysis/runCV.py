import os
import utilsCM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def runCV_execute(pretrained_val,savepath,Ypredict='Word2Sense',keyword={'DNNActvtn','ROIpred'},datapath='../../../data-00/',layer={'conv_1','conv_5','fc_3'},ROI={'EVC','ObjectROI'},Sub=[1,2,3,4],Keepncomps=list(range(2,42,2))):
    
    ### Subset of things info
    
    WIpath = '../../../data-04/'
    nsample = 12
    WrdThingsInfo = pd.read_csv(WIpath + 'KeptTHINGSInfo_n' + str(nsample) +'.csv',sep=',',index_col = 0)
    

    if Ypredict is 'Word2Vec':
        ### Load Word2Vec subset
        filename = 'ThingsWrd2Vec_subset.txt'
        filepath = '../../../data-10/'
        Wrd2Vec = pd.read_csv(filepath + filename,sep=',',index_col = 0)
        Y_embeddings_subset = Wrd2Vec.values[:,:].astype(np.float)
    elif Ypredict is 'Word2Sense':
        ### Load Word2Sense subset
        pathtofile = '../../../data-07/'
        Y_embeddings_subset = pd.read_csv(pathtofile + "ThingsWrd2Sns_subset.txt", sep=",",index_col = 0)
        Y_embeddings_subset = Y_embeddings_subset.values[:,:].astype(np.float)

    for ikeyword in keyword:
        for ilayer in layer:
            if ikeyword is 'ROIpred':
                for iROI in ROI: 
                    savefigname = 'Predict' + Ypredict + '_' + ikeyword + '_' +iROI + '_'+ ilayer
            
                    for iSub in Sub:
                        Subfile = datapath +  "ROIpred_Sub" + str(iSub) + '_' + iROI + "_" + ilayer 

                        if not pretrained_val:
                            Subfile = Subfile + '_untrained'

                        
                        thisSub = np.load(Subfile + '.npy')
                        
                        thisSub = thisSub[WrdThingsInfo['old_index']]
                        
                        if iSub is 1:
                            predictor_variable_sub = thisSub
                        else:
                            predictor_variable_sub = np.append( predictor_variable_sub , thisSub, axis = 1)

                    print(iROI, predictor_variable_sub.shape)

                    iterOverPCs(Keepncomps,pretrained_val,savepath,predictor_variable_sub,Y_embeddings_subset,ikeyword,ilayer,Ypredict,iROI)

            elif ikeyword is 'DNNActvtn':

                predictor_variable_file = datapath +  "things_" + ilayer 
                if not pretrained_val:
                    predictor_variable_file = predictor_variable_file + '_untrained'

                
                predictor_variable_sub = pd.read_csv(predictor_variable_file + '.csv', header=None, index_col=0).iloc[:,:].to_numpy()            
                predictor_variable_sub = predictor_variable_sub[WrdThingsInfo['old_index']]


                iterOverPCs(Keepncomps,pretrained_val,savepath,predictor_variable_sub,Y_embeddings_subset,ikeyword,ilayer,Ypredict)



def iterOverPCs(Keepncomps,pretrained_val,savepath,predictor_variable_sub,Y_embeddings_subset,ikeyword,ilayer,Ypredict,iROI=None):
    for icomps in Keepncomps:
        if iROI is None:
            savefilename = 'Predict' + Ypredict + '_'  + ikeyword + '_'+ ilayer + '_'+ str(icomps) +'PCs'
        else:
            savefilename = 'Predict' + Ypredict + '_' + ikeyword + '_' +iROI + '_'+ ilayer + '_'+ str(icomps) +'PCs'
        print(savefilename)
        if not pretrained_val:
            savefilename = savefilename+'_untrained'
        if not os.path.isfile(savepath + savefilename + '.npy'):
            utilsCM.iter_cvregress_SaveInfo(predictor_variable_sub,Y_embeddings_subset,ilayer,icomps,iROI,saveinfo = savepath + savefilename)        



def buildDict(datapath,figurepath,Ypredict='Word2Sense',setbonf = True, keyword={'ROIpred','DNNActvtn'},layer={'conv_5'},ROI={'EVC','ObjectROI'},Sub=[1,2,3,4],Keepncomps=list(range(2,42,2)),pretrained_val = True,RandomWs=False):
    if Ypredict is 'Word2Sense':
        ### Load Word2Vec subset
        filename = 'ThingsWrd2Vec_subset.txt'
        filepath = '../../../data-10/'
        Wrd2Vec = pd.read_csv(filepath + filename,sep=',',index_col = 0)
        Y_embeddings_subset = Wrd2Vec.values[:,:].astype(np.float)
        Y_embeddings_subset
    elif Ypredict is 'Word2Vec':
        ### Load Word2Sense subset
        pathtofile = '../../../data-07/'
        Y_embeddings_subset = pd.read_csv(pathtofile + "ThingsWrd2Sns_subset.txt", sep=",",index_col = 0)

    if setbonf:
        tresh_bonf = utilsCM.p2r(.05/Y_embeddings_subset.shape[1], Y_embeddings_subset.shape[0])
    else:
        tresh_bonf = 0

    myDict_count = {}
    myDict_mean = {}
    myDict_max = {}
    myDict_median = {}

    RandomWs_val = RandomWs

    for ilayer in layer:
        for ikeyword in keyword:            
            for icomps in Keepncomps:
                thisPrediction = []               
                if ikeyword is 'DNNActvtn':
                    filename = 'Predict' + Ypredict + '_' + ikeyword + '_'+ ilayer + '_'+ str(icomps) +'PCs'
                    DictKey = ikeyword
                    filename, DictKey = completeName(pretrained_val,filename,DictKey)
                    myDict_count,myDict_mean,myDict_max,myDict_median = buldDict_execute(datapath,DictKey,filename,tresh_bonf,myDict_count,myDict_mean,myDict_max,myDict_median)  
                    
                elif ikeyword is 'ROIpred':  
                    # for RandomWs_val in RandomWs:  

                    for iROI in ROI:
                        filename = 'Predict' + Ypredict + '_' + ikeyword + '_' +iROI + '_'+ ilayer + '_'+ str(icomps) +'PCs'
                        DictKey = iROI
                        filename, DictKey = completeName(pretrained_val,filename,DictKey,RandomWs_val)
                        myDict_count,myDict_mean,myDict_max,myDict_median = buldDict_execute(datapath,DictKey,filename,tresh_bonf,myDict_count,myDict_mean,myDict_max,myDict_median)
        
    myDict_count['PCs'] = []
    myDict_mean['PCs'] = []
    myDict_max['PCs'] = []
    myDict_median['PCs'] = []
    myDict_count['Metric'] = []
    myDict_mean['Metric'] = []
    myDict_max['Metric'] = []
    myDict_median['Metric'] = []
    for i in Keepncomps:
        myDict_count['PCs'].append(i)
        myDict_mean['PCs'].append(i)
        myDict_max['PCs'].append(i)
        myDict_median['PCs'].append(i)
        myDict_count['Metric'].append('count')
        myDict_mean['Metric'].append('mean')
        myDict_max['Metric'].append('max')
        myDict_median['Metric'].append('median')

    return myDict_median,myDict_max,myDict_count,myDict_mean



def buldDict_execute(datapath,DictKey,filename,tresh_bonf,myDict_count,myDict_mean,myDict_max,myDict_median):
    if DictKey not in myDict_count:
        myDict_count[DictKey] = []
        myDict_mean[DictKey] = []
        myDict_max[DictKey] = []
        myDict_median[DictKey] = []
    thisPrediction = np.load(datapath + filename + '.npy')
    # print(thisPrediction.shape)
    pred_thresh = thisPrediction[thisPrediction>tresh_bonf]
    if not pred_thresh.any() or len(pred_thresh)<=2:
        myDict_max[DictKey].append(0)
        myDict_mean[DictKey].append(0)
        myDict_median[DictKey].append(0)
    else:
        myDict_max[DictKey].append(pred_thresh.max())
        myDict_mean[DictKey].append(pred_thresh.mean())
        myDict_median[DictKey].append(np.median(pred_thresh))                        
                        
    myDict_count[DictKey].append(pred_thresh.shape[0])
    # myDict_median[DictKey].append(np.median(pred_thresh))

    return myDict_count,myDict_mean,myDict_max,myDict_median


def completeName(pretrained_val,filename,DictKey,RandomWs_val = None):
    if not pretrained_val:
        filename = filename + '_untrained'
        DictKey = DictKey + '_untrained'
    if RandomWs_val:
        filename = filename + '_random'
        DictKey = DictKey + '_random'
    return filename,DictKey


