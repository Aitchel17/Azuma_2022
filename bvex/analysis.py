import pandas as pd
import tifffile
import numpy as np
from skimage import measure
import os
from tqdm import tqdm
from scipy import stats

"""

Functions for csf_analysis to extract data from the mask and image files , 


"""


def savexlsx_surface(histogram, save_folderpath, savename="csf_rawdata.xlsx"): # filtermode is exclude object with incrasing ratio
    # Create directory
    os.makedirs(save_folderpath, exist_ok=True)
    save_path = save_folderpath+"/"+savename
    # Change numpy data to
    histogram_pdframe = pd.DataFrame(histogram, index=["bins (ratio csf/bv)","count"])

    # save
    writer = pd.ExcelWriter(save_path, engine='xlsxwriter')
    histogram_pdframe.to_excel(writer, sheet_name='pixel_histogram')
   # print("Data saved in %s"%(save_path))

    writer.save()

def pensavexlsx_lengthresult(obj_data, save_folderpath, savename="csf_rawdata.xlsx"): # filtermode is exclude object with incrasing ratio
    # Create directory
    os.makedirs(save_folderpath, exist_ok=True)
    save_path = save_folderpath+"/"+savename
    # Change numpy data to
    csfdata_pdframe = pd.DataFrame(obj_data[1:,:])
    csfdata_pdframe = csfdata_pdframe.rename(columns = {
        0:"mouse",1:"tile_x",2:"tile_y", 3:"label",4:"size", 5:"depth",6:"order",7:"length",8:"BV_signal",9:"CSF_signal",10:"Ratio",11:"Background"
        })

    csfdata_pdframe.to_excel(save_path, index=False, sheet_name='raw_data')
    # Depth pivot
    csfdata_pdframe[["label","size","depth","Ratio"]]
    Size = csfdata_pdframe.pivot(index='depth',columns='label',values=['size'])
    Ratio = csfdata_pdframe.pivot(index='depth',columns='label',values=['Ratio'])
    CSF_signal = csfdata_pdframe.pivot(index='depth',columns='label',values=['CSF_signal'])
    BV_signal = csfdata_pdframe.pivot(index='depth',columns='label',values=['BV_signal'])

    # Length pivot
    comp_bv = csfdata_pdframe.pivot(index='order',columns='label',values=['BV_signal'])
    comp_ratio = csfdata_pdframe.pivot(index='order',columns='label',values=['Ratio'])
    comp_back = csfdata_pdframe.pivot(index='order',columns='label',values=['Background'])
    # Depth pivot save
    writer = pd.ExcelWriter(save_path, engine='xlsxwriter')
    csfdata_pdframe.to_excel(writer, sheet_name='raw_data')
    Size.to_excel(writer, sheet_name='Depth_Size')
    Ratio.to_excel(writer, sheet_name='Depth_Ratio')
    CSF_signal.to_excel(writer, sheet_name='Depth_CSF_signal')
    BV_signal.to_excel(writer, sheet_name='Depth_BV_signal')
    comp_bv.to_excel(writer, sheet_name='Result_BV_signal')
    comp_ratio.to_excel(writer, sheet_name='Result_Ratio')
    comp_back.to_excel(writer, sheet_name='Result_background')

    writer.save()


def total_analysis(ori_path,pro_path):
    def good_path(original_folderpath,processed_folderpath):
        original_filelist = os.listdir(original_folderpath)
        processed_filelist = os.listdir(processed_folderpath)
        match_dic = {original_folderpath+"/"+original_file:processed_folderpath+"/"+processed_file 
                    for original_file in original_filelist 
                    for processed_file in processed_filelist 
                    if original_file[-9:] == processed_file[-9:] and original_file[0]=='g'}
        return match_dic

    
    for object_name in os.listdir(pro_path):


        ## Control
        ori_filepath       = ori_path+"/"+object_name
        xlsx_save_path     = pro_path+"/"+object_name+"/"+"result"
        surf_pro_filepath  = pro_path+"/"+object_name+"/"+"surface"
        pro_filepath       = pro_path+"/"+object_name+"/"+"bvex"

        # Control surface vessel analysis
        ## Path load
        surf_path_dic = good_path(ori_filepath,surf_pro_filepath)
        path_dic = good_path(ori_filepath,pro_filepath)
        ## Surface Analysis
        surf_histogram = surf_analysis(surf_path_dic,to_surf=False) # surf_histogram

        savexlsx_surface(
            surf_histogram,
            xlsx_save_path,
            savename=str(object_name)+"surface_data.xlsx") # Conversion of data [length_bv, length_ratio]


        ## penetrating Analysis
        penetrate_result = penetrate_analysis(path_dic,surf_path_dic,to_surf=True) # result_array
    
        pensavexlsx_lengthresult(
            penetrate_result,
            xlsx_save_path,
            savename=str(object_name)+"penetrating_data.xlsx")  # Conversion of data [length_bv, length_ratio]

      
def surf_analysis(path_dic,to_surf=True):
    original_filelist = list(path_dic.keys())
    obj_data = np.zeros((1,8))
    loop_num = 0
    ratio_imgstack = None

    for ori_path in tqdm(original_filelist):  
        # print(str(round(loop_num*100/len(original_filelist)))+" %")
        mask_ori = tifffile.imread(path_dic.get(ori_path))
        bv = tifffile.imread(ori_path)[:,0,:,:]
        csf = tifffile.imread(ori_path)[:,1,:,:]
         # Ratio_img array for histogram, rounded after conversion to %
        ratio_img = np.divide(csf,bv,where= mask_ori!=0)
        if type(ratio_imgstack) != np.ndarray:
            ratio_imgstack = ratio_img
        else:
            ratio_imgstack = np.concatenate((ratio_imgstack, ratio_img), axis=0)
        
    # Histogram    
    freq_img = np.histogram(ratio_imgstack,bins=np.arange(0.0025,2.5025,0.0025))[0]
    bin_array = np.arange(0,2.5,0.0025)[1:]

    histogram = np.asarray([bin_array,freq_img])

    return histogram
    
def penetrate_analysis(penpath_dic,surfpath_dic,to_surf=True):
    original_filelist = list(penpath_dic.keys())
    
    obj_data = np.zeros((1,12))

    for ori_path in tqdm(original_filelist):
        
        namelist = ori_path.split('/')
        namelist[-1]
        tile_namex = int(namelist[-1].split('_')[1][:-4])
        tile_namey = int(namelist[-1].split('_')[0][1:])
        tile_code = (tile_namex*100+tile_namey)*10000
        img_name = namelist[-2].split('_')[0]
        penmask_ori = measure.label(tifffile.imread(penpath_dic.get(ori_path))) # get extracted labeled mask image & relabel
        backmask = penmask_ori+tifffile.imread(surfpath_dic.get(ori_path)) # Background data
        backmask = backmask == 0 # background mask
        
        img_data = tifffile.imread(ori_path)
        bv = img_data[:,0,:,:]
        csf = img_data[:,1,:,:]
        methoxy = None
        if img_data.shape[1] ==3:
            methoxy = img_data[:,2,:,:]
        background_list = np.average(csf*backmask,axis=(1,2))
        label_list = np.unique(penmask_ori)[1:] # exclude 0, always start from 0
        
        # Access to object
        for obj_no in label_list: 
            
            length_obj = 0
            obj = penmask_ori == obj_no
            z,x,y = np.where(obj)
            ori_depth_list = np.unique(z) # depth of object
            depth_list = np.unique(z) # depth of object
            
            # to surface
            if to_surf == True:
                if depth_list[0] != 0:
                    obj[depth_list[0]-1,:,:] = obj[depth_list[0],:,:]
                    z2,x,y = np.where(obj)
                else:
                    z2 = z

            depth_list = np.unique(z2) # depth of object

           # print(len(ori_depth_list),len(depth_list))
            obj_bv = bv*obj
            obj_csf = csf*obj
            if methoxy == np.ndarray :
                obj_methoxy = methoxy*obj
            # Make slice by slice profile
            for depth in depth_list: # length of object
                objz = obj[depth,:,:]
                objz_bv = obj_bv[depth,:,:]
                total_bvsingal = np.sum(objz_bv)
                objz_csf = obj_csf[depth,:,:]
                b,c = np.where(objz)
                # (0: mask_number, 1: size, 2: depth, 3:length, 4:total_length, 5:average BV_intensity, 6:average csf_intensity, 7: ratio)

                obj_number = obj_no
                size = len(b)
                depth = depth
                length_obj = length_obj
                total_length = len(depth_list)
                ave_bvsignal = total_bvsingal/size
                ave_csfsignal = np.sum(objz_csf)/size
                ratio = ave_csfsignal/ave_bvsignal
                backgroundratio = background_list[depth]
                data_array = np.array([[img_name,tile_namex,tile_namey,tile_code+obj_number,size,depth,length_obj,total_length,ave_bvsignal,ave_csfsignal,ratio,backgroundratio]],dtype = float)
                obj_data = np.concatenate(((obj_data,data_array)),axis=0) # add array
                length_obj = length_obj+1 # fix

        obj_data = obj_data[:,:]


    return obj_data



"""
load data

"""

def load_analysis(pro_path, userobjlist = [], side='none',len_thr=3):
    # determine ipsi or cont
    if side == 'ipsi':
        objectlist = [object for object in os.listdir(pro_path) if object[-4:] == 'ipsi']
        if len(userobjlist)>0:
            objectlist = userobjlist
    elif side == 'cont':
        objectlist = [object for object in os.listdir(pro_path) if object[-4:] == 'cont']
        if len(userobjlist)>0:
            objectlist = userobjlist

    else :
        objectlist = os.listdir(pro_path)
        if len(userobjlist)>0:
            objectlist = userobjlist

    surf_histogram_list = []
    total_rawdata = []

    total_penetrating_ratio = np.zeros((50,100000))
    total_penetrating_ratio[:,:] = np.NaN

    total_penetrating_blood = np.zeros((50,100000))
    total_penetrating_blood[:,:] = np.NaN
    start = 0
    end = 0
        
    for object_name in tqdm(objectlist):

        pro_filepath = pro_path+"/"+object_name+"/"+"result"

        # Load sheet
        raw_data = pd.read_excel(
            pro_filepath+"/" +str(object_name)+"penetrating_data.xlsx",sheet_name='raw_data',index_col=0
            )
        total_rawdata.append(raw_data)
               
        surface_histogram = pd.read_excel(
            pro_filepath+"/" +str(object_name)+"surface_data.xlsx",sheet_name='pixel_histogram'
            ).to_numpy()[:,:]
        surf_histogram_list.append(surface_histogram)

    pd_rawdata = pd.concat(total_rawdata,ignore_index=True)

    combined_label = pd_rawdata['mouse']*10000000000+pd_rawdata['label']
    pd_rawdata = pd_rawdata.assign(id=combined_label)
    output = pd_rawdata
    
    # return penetrating_ratio, surf_histogram_list # [average, standard deviation, total object number,]
    return output, surf_histogram_list







def enlarge(im,ex_percent):
    magnify = 50
    mask = np.zeros((im.shape[0],im.shape[1]+2*magnify,im.shape[2]+2*magnify),dtype=np.int64)
    mask[:,magnify:mask.shape[1]-magnify,magnify:mask.shape[2]-magnify] = im
    nullarray = np.zeros((im.shape[0],im.shape[1]+2*magnify,im.shape[2]+2*magnify),dtype=np.int64)

    for i in range(np.amax(im)):
        obj = mask == i+1
        z,xii,yii = np.where(obj)

        for depth in range(len(np.unique(z))):
                obz = obj[np.unique(z)[depth],:,:]
                obz_label = measure.label(obz)
    # seperate obj in one slice            
                for o in range(np.amax(obz_label)):
                    obzl = obz_label == o+1
                    x,y = np.where(obzl)
                    si_obzl = len(x)*ex_percent
                    peri_obzl = measure.perimeter(obzl, neighbourhood=4)
                    if peri_obzl > 0:
                        expixel = int(round(si_obzl/peri_obzl))

                        for k in range(len(np.unique(y))):
                            obx = obzl[:,np.unique(y)[k]]
                            xi = np.where(obx)
                            mxi , nxi = np.amax(xi), np.amin(xi)
                            addcord = np.arange(expixel) + 1
                            admx = addcord+np.amax(xi)
                            adnx = -1*addcord+np.amin(xi)
                            total = np.append(adnx,admx)
                            for l in range(len(total)):
                                mask[np.unique(z)[depth],total[l],np.unique(y)[k]] = i+1

                        for k in range(len(np.unique(x))):
                            oby = obzl[np.unique(x)[k],:]
                            yi = np.where(oby)
                            myi , nyi = np.amax(yi),np.amin(yi)
                            addcord = np.arange(expixel) + 1
                            admy = addcord+np.amax(yi)
                            adny = -1*addcord+np.amin(yi)
                            total = np.append(adny,admy)
                            for l in range(len(total)):
                                mask[np.unique(z)[depth],np.unique(x)[k],total[l]] = i+1
                    else:
                        for m in range(len(x)):
                            mask[np.unique(z)[depth],x[m],y[m]] = i+1 
    mask_ori = mask[:,magnify:mask.shape[1]-magnify,magnify:mask.shape[2]-magnify]
    return mask_ori