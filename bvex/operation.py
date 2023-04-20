# We don't have to declare function in excution loop, function's local parameters are initialize when function is end

import numpy as np
from skimage import measure
import cv2
import datetime


#functions
def surf_filter(im, big_thr):
 # size thresholding
    si_em = np.empty(im.shape)
    bool_em = im > 0
    total_number = None
    for sli in range(int(si_em.shape[0]/2)):
        print("Size filter processing:"+str(sli*100/im.shape[0])+"%")
        sli_em = measure.label(bool_em[sli,:,:])
        obj_number = np.amax(sli_em)
        for i in range(1,obj_number):
            id_im = sli_em == i
            x,y = np.where(id_im)
            if len(x) > big_thr:
                for j in range(len(x)):
                    si_em[sli,x[j],y [j]] = True
        total_number =+ obj_number 
            
    print("\n total object : %s" %np.amax(total_number))
    return si_em

# NEW SURFACE PROCESSING FUNCTION
def surf_section(im):
    cor_surf = np.zeros(im.shape)
    for i in range(im.shape[0]-1):
        upper = im[i,:,:]>0
        subt = np.invert(im[i+1,:,:]>0)
        mask = upper*subt
        cor_surf[i+1,:,:] = mask
    return cor_surf

def blur(im,sigma=10):
    cv2gblur_im = np.zeros(im.shape)
    for i in np.unique((np.where(im))[0]):
        sli = im[i,:,:]
        blur_sli = cv2.GaussianBlur(sli,(7,7),sigma)
        cv2gblur_im[i,:,:] = blur_sli
    return cv2gblur_im

def highpass(im, radi=3):
    lowpass_img = np.zeros(im.shape)
    x_mean,y_mean = round(im.shape[2]/2),round(im.shape[1]/2)
    for sli in range(im.shape[0]):
        fft_sli = np.fft.fft2(im[sli,:,:])
        fshift_sli = np.fft.fftshift(fft_sli)
        
        fshift_sli[x_mean-radi:x_mean+radi,y_mean-radi:y_mean+radi] = 0  

        f_ishift_sli = np.fft.ifftshift(fshift_sli)
        img_back = np.fft.ifft2(f_ishift_sli)
        real_sli = np.real(img_back)
        lowpass_img[sli,:,:] = real_sli

    return lowpass_img

def scharr(im,depth=-1):
    scharr_imx = np.zeros(im.shape)
    scharr_imy = np.zeros(im.shape)
    for i in range(im.shape[0]):
        sli = im[i,:,:]
        processed_slix = cv2.Scharr(sli,depth,0,1)
        processed_sliy = cv2.Scharr(sli,depth,1,0)
        scharr_imx[i,:,:] = np.abs(processed_slix)
        scharr_imy[i,:,:] = np.abs(processed_sliy)
    scharr_im = scharr_imx+scharr_imy
    return scharr_im

def size_thr(im, small_thr, big_thr):
    si_em = np.empty(im.shape)
    one_im = np.ones(im.shape)*im
    for sli in range(si_em.shape[0]):
        print("Size filter processing:"+str(sli*100/im.shape[0])+"%")
        sli_em = measure.label(one_im[sli,:,:])
        total = np.amax(sli_em)
        for i in range(total):
            id_im = sli_em == i+1
            x,y = np.where(id_im)
            if small_thr < len(x) and len(x) < big_thr:
                for j in range(len(x)):
                    si_em[sli,x[j],y[j]] = True
    szim = measure.label(si_em[:,:,:])
    return szim

def filler(im):
    
    fillerim = np.zeros(im.shape,dtype=bool)

    for sli in range(im.shape[0]):
        print("Filler processing:"+str(sli*100/im.shape[0])+"%")
        object2d = im[sli,:,:]
        object2d = measure.label(object2d)
        for i in np.unique(object2d):
            label2d = object2d == i
            x,y = np.where(label2d)
            uni_x,uni_y = np.unique(x),np.unique(y)

            label2dexx = np.zeros(label2d.shape) # object filled by x axis standard
            label2dexy = np.zeros(label2d.shape) # object filled by y axis standard

            for i in uni_y:
                obj1dx = np.where(label2d[:,i])

                if len(obj1dx[0])>1: # if only one pixel exist, pass
                    label2dexx[np.amin(obj1dx):np.amax(obj1dx),i] = True

            for i in uni_x:
                obj1dy = np.where(label2d[i,:])
                if len(obj1dy[0])>1:
                    label2dexy[i,np.amin(obj1dy):np.amax(obj1dy)] = True


            label2dex_xy = label2dexx*label2dexy
            holes = label2dex_xy*np.invert(label2d)
            x_hole,y_hole = np.where(holes)
            for j in range(len(x_hole)):
                fillerim[sli,x_hole[j],y_hole[j]] = True

    return fillerim

# Object z-length thresholding
def len_thr(si_em, thr):
    len_em = measure.label(si_em[:,:,:]>0)
    np.amax(si_em),np.amax(len_em)

    for i in range(np.amax(len_em)):
        obj = len_em == i+1
        z,x,y = np.where(obj)
        uni_z = np.unique(z)
        if thr > len(uni_z):
            for j in range(len(x)):
                len_em[z[j],x[j],y[j]] = 0
    result = measure.label(len_em)
    print("\ntotal object :%s"%np.amax(len_em))
    return result

def fakesurf(label_im,t_im):
    result = label_im
    for i in range(1,np.amax(label_im)+1):
        obj = label_im == i
        z,y,x = np.where(obj)
        top_z = np.amin(z)
        top_obj = obj[top_z,:,:]
        z = top_z
        if not top_z == 0:
            below_intensity = np.average(top_obj*t_im[z,:,:])
            surf_intensity = np.average(top_obj*t_im[z-1,:,:])
            while below_intensity > surf_intensity :
                z = z-1
                below_intensity = np.average(top_obj*t_im[z,:,:])
                surf_intensity = np.average(top_obj*t_im[z-1,:,:])
                result = result + obj[top_z-1,:,:]*i
                if z == 0:
                    break
    return result

# Levitated vessel thresholding MODIFIED FUNCTION!
def lev_thr(len_em,mask):
    lev_em = np.zeros(len_em.shape)
    relabel_lenem = measure.label(len_em)
    thr_mask = mask>0
    masked_em = relabel_lenem*thr_mask
    obj_list = np.unique(masked_em)
    # (obj_list, len(obj_list),np.amax(obj_list))
    for i in obj_list:
        if i == 0:
            pass
        else:
            obj = len_em == i
            lev_em = lev_em + obj
    lev_em = measure.label(lev_em)
    return lev_em

def operation_log(file_path, examiner, surf_thr, edge_size, highpass_radi, thr, len_thr, size_small, size_big):
    time = datetime.datetime.now().astimezone()
    log = open(file_path,'w')
    log.write("Time :  %s \n"%time)
    log.write("Examiner = %s \n\n"%examiner)

    log.write("Surface intensity threshold = %s \n"%surf_thr)
    log.write("Edge big threshold = %s \n"%edge_size)
    log.write("Highpass filter radius = %s \n"%highpass_radi)
    log.write("Threshold = %s \n"%thr)
    log.write("Length threhold = %s \n"%len_thr)
    log.write("Small size threshold = %s \n"%size_small)
    log.write("Big size threshold = %s \n"%size_big)
    log.close()

