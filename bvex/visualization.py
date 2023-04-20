import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

def histo_plot(data_array,label_array,title_name,save_path,std=1,ymax=5,cut=35):
    totalobj_num = len(data_array) # total number of object
    length_array = [data_array[num][4][0][:cut]*100 for num in range(totalobj_num)] # %
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    stdev_plt = plt.figure(figsize=(15, 15))
    plt.title(title_name,fontsize = 30)
    plt.ylim([0,ymax])
    plt.xlim(0,max(length_data.shape[0] for length_data in length_array))
    for num in range(len(length_array)):
        x = np.arange(length_array[num].shape[0])+1
        plt.plot(x,length_array[num], label = label_array[num])
        plt.xticks(x)
    
    plt.xlabel('Depth from surface', fontsize = 25)
    plt.ylabel('Normalized count (/total vessel)', fontsize = 20)
    plt.legend(ncol=1)
    plt.rc('legend', fontsize=20)
    plt.savefig(save_path+"/"+title_name+".png",)

def graph_plot(data_array,label_array,title_name,save_path,std=1,ymax=5,cut=35):
    totalobj_num = len(data_array) # total number of object
    length_array = [data_array[num][4][0][:cut]*100 for num in range(totalobj_num)] # %
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    stdev_plt = plt.figure(figsize=(15, 15))
    plt.title(title_name,fontsize = 30)
    plt.ylim([0,ymax])
    plt.xlim(0,max(length_data.shape[0] for length_data in length_array))
    for num in range(len(length_array)):
        x = np.arange(length_array[num].shape[0])+1
        plt.plot(x,length_array[num], label = label_array[num])
        plt.xticks(x)
    
    plt.xlabel('Depth from surface', fontsize = 25)
    plt.ylabel('Percentage (/total vessel)', fontsize = 20)
    plt.legend(ncol=1)
    plt.rc('legend', fontsize=20)
    plt.savefig(save_path+"/"+title_name+".png",)

def cumul_plot(data_array,label_array,title_name,save_path,std=1,ymax=5,cut=35):
    totalobj_num = len(data_array)
    length_array = [data_array[num][4][1][:cut]*100 for num in range(totalobj_num)]
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    stdev_plt = plt.figure(figsize=(15, 15))
    plt.title(title_name,fontsize = 30)
    plt.ylim([0,ymax])
    plt.xlim(0,max(length_data.shape[0] for length_data in length_array))
    for num in range(len(length_array)):
        x = np.arange(length_array[num].shape[0])+1
        plt.plot(x,length_array[num], label = label_array[num])
        plt.xticks(x)
    
    plt.xlabel('Depth from surface', fontsize = 25)
    plt.ylabel('Cumulative count (/total vessel)', fontsize = 20)
    plt.legend(ncol=1)
    plt.rc('legend', fontsize=20)
    plt.savefig(save_path+"/"+title_name+".png",)


def stdev_plot(data_array,label_array,title_name,save_path,std=1,ymax=1.3):
    totalobj_num = len(data_array)
    ave_array = [data_array[num][0] for num in range(totalobj_num)]
    std_array = [data_array[num][1] for num in range(totalobj_num)]
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    stdev_plt = plt.figure(figsize=(15, 15))
    plt.title(title_name,fontsize = 30)
    plt.ylim([0,ymax])
    plt.xlim(0,max(ave_data.shape[0] for ave_data in ave_array))
    for num in range(len(ave_array)):
        x = np.arange(ave_array[num].shape[0])
        plt.plot(x,ave_array[num], label = label_array[num])
        plt.fill_between(x,ave_array[num]+std_array[num]*(std/2),ave_array[num]-std_array[num]*(std/2),alpha=0.3)
    
    plt.xlabel('Depth from surface', fontsize = 25)
    plt.ylabel('Normalized average signal intensity (/ top vessel)', fontsize = 20)
    plt.legend(ncol=1)
    plt.rc('legend', fontsize=20)
    plt.savefig(save_path+"/"+title_name+".png",)


def bvbar_plot(data_array,label_array,title_name,save_path,std=1,ymax=1.3, bar_width = 0.2):
    totalobj_num = len(data_array)
    min_length = min([data_array[num][0].shape[0] for num in range(totalobj_num)])
    ave_array = [data_array[num][0][:min_length] for num in range(totalobj_num)]
    std_array = [data_array[num][1][:min_length] for num in range(totalobj_num)]
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(15, 15))
    plt.title(title_name,fontsize = 30)
    plt.ylim([0,ymax])
    plt.xlim(0,max(ave_data.shape[0] for ave_data in ave_array))
    color_array=['k','r','y','r','y','g','g']
    opacity_array = [1,1,1,0.5,0.5,1,0.5,1,0.5,1,0.5]
    init_index = np.arange(ave_array[0].shape[0])
    position = 0
    for num in range(len(ave_array)):
        plt.bar(init_index+position+bar_width/2, ave_array[num], bar_width,
            alpha=opacity_array[num],
            color=color_array[num],
            yerr=std_array[num],
            label=label_array[num])
        position += bar_width

    plt.xticks(init_index+position/2,np.arange(ave_array[0].shape[0])+1)
    plt.xlabel('Depth from surface', fontsize = 20)
    plt.ylabel('Signal intensity (A.U /blood signal)', fontsize = 20)
    plt.legend(ncol=1)
    plt.rc('legend', fontsize=20)    
    plt.tight_layout()
    plt.savefig(save_path+"/"+title_name+".png",)


def visualizeObject(*im_obj,save=False): # im_obj must be (image,label number)
    for i in range(len(im_obj)):
        im, obj_number = im_obj[i]
        visual_object = im == obj_number     
        z,x,y = np.where(visual_object)
        uni_z = np.unique(z)
        z_list = uni_z
        
        xstart, ystart = np.amin(x), np.amin(y)
        xend, yend = np.amax(x), np.amax(y)
        
        fig = plt.figure(figsize=(25,25))
        fig.suptitle('Object number : %s'%obj_number, fontsize=24,color="white")
        fig.tight_layout()
        plt.subplots_adjust(left=None, bottom=0.05, right=None, top=0.95, wspace=None, hspace=None)
        
        for i in range(len(z_list)):
            ax = fig.add_subplot(6, 6, i+1)
            imgplot = plt.imshow(visual_object[z_list[i],xstart:xend,ystart:yend],cmap='inferno')
            imgplot.set_clim(0, 2)
            ax.set_title("slice no."+str(z_list[i]))
        if save:
            plt.savefig("/content/drive/My Drive/imagedata/Analysis/%s.png")

def plotimg(*images_name,area=False): #name_image must be (image,title_name), image must have z axis value
 
    if area:
        print("List of z: %s \nType the slice number"%np.unique(np.where((images_name[0])[0])[0]))
        sli = int(input())
        print("image shape :%s\n please input area of image as xstart xend ystart yend \nexample:200 500 300 700"%str((images_name[0])[0].shape))
        coordlist = [int(item) for item in input("Enter the list coordination : ").split()] 
        xstart,xend,ystart,yend = coordlist
        fig = plt.figure(figsize=(15,13),facecolor="white")
        fig.suptitle('Overall result (slice no.%s)'%sli, fontsize=24,color="black")
        plt.subplots_adjust(left=None, bottom=None, right=None, top=0.99, wspace=None, hspace=None)

        for i in range(len(images_name)):
            images,name = images_name[i]
            ax = fig.add_subplot(2, 2, i+1)
            imgplot = plt.imshow(images[sli,ystart:yend,xstart:xend],cmap='inferno')
            imgplot.set_clim(int(np.amin(images)),int(np.amax(images)))
            ax.set_title(name)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(imgplot, cax=cax)
            fig.tight_layout()
        plt.show()

    else:
        print("List of z: %s \nType the slice number"%np.unique(np.where((images_name[0])[0])[0]))
        sli = int(input())
        fig = plt.figure(figsize=(15,13),facecolor="white")
        fig.suptitle('Overall result (slice no.%s)'%sli, fontsize=24,color="black")
        plt.subplots_adjust(left=None, bottom=None, right=None, top=0.99, wspace=None, hspace=None)

        for i in range(len(images_name)):
            images,name = images_name[i]
            ax = fig.add_subplot(2, 2, i+1)
            imgplot = plt.imshow(images[sli,:,:],cmap='inferno')
            imgplot.set_clim(int(np.amin(images)),int(np.amax(images)))
            ax.set_title(name)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(imgplot, cax=cax)
            fig.tight_layout()
        plt.show()

def onesli_show(*img):
    fig = plt.figure(figsize=(40,10),facecolor='white')
    for i in range(len(img)):
        ax = fig.add_subplot(1, 4, i+1)
        imgplot = plt.imshow(img[i],cmap='inferno',vmin=0)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(imgplot, cax=cax)
    
def showsli(img,img_name="",entire=False,area=False):
    fig = plt.figure(figsize=(20,10),facecolor='white')
    fig.suptitle('image:%s'%img_name)
    fig.tight_layout()
    plt.subplots_adjust(left=None, bottom=0.05, right=None, top=0.95, wspace=None, hspace=None)
    if area:
        print("image shape :%s\n please input area of image as xstart xend ystart yend \nexample:200 500 300 700"%str((img.shape)))
        coordlist = [int(item) for item in input("Enter the list coordination : ").split()]
        xstart,xend,ystart,yend = coordlist
    else:
        coordlist = [0,img.shape[2],0,img.shape[1]]
        xstart,xend,ystart,yend = coordlist
    if entire:
        mid = int(round(img.shape[0]/2))
        end = img.shape[0]
        list_sli = [0,1,2,3,mid-2,mid-1,mid,mid+1,end-4,end-3,end-2,end-1]
        for i in range(12):
            x = list_sli[i]
            ax = fig.add_subplot(3, 4, i+1)
            imgplot = plt.imshow(img[list_sli[i],ystart:yend,xstart:xend],cmap='inferno')
            ax.set_title("slice no."+str(x))
    else:
        for i in range(12):
            ax = fig.add_subplot(3, 4, i+1)
            imgplot = plt.imshow(img[i,ystart:yend,xstart:xend],cmap='inferno')
            ax.set_title("slice no."+str(i))