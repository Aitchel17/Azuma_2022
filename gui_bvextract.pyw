import sys, os
from PyQt5 import uic
from matplotlib.figure import Figure

import numpy as np

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import bvex.operation as bvop
import tifffile



# UIfile load
# Last update 22.04.02
# GUI.UI update
uipath = "./bvex"
version = 1.02

form_class = uic.loadUiType(uipath+"/gui.ui")[0]


        
def prefunc(im,surf_thr,surf_bigthr,highpass_radi):
    surfthr_im = im > surf_thr # binary to detect surface vessel
    surf_im = bvop.surf_filter(surfthr_im, surf_bigthr) # Size threshold (basically same with "size_thr")
    if highpass_radi >0:
        highpass_im = bvop.highpass(im,radi=highpass_radi)
    else:
        highpass_im = im
    rvsurf_im = highpass_im*np.invert(surf_im>0)
    preresult = rvsurf_im # bvop.scharr(highpass_im)
    return surf_im, preresult

def afterfunc(preresult,surf_im,thr,len_thr,small_thr,big_thr):
    thr_im = preresult > thr
    size_im = bvop.size_thr(thr_im, small_thr, big_thr)

    # For levitate filter in afterprocess (inverted and)
    corsurf = bvop.surf_section(surf_im)

    #filled_im = bvop.filler(size_im)
    len_im = bvop.len_thr(size_im, len_thr)
    lev_im = bvop.lev_thr(len_im, corsurf)
    return lev_im

def savefunc(surf_im,postresult,file_path,save_path,examiner_name,surf_sbox,surfbigthr_sbox,hpassradi_sbox,thr_slider,len_sbox,small_sbox,big_sbox):
        slash_loc1 = file_path.rfind('/')
        slash_loc2 = file_path[:slash_loc1].rfind('/')
        img_name = file_path[-9:]
        file_name = file_path[slash_loc2+1:slash_loc1]        
        save_path = save_path+"/"+file_name
        os.makedirs(save_path+'/surface',exist_ok=True)
        os.makedirs(save_path+'/bvex',exist_ok=True)
        os.makedirs(save_path+'/log',exist_ok=True)
        path = save_path+'/log/'+"log_"+img_name+".txt"
        bvop.operation_log(path,examiner_name,surf_sbox,surfbigthr_sbox,hpassradi_sbox,thr_slider,len_sbox,small_sbox,big_sbox)

        surf = np.array(surf_im,dtype=np.uint16)
        result = np.array(postresult,dtype=np.uint16)

        tifffile.imwrite(save_path+'/surface/surface_'+img_name, surf, imagej=True)
        tifffile.imwrite(save_path+'/bvex/result_'+img_name, result, imagej=True)




class MainClass(QMainWindow, form_class):


    def __init__(self) :
        super().__init__()
    # ready 
        self.setupUi(self)
    # show screen
        self.setWindowTitle("BV extractor ver."+str(version))
        self.setWindowIcon(QIcon(uipath+"/icon.png"))
        self.show()
    # Parameter
        self.progressBar.setValue(0)
        self.file_path = None
        self.save_path = None
        self.preresult = None
        self.tray = None
        self.bvimg = np.zeros([512,512])
        self.img = np.zeros([512,512])
        self.tray_sli = np.zeros([512,512,3])
        self.sli = 12
        self.zmax = 0
     
    # Mpl set
        # Put layout in Mpl_widget (vertical)
        self.hbox = QHBoxLayout(self.mpl_widget)
        # Creat canvas
        self.figure = Figure(figsize=(40,40),tight_layout=True)
        self.canvas = FigureCanvas(self.figure)

        self.figure2 = Figure(figsize=(40,40),tight_layout=True)
        self.canvas2 = FigureCanvas(self.figure2)
        # Put canvas in layout
        self.hbox.addWidget(self.canvas)
        self.hbox.addWidget(self.canvas2)
    # Draw figure
        self.axes = self.figure.subplots(1,1)
        self.axes2 = self.figure2.subplots(1,1)
        self.plot_object = self.axes.imshow(self.bvimg,cmap='Reds',alpha=0.5, vmin=0, vmax=4095)
        self.plot_object2 = self.axes.imshow(self.bvimg,cmap='Greens',alpha=0.5, vmin=0, vmax=4095)

        self.plot_result = self.axes2.imshow(self.tray_sli,vmin=0, vmax=255)

    # Toolbar
        self.addToolBar(NavigationToolbar(self.canvas, self))
        self.addToolBar(NavigationToolbar(self.canvas2, self))
    # Slider
        self.sli_slider.valueChanged[int].connect(self.zchange)
        self.sli_slider.setRange(0,self.zmax)
        # Minimum bright_slider of original image
        self.minbright_slider.valueChanged[int].connect(self.thrbright)
        self.minbright_slider.setRange(0,0)
        # Maximum bright_slider of original image
        self.maxbright_slider.valueChanged[int].connect(self.thrbright)
        self.maxbright_slider.setRange(0,0)
        # Threshold of original image
        self.thr_slider.valueChanged[int].connect(self.threshold)
        self.thr_slider.setRange(0,0)


    # Spinbox
        self.minbright_sbox.setRange(0,0)
        self.thr_sbox.setRange(0,0)
        self.surfbigthr_sbox.setRange(0,9999)
        self.surfbigthr_sbox.setValue(600)
        self.big_sbox.setRange(0,9999)
        self.hpassradi_sbox.setValue(10)


    # Button
        self.load_button.clicked.connect(self.load)
        self.save_button.clicked.connect(self.save)
        self.preprocess_button.clicked.connect(self.preprocess)
        self.afterprocess_button.clicked.connect(self.afterprocess)
        self.autoapply_button.clicked.connect(self.auto)
        self.reverse_button.clicked.connect(self.reverse)
        #self.autoapply_button(self.autofunc)


    def reverse(self):
        self.bvim = self.bvim[::-1,:,:]

    def threshold(self,thr):
        img = self.img
        thr_img = img>thr
        img = img*thr_img
        self.plot_object2.set_data(img)
        self.canvas.draw_idle()
        
        rgb_sli = self.tray_sli
        rgb_sli[:,:,0] = thr_img*250
        self.plot_result.set_data(rgb_sli)
        self.canvas2.draw_idle()
  
    def thrbright(self):
        img = self.bvimg
        max = (img>self.maxbright_slider.value()-1)*4095
        pass_im = max+img*(img<self.maxbright_slider.value())
        result = pass_im*(img>self.minbright_slider.value())
        self.plot_object.set_data(result)
        self.canvas.draw_idle()
        self.canvas2.draw_idle()


    def zchange(self,z):
        self.bvimg = self.bvim[z,:,:]
        # same as thrbright
        img = self.bvimg
        max = (img>self.maxbright_slider.value()-1)*4095
        pass_im = max+img*(img<self.maxbright_slider.value())
        result = pass_im*(img>self.minbright_slider.value())

        img2 = self.preresult[z,:,:]
        self.img = img2
        result2 = img2*(img2>self.thr_slider.value())
        self.plot_object.set_data(result)
        self.plot_object2.set_data(result2)
        self.canvas.draw_idle()
        #
        self.tray_sli = self.tray[z,:,:,:]
        rgb_sli = self.tray_sli
        rgb_sli[:,:,0] = result2*250
        self.plot_result.set_data(rgb_sli)
        self.canvas2.draw_idle()


    def load(self):
        load_fileName = QFileDialog.getOpenFileName(self, 'load vessel image stacks',filter="tif file (*.tif)")
        self.file_path = load_fileName[0]
        self.load_label.setText("load:"+self.file_path)
        if self.file_path:
        # Load img
            im = tifffile.imread(self.file_path)
            if im.ndim == 3:
                self.bvim = im

            elif im.ndim == 4:
                self.bvim = im[:,0,:,:]


        # Set control parameter
            # z axis parameter 
            self.zmax = self.bvim.shape[0]
            self.sli_sbox.setMaximum(self.zmax-1)
            self.sli_slider.setMaximum(self.zmax-1)
            # Max brightness parameter and activate slider and sbox``
            maxintensity = np.amax(self.bvim)
            self.minbright_slider.setRange(0,maxintensity), self.minbright_sbox.setRange(0,maxintensity)
            self.maxbright_slider.setRange(0,maxintensity), self.maxbright_sbox.setRange(0,maxintensity),self.maxbright_sbox.setValue(maxintensity)
            self.surf_sbox.setRange(0,maxintensity), self.surf_sbox.setValue(round(maxintensity*0.5))
        # Initialize matplotlib
            self.tray = np.zeros([im.shape[0],im.shape[2],im.shape[3],3])
            self.preresult = np.zeros(self.bvim.shape)
            self.surf_im = np.zeros(self.bvim.shape)
            self.sli_sbox.setValue(self.sli)
            self.bvimg = self.bvim[self.sli,:,:]
            self.plot_object.set_data(self.bvimg)
            self.canvas.draw_idle()

            print("load_succeed")
    
    def save(self):
            self.save_path = QFileDialog.getExistingDirectory(self, 'saving location')
            self.save_label.setText("save:"+self.save_path)
            savefunc(self.surf_im,self.postresult,self.file_path,self.save_path,
            self.examiner_name.text(),self.surf_sbox.value(),self.surfbigthr_sbox.value(),
            self.hpassradi_sbox.value(), self.thr_slider.value(), self.len_sbox.value(),self.small_sbox.value(),
            self.big_sbox.value())
    
    
    def preprocess(self):
        self.surf_im, self.preresult = prefunc(self.bvim,self.surf_sbox.value(),self.surfbigthr_sbox.value(),self.hpassradi_sbox.value())
        self.tray = np.zeros([self.bvim.shape[0],self.bvim.shape[1],self.bvim.shape[2],3])
        self.tray[:,:,:,1] = self.surf_im*250

        # Draw figure, set parameter
        maxintensity = int(np.amax(self.preresult))
        self.thr_slider.setRange(0,maxintensity)
        self.thr_sbox.setRange(0,maxintensity)
        self.plot_object2.set_data(self.preresult[self.sli_sbox.value(),:,:])
        self.plot_result.set_data(self.tray[self.sli_sbox.value(),:,:,:])
        self.canvas.draw_idle()
        self.canvas2.draw_idle()
        self.thr_sbox.setValue(1300)
        self.len_sbox.setValue(3)
        self.small_sbox.setValue(16)
        self.big_sbox.setValue(1000)

    def afterprocess(self):
        self.postresult = afterfunc(self.preresult,self.surf_im,self.thr_slider.value(), self.len_sbox.value(), self.small_sbox.value(), self.big_sbox.value())

        aftershow = (self.postresult>0)*250
        self.tray[:,:,:,2] = aftershow
        self.plot_result.set_data(self.tray[self.sli_sbox.value(),:,:,:])
        self.canvas2.draw_idle()
        
        # Draw figure, set parameter
        self.plot_object.set_data(self.preresult[self.sli,:,:])
        self.canvas.draw_idle()

    def auto(self):
        load_filename = QFileDialog.getOpenFileNames(self, 'load autoappy files',filter="tif file (*.tif)")[0]
        for file_path in load_filename:
            i = 1
            bvim = tifffile.imread(file_path)[:,0,:,:]
            surf_im, preresult = prefunc(bvim,self.surf_sbox.value(),self.surfbigthr_sbox.value(),self.hpassradi_sbox.value())
            postresult = afterfunc(preresult,surf_im,self.thr_slider.value(), self.len_sbox.value(), self.small_sbox.value(),self.big_sbox.value())
            savefunc(surf_im, postresult, file_path, self.save_path,
            self.examiner_name.text(), self.surf_sbox.value(), self.surfbigthr_sbox.value(),
            self.hpassradi_sbox.value(), self.thr_slider.value(), self.len_sbox.value(),
            self.small_sbox.value(), self.big_sbox.value())
            i += 1
            self.progressBar.setValue(int(i/len(load_filename)))
            


if __name__ == "__main__" :
    app = QApplication(sys.argv) 
    window = MainClass() 
    app.exec_()
    