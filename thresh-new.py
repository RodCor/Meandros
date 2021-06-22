import os
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from PIL import Image as Image_pil
import bezierPD
import exclusion
import zoom
import pickle
import get_profile
import landmarks
import roi_selection
import roi_detect
import pandas as pd
import group_stat
import tifffile as tif
import numpy as np
from skimage import io
from pathlib import Path
import frequency_plot as fp


class WindowAlert(Tk):
    def __init__(self, mainframe):
        ttk.Frame.__init__(self, master=mainframe)
        self.master.title('view')
        self.master.minsize(100, 100)
        self.master.geometry('200x200+500+500')


class File:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.alpha = None
        self.image = Image_pil.open(path)
        self.roi = None
        self.threshold = None
        self.curve = None
        self.exclude = None
        self.landmarks = None
        self.amp_plane = None
        self.channel = 0
        self.profile = None
        self.mask_th = []
        self.intersect_list = []


class Root(Tk):
    
    FLAG_DOUBLE_CLICK = False
    FLAG_IMAGE_LOADED = False
    FLAG_ROI_LOADED = False
    FLAG_READY_FOR_EXPORT = False
    FLAG_AXIS_SAVED = False

    def __init__(self):

        super(Root, self).__init__()
        self.wm_iconbitmap('logosysbio.ico')
        self.title('Meandros')
        self.minsize(400, 300)

        self.labelFrame0 = ttk.LabelFrame(self, text="Controls")
        self.labelFrame0.grid(row=1, column=1, padx=15, pady=15)

        self.labelFrame01 = ttk.LabelFrame(self.labelFrame0, text="Inputs")
        self.labelFrame01.grid(row=1, column=1, padx=10, pady=20)

        self.labelFrame03 = ttk.LabelFrame(self.labelFrame0, text="Threshold")
        self.labelFrame03.grid(row=2, column=1, padx=30, pady=40)

        self.labelFrame04 = ttk.LabelFrame(self, text="Display")
        self.labelFrame04.grid(row=2, column=1, padx=20, pady=20)

        self.labelFrame05 = ttk.LabelFrame(self, text="File list")
        self.labelFrame05.grid(row=1, column=3, padx=20, pady=10)

        self.labelFrame02 = ttk.LabelFrame(self.labelFrame05, text="Outputs")
        self.labelFrame02.grid(row=3, column=1, padx=20, pady=40)

        self.labelFrame07 = ttk.LabelFrame(self.labelFrame0, text="Analysis")
        self.labelFrame07.grid(row=3, column=1, padx=15, pady=15)

        self.labelFrame08 = ttk.LabelFrame(self.labelFrame0, text="Statistics")
        self.labelFrame08.grid(row=5, column=1, padx=5, pady=10)

        self.grouplist = Listbox(self.labelFrame05, selectmode=EXTENDED,
                                 width=60, height=10,
                                 highlightcolor='yellow',
                                 highlightthickness=3,
                                 selectbackground='orange')  
        self.grouplist.grid(row=1, column=1, padx=20, pady=10)
        self.grouplist.bind('<Double-Button-1>', self.doubleclick)

        self.button1()
        self.button2()
        self.button3()
        self.button4()
        self.button5()
        self.button6()
        self.button7()
        self.button8()
        self.button9()
        self.button10()
        self.button11()
        self.button12()
        self.button13()
        self.button14()
        self.button15()
        self.button16()
        self.button17()

        self.OptionList = {'RFP': 2, 'GFP': 1}
        self.variable = StringVar(self)
        self.variable.set('RFP')

        opt = OptionMenu(self.labelFrame03, self.variable, *self.OptionList.keys())
        opt.config(width=4, font=('Helvetica', 10))
        opt.grid(row=1, column=1, padx=20, pady=20)

        self.OptionList_statistic = {'Mann-Whitney': 0, 'Student': 1}
        self.variable_statistic = StringVar(self)
        self.variable_statistic.set('Mann-Whitney')

        opt_statistic = OptionMenu(self.labelFrame08, self.variable_statistic, *self.OptionList_statistic.keys())
        opt_statistic.config(width=10, font=('Helvetica', 10))
        opt_statistic.grid(row=2, column=3, padx=20, pady=20)

        self.w2 = Scale(self.labelFrame03, from_=0, to=255, tickinterval=100, orient=HORIZONTAL, length=100)
        self.w2.grid(row=1, column=2)
        self.w2.set(23)

        self.bins_label = Label(self.labelFrame08, text='bin size').grid(row=3, column=1)
        self.bins_input = Scale(self.labelFrame08, from_=0, to=100, tickinterval=50, orient=HORIZONTAL, length=100)
        self.bins_input.grid(row=3, column=2)
        self.bins_input.set(10)

        self.FLAG_FIRST_EXPORT = True
        self.curves = []  
        self.object_list = []
        self.group_1 = None
        self.group_2 = None

    def get_index(self):
        if self.grouplist.curselection():
            return self.grouplist.curselection()
        else:
            print('No element has been selected')

    def doubleclick(self, event) -> None:
        """

        :type event: object
        """
        _ = event
        try:
            index = self.get_index()[0]
            return self.display(self.object_list[index])
        except IndexError:
            msg = 'Any file loaded yet'
            raise IndexError(msg)

    def button1(self):
        button1 = ttk.Button(self.labelFrame01, text="Signal channel", command=self.filedialog1)
        button1.grid(row=1, column=1)

    def button9(self):
        button9 = ttk.Button(self.labelFrame01, text="BrightField", command=self.filedialog2)
        button9.grid(row=1, column=2)

    def button2(self):
        button2 = ttk.Button(self.labelFrame07, text="ROI detection", command=self.roi_detection)  # self.roi_detection
        button2.grid(row=1, column=1)

    def button12(self):
        button12 = ttk.Button(self.labelFrame07, text="Amputation Plane", command=self.amputation_plane)
        button12.grid(row=1, column=2)

    def button10(self):
        button10 = ttk.Button(self.labelFrame07, text="Exclude Regions", command=self.exclude_regions)
        button10.grid(row=2, column=1)

    def button11(self):
        button11 = ttk.Button(self.labelFrame07, text="Landmarks", command=self.landmarks)
        button11.grid(row=2, column=2)

    def button5(self):
        button5 = ttk.Button(self.labelFrame07, text="Draw axis", command=self.draw_axis)
        button5.grid(row=3, column=1)

    def button8(self):
        button8 = ttk.Button(self.labelFrame07, text="Get Profile", command=self.get_profile)
        button8.grid(row=3, column=2)

    def button3(self):
        button3 = ttk.Button(self.labelFrame03, text="Apply", command=self.run)
        button3.grid(row=1, column=3)

    def button4(self):
        button4 = ttk.Button(self.labelFrame02, text="Export Project", command=self.export_all)
        button4.grid(row=1, column=1)

    def button16(self):
        button16 = ttk.Button(self.labelFrame02, text="Import Project", command=self.import_all)
        button16.grid(row=1, column=2)

    def button6(self):
        button6 = ttk.Button(self.labelFrame02, text="Save axis", command=self.export_axis)
        button6.grid(row=1, column=3)

    def button7(self):
        button7 = ttk.Button(self.labelFrame05, text="Delete", command=self.delete)
        button7.grid(row=2, column=1)

    def button17(self):
        button13 = ttk.Button(self.labelFrame08, text="3D Frequency", command=self.frequency_3d)
        button13.grid(row=1, column=1)

    def button13(self):
        button13 = ttk.Button(self.labelFrame08, text="Group 1", command=self.statistic_group_1)
        button13.grid(row=2, column=1)

    def button14(self):
        button14 = ttk.Button(self.labelFrame08, text="Group 2", command=self.statistic_group_2)
        button14.grid(row=2, column=2)

    def button15(self):
        button14 = ttk.Button(self.labelFrame08, text="Apply", command=self.statistic_run)
        button14.grid(row=3, column=3)

    def delete(self) -> None:
        select_item = self.grouplist.curselection()
        for elem in reversed(select_item):
            self.grouplist.delete(elem)
            try:
                self.object_list[elem].win1.cerrar()
                self.object_list[elem].win2.cerrar()
            except AttributeError:
                pass
            del self.object_list[elem]

    def export_axis(self) -> None:
        select_item = self.grouplist.curselection()
        for elem in select_item:
            self.grouplist.itemconfig(elem, {'bg': 'yellow'})


    def display(self, choice) -> None:
        """

        :param choice: type File
        :return: None
        """
        img_l = choice.image.convert('RGB')
        img_r = choice.image.convert('RGB')
        choice.win1 = zoom.Zoom_Advanced(Toplevel(self), img_l)
        choice.win1.master.title('current view')
        choice.win2 = zoom.Zoom_Advanced(Toplevel(self), img_r)
        choice.win2.master.title('original view')

    def filedialog1(self) -> None:
        try:
            filename = filedialog.askopenfilename(initialdir="",
                                                  title='Select signal chanel',
                                                  multiple=True,
                                                  filetypes=(("tif files", "*.tif"),
                                                      ("jpg files", "*.jpg"), ("all files", "*.*")))
            if len(filename):
                for k in enumerate(filename):  # range(len(filename)):
                    finalname = os.path.splitext(os.path.split(k[1])[1])[0]
                    if k[-3:] == "tif":
                        if not os.path.exists('jpg_images'):
                            os.makedirs('jpg_images')
                        image = tif.imread(k)
                        if image.ndim > 2:
                            bright = image[1,:,:]
                            marker = image[0,:,:]
                        io.imsave(f"""jpg_images/{finalname}_brightfield.jpg""", bright)
                        io.imsave(f"""jpg_images/{finalname}_marker.jpg""", marker)
                    self.object_list.append(File(finalname, k[1]))
                    self.grouplist.insert(END, finalname)
                self.FLAG_IMAGE_LOADED = True
            else:
                return
        except AttributeError:
            pass
    
    def load_list_files(self, option_list, index):
        current_img = self.object_list[index].image.convert('RGB')
        current_obj = self.object_list[index]

        pixels = current_img.load()
        self.object_list[index].mask_th = pixels
        for i in current_obj.roi:
            x = i[1]
            y = i[0]
            if pixels[int(y), int(x)][option_list[self.variable.get()]] >= self.w2.get():
                if self.variable.get() == 'RFP':
                    pixels[int(y), int(x)] = (255, 0, 0)
                else:
                    pixels[int(y), int(x)] = (0, 255, 0)
                          
        
        current_obj.win1.image = current_img
        current_obj.threshold = self.w2.get()
        current_obj.win1.show_image()
        self.FLAG_READY_FOR_EXPORT = True

        self.grouplist.delete(index, index)
        self.grouplist.insert(index, current_obj.name + ': ' + str(self.w2.get()))
        self.object_list[index].threshold = self.w2.get()
        print("THRESHOLD Saved")
        self.object_list[index].channel = self.OptionList[self.variable.get()]
        
        return current_obj.win1

    def run(self):
        option_list = {'RFP': 0, 'GFP': 1}
        index = self.get_index()[0]

        if self.FLAG_IMAGE_LOADED and self.FLAG_ROI_LOADED:
            
            return self.load_list_files(option_list, index)
        else:
            if not self.FLAG_IMAGE_LOADED:
                print("Any image loaded yet!")
            if not self.FLAG_ROI_LOADED:
                print("You must load the ROI first")
            return

    def filedialog2(self) -> None:
        index = self.get_index()[0]

        filename = filedialog.askopenfilename(initialdir="",
                                              title='Select Bright Field',
                                              multiple=False,
                                              filetypes=(
                                                  ("tif files", "*.tif"),
                                                  ("jpg files", "*.jpg"),
                                                  ("all files", "*.*")))  # multiple=True,
        if len(filename):
            self.object_list[index].alpha = filename
        else:
            return

    def draw_axis(self):
        index = self.get_index()[0]
        filename = self.object_list[index].alpha
        axis_pd, intersect_list = bezierPD.BezierPD(filename=filename, roi=self.object_list[index].roi).run("Axis")
        self.object_list[index].intersect_list = intersect_list
        self.object_list[index].curve = set(map(tuple, axis_pd))
        if self.object_list[index].curve is not None:
            self.FLAG_AXIS_SAVED = True
            print("Axis Saved")
        return


    def exclude_regions(self):

        index = self.get_index()[0]
        filename = self.object_list[index].path
        excl = exclusion.Exclusion(filename=filename)
        polygons = excl.run("Exclude Regions")
        self.object_list[index].exclude = polygons
        if self.object_list[index].exclude is not None:
            print("Exclusion Regions Saved")

        return

    def roi_detection(self):
        index = self.get_index()[0]
        filename = self.object_list[index].alpha
        if filename[-3:] == "tif":
            image_tif = tif.imread(filename)
            if image_tif.ndim > 2:
                image_tif = image_tif[1,:,:]
            if not os.path.exists('jpg_images'):
                os.makedirs('jpg_images')
            io.imsave(f"""jpg_images//{Path(filename).name}.jpg""", image_tif)
            image_tif = f"""jpg_images//{Path(filename).name}.jpg"""
        else:
            image_tif = filename
        approx_roi = roi_detect.model_worker(image_tif)
        roi = roi_selection.RoiSelection(filename=image_tif, approx=approx_roi).run("Roi Detection")
        self.object_list[index].roi = roi
        if self.object_list[index].roi is not None:
            self.FLAG_ROI_LOADED = True
            print("ROI Saved")
        return

    def get_profile(self):
        if self.FLAG_AXIS_SAVED == True:
            index = self.get_index()[0]
            filename = self.object_list[index].path
            img = get_profile.load_exclusion(filename, self.object_list[index].exclude)
            ap = self.object_list[index].roi & self.object_list[index].amp_plane
            conj_mask = self.object_list[index].roi
            p_d = self.object_list[index].curve
            channel = self.object_list[index].channel
            threshold = self.object_list[index].threshold
            landmarks = self.object_list[index].landmarks
            self.object_list[index].profile = get_profile.analysis(p_d, conj_mask, ap, img, channel, threshold, landmarks)
            if self.object_list[index].profile is not None:
                print("Profile Saved")
        else:
            print("You need to generate the axis first")
        return

    def landmarks(self):
        index = self.get_index()[0]
        filename = self.object_list[index].alpha
        self.object_list[index].landmarks = landmarks.Landmarks(filename=filename).run("Landmarks")
        if self.object_list[index].landmarks is not None:
            print("Landmarks Saved")
        return

    def amputation_plane(self):
        index = self.get_index()[0]
        filename = self.object_list[index].alpha
        ap_plane, intersect_ap_list = bezierPD.BezierPD(filename=filename, roi=self.object_list[index].roi).run("Amputation Plane")
        self.object_list[index].amp_plane = set(map(tuple, ap_plane))
        if self.object_list[index].amp_plane is not None:
            print("Amputation Plane Saved")
        return

    def statistic_group_1(self):
        index = self.get_index()
        group = pd.concat([self.object_list[i].profile for i in index],
                          ignore_index=True,
                          sort=False).sort_values(by=['PD'],
                                                  ascending=True)
        select_item = self.grouplist.curselection()
        for elem in select_item:
            self.grouplist.itemconfig(elem, {'bg': 'green'})
        self.group_1 = group
        return

    def statistic_group_2(self):
        index = self.get_index()
        group = pd.concat([self.object_list[i].profile for i in index],
                          ignore_index=True,
                          sort=False).sort_values(by=['PD'],
                                                  ascending=True)
        select_item = self.grouplist.curselection()
        for elem in select_item:
            self.grouplist.itemconfig(elem, {'bg': 'orange'})
        self.group_2 = group
        return

    def statistic_run(self):
        if self.variable_statistic.get() == "Student":
            if self.group_1 is not None and self.group_2 is not None:
                return group_stat.Statistics(self.group_1, self.group_2, self.bins_input.get(), self.variable_statistic.get())
            else:
                print("You need to add groups first")
        elif self.variable_statistic.get() == "Mann-Whitney":
            if self.group_1 is not None and self.group_2 is not None:
                return group_stat.Statistics(self.group_1, self.group_2, self.bins_input.get(), self.variable_statistic.get())
            else:
                print("You need to add groups first")
    
    def frequency_3d(self):
        index = self.get_index()[0]
        fp.data_collection(list(self.object_list[index].curve), self.object_list[index].intersect_list, self.object_list[index].mask_th, self.object_list[index].threshold, self.object_list[index].channel)


    def export_all(self):
        try:

            saving_path = filedialog.asksaveasfile(mode='wb', initialdir="", initialfile='data',
                                                   defaultextension=".p")
            data = self.object_list 
            pickle.dump(data, open(saving_path.name, 'wb'))
        except Exception as e:
            print(e)

    
    def import_all(self):
        try:
            open_path = filedialog.askopenfilename(initialdir="", initialfile='data',
                                        defaultextension=".p")
            list_files_import = pickle.load(open(open_path, 'rb'))
            for f in list_files_import:
                
                self.object_list.append(f)
                self.grouplist.insert(END, f.name)
                if f.roi is not None:
                    self.FLAG_ROI_LOADED = True
                if f.curve is not None:
                    self.FLAG_AXIS_SAVED = True
            self.FLAG_IMAGE_LOADED = True

        except Exception as e:
            print(e)

def main():
    if os.environ.get('DISPLAY','') == '':
        print('no display found. Using :0.0')
        os.environ.__setitem__('DISPLAY', ':0.0')
    root = Root()
    root.protocol('WM_DELETE_WINDOW', root.destroy)
    root.mainloop()
    return root


if __name__ == '__main__':
    main()

