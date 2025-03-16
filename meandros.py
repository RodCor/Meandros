import copy
import os
from tkinter import *
from tkinter import ttk, filedialog, messagebox
from PIL import Image as Image_pil, ImageTk
from generals import bezierPD
import exclusion_module.exclusion as exclusion
from generals.zoom import Zoom_Advanced
import pickle
import profile_module.get_profile as get_profile
import landmarks_module.landmarks as landmarks
from roi_detection_module import roi_selection, roi_detect
import pandas as pd
import statistics_module.group_stat as group_stat
import tifffile as tif
from skimage import io
from pathlib import Path
import axis_detection_module.axis_approx as axis_approx
from roi_detection_module.config import Config
import roi_detection_module.model as modellib
import reports_sd.sd_plots as sdp
from scipy.stats import sem
import numpy as np
import csv


def fix_tif(filename):
    """
    This function reads a tif file and saves it as a jpg file
    """

    if not os.path.exists(f"""jpg_images//{Path(filename).name}.jpg"""):
        image_tif = tif.imread(filename)
        if image_tif.ndim > 2:
            image_tif = image_tif[1, :, :]
        if not os.path.exists("jpg_images"):
            os.makedirs("jpg_images")
        io.imsave(f"""jpg_images//{Path(filename).name}.jpg""", image_tif)
    image_tif = f"""jpg_images//{Path(filename).name}.jpg"""
    return image_tif


class AxoHandConfig(Config):
    """
    Configuration for training on the axolotl hand dataset.
    Args:
        num_classes: Number of classes (including background)
    Outputs:
        NAME: Name of the configuration
        NUM_CLASSES: Number of classes
        DETECTION_MIN_CONFIDENCE: Minimum confidence for detection
        GPU_COUNT: Number of GPUs
        IMAGES_PER_GPU: Number of images per GPU
    """

    NAME = "axol_hand"

    NUM_CLASSES = 1 + 1

    DETECTION_MIN_CONFIDENCE = 0.9

    GPU_COUNT = 1

    IMAGES_PER_GPU = 1

    def __init__(self, num_classes):
        self.NUM_CLASSES = num_classes
        super().__init__()


def weight_loader(model_type):
    """
    This function loads the weights of the model selected by the user.
    """

    # Checks the model type and loads the corresponding model
    MODEL_DICT = {
        "Axolotl Model": "mask_rcnn_axol_hand_ax",
        "Gastruloids Model": "mask_rcnn_gastruloids_roi_0120",
    }
    CLASSES_DICT = {
        "Axolotl Model": ["late_limb", "early_limb"],
        "Gastruloids Model": ["gastruloid_roi"],
    }
    # Checks if the model is the gastruloids, if it is then it will pop up a popup window alert
    if model_type == "Gastruloids Model":
        messagebox.showinfo(
            title="Alert!", message=f"Gastruloids model is not available yet"
        )
        return
    # Creates the inference configuration
    inference_config = AxoHandConfig(num_classes=1 + len(CLASSES_DICT.get(model_type)))
    ROOT_DIR = os.getcwd()
    # Checks if the model directory exists if not it will generate a new one
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    # Creates the model
    model = modellib.MaskRCNN(
        mode="inference", config=inference_config, model_dir=MODEL_DIR
    )
    # Finds the model weights file name
    model_name = MODEL_DICT.get(model_type)
    model_path = os.path.join(ROOT_DIR, f"roi_detection_module/{model_name}.h5")
    # Loads the model weights
    model.load_weights(model_path, by_name=True)

    return model


class WindowAlert(Tk):
    """
    This class creates a window alert.
    """

    def __init__(self, mainframe):
        Frame.__init__(self, master=mainframe)
        self.master.title("view")
        self.master.minsize(100, 100)
        self.master.geometry("200x200+500+500")


class File:
    """
    This class creates a file object.
    Args:
        name: Name of the file
        path: Path of the file

    """

    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.alpha = None
        self.image = Image_pil.open(path)
        self.roi = None
        self.threshold = None
        self.curve = None
        self.axis_list = None
        self.exclude = None
        self.landmarks = None
        self.amp_plane = None
        self.channel = 0
        self.profile = None
        self.mask_th = []
        self.intersect_list = []
        self.ctrlPoints = None
        self.class_id = None
        self.elbow_ratio = None
        self.wrist_ratio = None


class Root(Tk):
    """
    This class creates the main window.
    Args:
        Tk: Tkinter class
    """

    FLAG_DOUBLE_CLICK = False
    FLAG_IMAGE_LOADED = False
    FLAG_ROI_LOADED = False
    FLAG_READY_FOR_EXPORT = False
    FLAG_AXIS_SAVED = False

    def __init__(self):
        super(Root, self).__init__()
        ### Window configuration
        try:
            # Load the icon with PIL for better quality
            icon = Image_pil.open("meandros_logo.ico")
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(icon)
            # Set the window icon
            self.wm_iconphoto(True, photo)
        except Exception as e:
            # Fallback to basic icon if there's an error
            print(f"Error loading icon: {e}")
            try:
                self.wm_iconbitmap("meandros_logo.ico")
            except:
                pass

        self.title("Meandros")
        self.minsize(400, 300)
        ### Main frame
        self.labelFrame0 = LabelFrame(self, text="Controls")
        self.labelFrame0.grid(row=1, column=1, padx=15, pady=15)

        self.labelFrame01 = LabelFrame(self.labelFrame0, text="Inputs")
        self.labelFrame01.grid(row=1, column=1, padx=10, pady=20)

        self.labelFrame03 = LabelFrame(self.labelFrame0, text="Threshold")
        self.labelFrame03.grid(row=2, column=1, padx=30, pady=40)

        self.labelFrame04 = LabelFrame(self, text="Display")
        self.labelFrame04.grid(row=2, column=1, padx=20, pady=20)

        self.labelFrame_ModelList = LabelFrame(self, text="Model Selection")
        self.labelFrame_ModelList.grid(row=0, column=3, pady=5)

        self.labelFrame05 = LabelFrame(self, text="Project")
        self.labelFrame05.grid(row=1, column=3, padx=30, pady=0)

        self.labelFrame02 = LabelFrame(self.labelFrame05, text="Project Management")
        self.labelFrame02.grid(row=4, column=1, padx=0, pady=10)
        self.labelFrame09 = LabelFrame(self.labelFrame05, text="File Exports")
        self.labelFrame09.grid(row=4, column=2, padx=0, pady=0)

        self.labelFrame07 = LabelFrame(self.labelFrame0, text="Analysis")
        self.labelFrame07.grid(row=3, column=1, padx=15, pady=15)

        self.labelFrame08 = LabelFrame(self.labelFrame0, text="Plotting")
        self.labelFrame08.grid(row=5, column=1, padx=5, pady=10)
        ### Listbox
        self.grouplist = Listbox(
            self.labelFrame05,
            selectmode=EXTENDED,
            width=60,
            height=10,
            highlightcolor="yellow",
            highlightthickness=3,
            selectbackground="orange",
        )
        self.grouplist.grid(row=1, column=1, padx=20, pady=10)
        self.grouplist.bind("<Double-Button-1>", self.doubleclick)

        ### Buttons
        self.button_SignalChannel()
        self.button_RoiDetection()
        self.button_ApplyThresh()
        self.button_ExportProject()
        self.button_AxisGenerator()
        self.button_SaveAxis()
        self.button_Delete()
        self.button_GetProfile()
        self.button_BrightFieldSelector()
        self.button_ExcludeRegion()
        self.button_Landmarks()
        self.button_AmputationPlane()
        self.button_ImportProject()
        self.button_MagicReports()
        self.button_ModelSelector()
        ### Scale

        self.OptionList = {"Red": 2, "Green": 1}
        self.variable = StringVar(self)
        self.variable.set("Red")

        opt = OptionMenu(self.labelFrame03, self.variable, *self.OptionList.keys())
        opt.config(width=4, font=("Helvetica", 10))
        opt.grid(row=1, column=1, padx=20, pady=20)
        self.OptionList_statistic = {"Mann-Whitney": 0, "Student": 1}
        self.variable_statistic = StringVar(self)
        self.variable_statistic.set("Mann-Whitney")

        ### Model selection
        MODEL_LIST = ["Axolotl Model", "Gastruloids Model"]
        self.variable_model_list = StringVar(self)
        self.variable_model_list.set(MODEL_LIST[0])
        opt_models = OptionMenu(
            self.labelFrame_ModelList, self.variable_model_list, *MODEL_LIST
        )
        opt_models.config(width=20, font=("Helvetica", 10))
        opt_models.grid(row=1, column=0, padx=5)

        self.w2 = Scale(
            self.labelFrame03,
            from_=0,
            to=255,
            tickinterval=100,
            orient=HORIZONTAL,
            length=100,
        )
        self.w2.grid(row=1, column=2)
        self.w2.set(23)

        self.bins_input = Scale(
            self.labelFrame08,
            from_=0,
            to=100,
            tickinterval=50,
            orient=HORIZONTAL,
            length=100,
        )
        self.bins_input.set(10)

        self.FLAG_FIRST_EXPORT = True
        self.curves = []
        self.object_list = []
        self.group_1 = None
        self.group_2 = None
        self.model = None

    def get_index(self):
        if self.grouplist.curselection():
            return self.grouplist.curselection()
        else:
            print("No element has been selected")

    def doubleclick(self, event) -> None:
        """

        :type event: object
        """
        _ = event
        try:
            index = self.get_index()[0]
            return self.display(self.object_list[index])
        except IndexError:
            msg = "Any file loaded yet"
            raise IndexError(msg)

    def button_SignalChannel(self):
        button_SignalChannel = ttk.Button(
            self.labelFrame01, text="Signal channel", command=self.filedialog1
        )
        button_SignalChannel.grid(row=1, column=1)

    def button_BrightFieldSelector(self):
        button_BrightFieldSelector = ttk.Button(
            self.labelFrame01, text="BrightField", command=self.filedialog2
        )
        button_BrightFieldSelector.grid(row=1, column=2)

    def button_RoiDetection(self):
        button_RoiDetection = ttk.Button(
            self.labelFrame07, text="ROI detection", command=self.roi_detection
        )  # self.roi_detection
        button_RoiDetection.grid(row=1, column=1)

    def button_AmputationPlane(self):
        button_AmputationPlane = ttk.Button(
            self.labelFrame07, text="Amputation Plane", command=self.amputation_plane
        )
        button_AmputationPlane.grid(row=1, column=4, padx=5)

    def button_ExcludeRegion(self):
        button_ExcludeRegion = ttk.Button(
            self.labelFrame07, text="Exclude Regions", command=self.exclude_regions
        )
        button_ExcludeRegion.grid(row=2, column=4)

    def button_Landmarks(self):
        button_Landmarks = ttk.Button(
            self.labelFrame07, text="Landmarks", command=self.landmarks
        )
        button_Landmarks.grid(row=2, column=1, padx=5)

    def button_AxisGenerator(self):
        button_AxisGenerator = ttk.Button(
            self.labelFrame07, text="Axis detection", command=self.draw_axis
        )
        button_AxisGenerator.grid(row=3, column=1)

    def button_GetProfile(self):
        button_GetProfile = ttk.Button(
            self.labelFrame08, text="Multi-Plot", command=self.get_profile
        )
        button_GetProfile.grid(row=2, column=3, padx=10)

    def button_ApplyThresh(self):
        button_ApplyThresh = ttk.Button(
            self.labelFrame03, text="Apply", command=self.run
        )
        button_ApplyThresh.grid(row=1, column=3)

    def button_ExportProject(self):
        button_ExportProject = ttk.Button(
            self.labelFrame02, text="Save Project", command=self.export_all
        )
        button_ExportProject.grid(row=1, column=1)

    def button_ImportProject(self):
        button_ImportProject = ttk.Button(
            self.labelFrame02, text="Load Project", command=self.import_all
        )
        button_ImportProject.grid(row=1, column=2)

    def button_SaveAxis(self):
        button_SaveAxis = ttk.Button(
            self.labelFrame09, text="Export axis (CSV)", command=self.export_axis
        )
        button_SaveAxis.grid(row=1, column=1)

    def button_Delete(self):
        button_Delete = ttk.Button(
            self.labelFrame05, text="Delete", command=self.delete
        )
        button_Delete.grid(row=2, column=1)

    def button_MagicReports(self):
        button_MagicReports = ttk.Button(
            self.labelFrame08, text="Single Plot", command=self.reports
        )
        button_MagicReports.grid(row=2, column=1, padx=10)

    def button_ModelSelector(self):
        button_ModelSelector = ttk.Button(
            self.labelFrame_ModelList,
            text="Load Model",
            command=self.load_model_selected,
        )
        button_ModelSelector.grid(row=1, column=2, padx=5)

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
        if self.get_index():
            index = self.get_index()[0]
        else:
            messagebox.showinfo(title="Alert!", message=f"No element has been selected")
            return
        f = open("Axis_Extractor.csv", "a+")
        writer = csv.writer(f)
        name = self.object_list[index].name
        for x in self.object_list[index].ctrlPoints:
            r = [x[0], x[1], name]
            writer.writerow(r)
        f.close()

    def display(self, choice) -> None:
        """

        :param choice: type File
        :return: None
        """
        img_l = choice.image.convert("RGB")
        img_r = choice.image.convert("RGB")
        choice.win1 = Zoom_Advanced(Toplevel(self), img_l)
        choice.win1.master.title("current view")
        choice.win2 = Zoom_Advanced(Toplevel(self), img_r)
        choice.win2.master.title("original view")

    def filedialog1(self) -> None:
        try:
            filename = filedialog.askopenfilename(
                initialdir="",
                title="Select signal chanel",
                multiple=True,
                filetypes=(
                    ("jpg files", "*.jpg"),
                    ("tif files", "*.tif"),
                    ("all files", "*.*"),
                ),
            )
            if len(filename):
                for k in enumerate(filename):  # range(len(filename)):
                    finalname = os.path.splitext(os.path.split(k[1])[1])[0]
                    if k[-3:] == "tif":
                        if not os.path.exists("jpg_images"):
                            os.makedirs("jpg_images")
                        image = tif.imread(k)
                        if image.ndim > 2:
                            bright = image[1, :, :]
                            marker = image[0, :, :]
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
        current_img = self.object_list[index].image.convert("RGB")
        current_obj = self.object_list[index]
        color = option_list[self.variable.get()]
        if current_obj.roi is not None:
            pixels = current_img.load()
            self.object_list[index].mask_th = pixels
            for i in current_obj.roi:
                x = i[1]
                y = i[0]
                if pixels[int(y), int(x)][color] >= self.w2.get():
                    if self.variable.get() == "Red":
                        pixels[int(y), int(x)] = (255, 0, 0)
                    else:
                        pixels[int(y), int(x)] = (0, 255, 0)

            current_obj.win1.image = current_img
            current_obj.threshold = self.w2.get()
            current_obj.win1.show_image()
            self.FLAG_READY_FOR_EXPORT = True

            self.grouplist.delete(index, index)
            self.grouplist.insert(index, current_obj.name + ": " + str(self.w2.get()))
            self.object_list[index].threshold = self.w2.get()
            messagebox.showinfo(title="Information", message=f"Threshold Saved")
            self.object_list[index].channel = self.OptionList[self.variable.get()]
        else:
            messagebox.showinfo(
                title="Alert!", message=f"You need to generate ROI first"
            )

        return current_obj.win1

    def run(self):
        option_list = {"Red": 0, "Green": 1}
        if self.get_index():
            index = self.get_index()[0]
        else:
            messagebox.showinfo(title="Alert!", message=f"No element has been selected")
            return

        if self.FLAG_IMAGE_LOADED and self.FLAG_ROI_LOADED:

            return self.load_list_files(option_list, index)
        else:
            if not self.FLAG_IMAGE_LOADED:
                messagebox.showinfo(title="Alert!", message=f"No image loaded")
            if not self.FLAG_ROI_LOADED:
                messagebox.showinfo(
                    title="Alert!", message=f"You need to generate ROI first"
                )
            return

    def filedialog2(self) -> None:
        if self.get_index():
            index = self.get_index()[0]
        else:
            messagebox.showinfo(title="Alert!", message=f"No element has been selected")
            return

        filename = filedialog.askopenfilename(
            initialdir="",
            title="Select Bright Field",
            multiple=True,
            filetypes=(
                ("tif files", "*.tif"),
                ("jpg files", "*.jpg"),
                ("all files", "*.*"),
            ),
        )  # multiple=True,
        if len(filename):
            self.object_list[index].alpha = filename
        else:
            return

    def draw_axis(self):
        index = self.get_index()[0]
        filename = self.object_list[index].path
        if filename[-3:] == "tif":
            image_tif = fix_tif(filename)
        else:
            image_tif = filename
        if self.object_list[index].axis_list is None:
            approx_pd = axis_approx.approx_line(
                self.object_list[index].ctrlPoints, self.object_list[index].class_id
            )
            approx_pd = [tuple(x) for x in approx_pd.values]
        else:
            approx_pd = self.object_list[index].axis_list
        axis_pd, intersect_list = bezierPD.BezierPD(
            filename=image_tif, roi=self.object_list[index].roi, ctrlPoints=approx_pd
        ).run("Axis")
        self.object_list[index].intersect_list = intersect_list
        self.object_list[index].curve = set(map(tuple, axis_pd))
        self.object_list[index].axis_list = approx_pd
        if self.object_list[index].curve is not None:
            self.FLAG_AXIS_SAVED = True
            messagebox.showinfo(title="Information", message=f"Axis Saved")
        return

    def exclude_regions(self):

        if self.get_index():
            index = self.get_index()[0]
        else:
            messagebox.showinfo(title="Alert!", message=f"No element has been selected")
            return
        filename = self.object_list[index].path
        excl = exclusion.Exclusion(filename=filename)
        polygons = excl.run("Exclude Regions")
        self.object_list[index].exclude = polygons
        if self.object_list[index].exclude is not None:
            messagebox.showinfo(title="Information", message=f"Exclusion Regions Saved")

        return

    def roi_detection(self):
        if self.get_index():
            index = self.get_index()[0]
        else:
            messagebox.showinfo(title="Alert!", message=f"No element has been selected")
            return
        if self.model:
            filename = self.object_list[index].path
            if filename[-3:] == "tif":
                image_tif = fix_tif(filename)
            else:
                image_tif = filename
            if not self.object_list[index].ctrlPoints:
                approx_roi, class_id = roi_detect.model_worker(image_tif, self.model)
                self.object_list[index].class_id = class_id
            else:
                approx_roi = self.object_list[index].ctrlPoints
            roi, self.object_list[index].ctrlPoints = roi_selection.RoiSelection(
                filename=image_tif, approx=approx_roi
            ).run("Roi Detection")
            self.object_list[index].roi = roi
            if (
                self.object_list[index].roi is not None
                and self.object_list[index].ctrlPoints is not None
            ):
                self.FLAG_ROI_LOADED = True
                messagebox.showinfo(title="Information", message=f"ROI Saved")
                # ##
        else:
            messagebox.showinfo(
                title="Alert!", message=f"You must select a model first"
            )
        return

    def get_profile(self, indexes=None):
        if self.FLAG_AXIS_SAVED == True:
            try:
                if not indexes:
                    index = self.get_index()[0]
                    filename = self.object_list[index].path
                    img = get_profile.load_exclusion(
                        filename, self.object_list[index].exclude
                    )
                    ap = (
                        self.object_list[index].roi & self.object_list[index].amp_plane
                        if self.object_list[index].amp_plane is not None
                        else self.object_list[index].roi
                    )
                    conj_mask = self.object_list[index].roi
                    p_d = self.object_list[index].curve
                    channel = self.object_list[index].channel
                    threshold = self.object_list[index].threshold
                    landmarks = self.object_list[index].landmarks
                    (
                        self.object_list[index].profile,
                        self.object_list[index].elbow_ratio,
                        self.object_list[index].wrist_ratio,
                    ) = get_profile.analysis(
                        list(p_d),
                        list(conj_mask),
                        ap,
                        img,
                        channel,
                        threshold,
                        landmarks,
                    )
                else:
                    for index in indexes:
                        filename = self.object_list[index].path
                        img = get_profile.load_exclusion(
                            filename, self.object_list[index].exclude
                        )
                        ap = (
                            self.object_list[index].roi
                            & self.object_list[index].amp_plane
                        )
                        conj_mask = self.object_list[index].roi
                        p_d = self.object_list[index].curve
                        channel = self.object_list[index].channel
                        threshold = self.object_list[index].threshold
                        landmarks = self.object_list[index].landmarks
                        (
                            self.object_list[index].profile,
                            self.object_list[index].elbow_ratio,
                            self.object_list[index].wrist_ratio,
                        ) = get_profile.analysis(
                            list(p_d),
                            list(conj_mask),
                            ap,
                            img,
                            channel,
                            threshold,
                            landmarks,
                        )
            except Exception as e:
                messagebox.showinfo(
                    title="Alert!", message=f"Profiling Failed, please check the axis"
                )
                print(e)
        else:
            messagebox.showinfo(
                title="Alert!", message=f"You need to generate Axis first"
            )
        return

    def landmarks(self):
        if self.get_index():
            index = self.get_index()[0]
        else:
            messagebox.showinfo(title="Alert!", message=f"No element has been selected")
            return
        filename = self.object_list[index].path
        if self.object_list[index].landmarks is not None:
            landmark_pts = self.object_list[index].landmarks
        else:
            landmark_pts = []
        self.object_list[index].landmarks = landmarks.Landmarks(
            filename=filename, ctrl_points=landmark_pts
        ).run("Landmarks")
        if self.object_list[index].landmarks is not None:
            messagebox.showinfo(title="Information", message=f"Landmarks Saved")
        return

    def amputation_plane(self):
        if self.get_index():
            index = self.get_index()[0]
        else:
            messagebox.showinfo(title="Alert!", message=f"No element has been selected")
            return
        filename = self.object_list[index].path
        if filename[-3:] == "tif":
            image_tif = fix_tif(filename)
        else:
            image_tif = filename

        if self.object_list[index].amp_plane is None:
            amp_plane = []
        else:
            amp_plane = list(self.object_list[index].amp_plane)
        ap_plane, intersect_ap_list = bezierPD.BezierPD(
            filename=image_tif, roi=self.object_list[index].roi, ctrlPoints=amp_plane
        ).run("Amputation Plane")
        self.object_list[index].amp_plane = set(map(tuple, ap_plane))
        if self.object_list[index].amp_plane is not None:
            messagebox.showinfo(title="Information", message=f"Amputation Plane Saved")
        return

    def statistic_group_1(self, indexes):
        if len(indexes) == 1:
            group = self.object_list[indexes[0]].profile.sort_values(
                by=["PD"], ascending=True
            )
        else:
            group = pd.concat(
                [self.object_list[i].profile for i in indexes],
                ignore_index=True,
                sort=False,
            ).sort_values(by=["PD"], ascending=True)
        select_item = self.grouplist.curselection()
        for elem in select_item:
            self.grouplist.itemconfig(elem, {"bg": "yellow"})
        self.group_1 = group
        self.group_1.to_csv("ctrl-group.csv")
        elbows = [self.object_list[i].elbow_ratio for i in indexes]
        wrists = [self.object_list[i].wrist_ratio for i in indexes]
        elbow_mean = np.mean([e for e in elbows if e is not None])
        wrist_mean = np.mean([w for w in wrists if w is not None])
        elbow_sem = sem(elbows)
        wrist_sem = sem(wrists)
        return elbow_mean, elbow_sem, wrist_mean, wrist_sem

    def statistic_run(self):
        if self.variable_statistic.get() == "Student":
            if self.group_1 is not None and self.group_2 is not None:
                return group_stat.Statistics(
                    self.group_1,
                    self.group_2,
                    self.bins_input.get(),
                    self.variable_statistic.get(),
                )
            else:
                print("You need to add groups first")
        elif self.variable_statistic.get() == "Mann-Whitney":
            if self.group_1 is not None and self.group_2 is not None:
                return group_stat.Statistics(
                    self.group_1,
                    self.group_2,
                    self.bins_input.get(),
                    self.variable_statistic.get(),
                )
            else:
                print("You need to add groups first")

    def export_all(self):
        if self.get_index():
            index = self.get_index()[0]
        else:
            messagebox.showinfo(title="Alert!", message=f"No element has been selected")
            return
        try:
            saving_path = filedialog.asksaveasfile(
                mode="wb", initialdir="", initialfile="data", defaultextension=".p"
            )
            if self.object_list[index].threshold:
                print("Brightfield cannot be saved")
                self.object_list[index].threshold = None

            # Create a deep copy of the object list
            data = copy.deepcopy(self.object_list)

            # Iterate over the copied list and remove the Tkinter objects
            for obj in data:
                if hasattr(obj, "tkinter_object_attribute"):
                    del obj.tkinter_object_attribute

            pickle.dump(data, open(saving_path.name, "wb"))
        except Exception as e:
            print(e)

    def import_all(self):
        try:
            open_path = filedialog.askopenfilename(
                initialdir="", initialfile="data", defaultextension=".p"
            )
            list_files_import = pickle.load(open(open_path, "rb"))
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

    def load_model_selected(self):
        # Loads the model selected by the user
        self.model = weight_loader(self.variable_model_list.get())
        # Shows a popup window alert that the model has been loaded if not it will not raise any alert
        if self.model is not None:
            messagebox.showinfo(
                title="Information", message=f"{self.variable_model_list.get()} loaded"
            )

    def reports(self):
        indexes = list(self.get_index())
        if len(indexes) >= 1:
            self.get_profile(indexes)
            print(indexes)
            e, e_sem, w, w_sem = self.statistic_group_1(indexes)
            return sdp.mean_sd_plots(self.group_1, e, e_sem, w, w_sem)
        else:
            messagebox.showinfo(title="Alert!", message=f"No element has been selected")
            return


def main():
    if os.environ.get("DISPLAY", "") == "":
        os.environ.__setitem__("DISPLAY", ":0.0")
    root = Root()
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    return root


if __name__ == "__main__":
    main()
