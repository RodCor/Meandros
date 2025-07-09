import copy
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image as Image_pil, ImageTk, Image
from generals import bezierPD
import exclusion_module.exclusion as exclusion
from generals.zoom_modern import Zoom_Advanced
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
import tkinter as tk
import cv2

# Set dark appearance
ctk.set_appearance_mode("Dark")  # Dark theme
ctk.set_default_color_theme("blue")  # Clean blue accents

# ========================================
# COLOR THEME CONFIGURATION
# ========================================
# Change these values to modify the entire theme
THEME_CONFIG = {
    "primary_bg": "#000000",        # Main background
    "secondary_bg": "#1A1A1A",      # Sidebar/panel background  
    "tertiary_bg": "#2A2A2A",       # Cards/sections background
    "border_color": "#333333",      # Borders and separators
    "text_primary": "#FFFFFF",      # Main text color
    "text_secondary": "#CCCCCC",    # Secondary text color
    "accent_primary": "#8A2BE2",    # Primary accent (Blue Violet)
    "accent_secondary": "#6B46C1",  # Secondary accent (Dark Purple)
    "accent_light": "#9D4EDD",      # Light accent for hovers
    "error_color": "#FF3B30",       # Error/danger color
    "success_color": "#34C759"      # Success color
}

# Apply theme colors to variables for backwards compatibility
DARK_BLACK = THEME_CONFIG["primary_bg"]
DARK_GRAY = THEME_CONFIG["secondary_bg"] 
DARK_PANEL = THEME_CONFIG["tertiary_bg"]
DARK_BORDER = THEME_CONFIG["border_color"]
TEXT_WHITE = THEME_CONFIG["text_primary"]
TEXT_GRAY = THEME_CONFIG["text_secondary"]
ACCENT_PURPLE = THEME_CONFIG["accent_primary"]
ACCENT_DARK_PURPLE = THEME_CONFIG["accent_secondary"]
ACCENT_LIGHT_PURPLE = THEME_CONFIG["accent_light"]
DARK_RED = THEME_CONFIG["error_color"]
DARK_GREEN = THEME_CONFIG["success_color"]

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
    MODEL_DICT = {
        "Axolotl Model": "mask_rcnn_axol_hand_ax",
        "Gastruloids Model": "mask_rcnn_gastruloids_roi_0120",
    }
    CLASSES_DICT = {
        "Axolotl Model": ["late_limb", "early_limb"],
        "Gastruloids Model": ["gastruloid_roi"],
    }
    
    inference_config = AxoHandConfig(num_classes=1 + len(CLASSES_DICT.get(model_type)))
    ROOT_DIR = os.getcwd()
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    model = modellib.MaskRCNN(
        mode="inference", config=inference_config, model_dir=MODEL_DIR
    )
    model_name = MODEL_DICT.get(model_type)
    model_path = os.path.join(ROOT_DIR, f"roi_detection_module/{model_name}.h5")
    model.load_weights(model_path, by_name=True)
    return model

class IntegratedCanvasEditor:
    """
    Integrated canvas editor that works within the main canvas instead of opening popup windows.
    """
    def __init__(self, canvas_viewer, filename, ctrl_points=None, parent=None):
        self.canvas_viewer = canvas_viewer
        self.filename = filename
        self.parent = parent
        self.ctrlPoints = ctrl_points if ctrl_points is not None else []
        self.finalpoints = []
        self.result = None
        
        # Flags for different modes
        self.INSERT_FLAG = False
        self.MOVE_FLAG = False
        self.DELETE_FLAG = False
        self.ACTIVE_MOV_FLAG = False
        self.ind = None
        
        # Visual settings
        self.thickness = 8
        self.epsilon = 15
        self.RED = "#FF0000"
        self.WHITE = "#FFFFFF"
        self.ORANGE = "#FFA500"
        self.BLUE = "#0000FF"
        
        # Get the canvas from the viewer
        self.canvas = canvas_viewer.canvas
        self.scale_factor = canvas_viewer.imscale
        
        # Bind canvas events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        
        # Bind keyboard events to the main window
        if parent:
            parent.bind("<KeyPress>", self.on_key_press)
            parent.focus_set()
        
        # Create toolbar in the main window
        self.create_toolbar()
        
        # Start in insert mode
        self.activate_insert_mode()
        
        # Draw existing control points if any - use multiple delay strategies
        if self.ctrlPoints:
            # Try multiple strategies to ensure points are drawn
            if parent:
                parent.after(100, self.redraw_points)  # 100ms delay
                parent.after(500, self.redraw_points)  # 500ms delay as backup
            else:
                self.redraw_points()
                # Also try a delayed redraw
                if hasattr(self, 'canvas') and self.canvas:
                    self.canvas.after(100, self.redraw_points)
    
    def create_toolbar(self):
        """Create toolbar in the main window"""
        if not self.parent:
            return
            
        # Create toolbar frame at the top of the canvas
        self.toolbar_frame = ctk.CTkFrame(self.parent.canvas_frame, height=50, fg_color=DARK_PANEL)
        self.toolbar_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 0))
        self.toolbar_frame.grid_propagate(False)
        
        # Toolbar buttons
        button_style = {
            "height": 32,
            "corner_radius": 6,
            "font": ctk.CTkFont(size=11, weight="normal"),
            "fg_color": DARK_BORDER,
            "hover_color": ACCENT_PURPLE,
            "text_color": TEXT_WHITE,
            "border_width": 1,
            "border_color": ACCENT_PURPLE
        }
        
        self.insert_btn = ctk.CTkButton(
            self.toolbar_frame, 
            text="Insert (I)", 
            command=self.activate_insert_mode,
            width=80,
            **button_style
        )
        self.insert_btn.pack(side="left", padx=5, pady=9)
        
        self.move_btn = ctk.CTkButton(
            self.toolbar_frame, 
            text="Move (M)", 
            command=self.activate_move_mode,
            width=80,
            **button_style
        )
        self.move_btn.pack(side="left", padx=5, pady=9)
        
        self.delete_btn = ctk.CTkButton(
            self.toolbar_frame, 
            text="Delete (D)", 
            command=self.activate_delete_mode,
            width=80,
            **button_style
        )
        self.delete_btn.pack(side="left", padx=5, pady=9)
        
        self.finish_btn = ctk.CTkButton(
            self.toolbar_frame, 
            text="Finish", 
            command=self.finish_editing,
            width=80,
            fg_color=ACCENT_PURPLE,
            hover_color=ACCENT_LIGHT_PURPLE,
            text_color=TEXT_WHITE,
            **{k: v for k, v in button_style.items() if k not in ['fg_color', 'hover_color', 'text_color']}
        )
        self.finish_btn.pack(side="right", padx=5, pady=9)
        
        # Status label
        self.status_label = ctk.CTkLabel(
            self.toolbar_frame,
            text=f"Points: {len(self.ctrlPoints)} | Mode: Insert",
            font=ctk.CTkFont(size=11, weight="normal"),
            text_color=TEXT_WHITE
        )
        self.status_label.pack(side="right", padx=20, pady=9)
        
        # Update button states
        self.update_button_states()
    
    def activate_insert_mode(self):
        """Activate insert mode"""
        self.INSERT_FLAG = True
        self.MOVE_FLAG = False
        self.DELETE_FLAG = False
        self.update_button_states()
        self.update_status("Insert")
    
    def activate_move_mode(self):
        """Activate move mode"""
        self.INSERT_FLAG = False
        self.MOVE_FLAG = True
        self.DELETE_FLAG = False
        self.update_button_states()
        self.update_status("Move")
    
    def activate_delete_mode(self):
        """Activate delete mode"""
        self.INSERT_FLAG = False
        self.MOVE_FLAG = False
        self.DELETE_FLAG = True
        self.update_button_states()
        self.update_status("Delete")
    
    def update_button_states(self):
        """Update button visual states"""
        # Reset all buttons
        active_color = ACCENT_PURPLE
        inactive_color = DARK_BORDER
        
        self.insert_btn.configure(fg_color=active_color if self.INSERT_FLAG else inactive_color)
        self.move_btn.configure(fg_color=active_color if self.MOVE_FLAG else inactive_color)
        self.delete_btn.configure(fg_color=active_color if self.DELETE_FLAG else inactive_color)
    
    def update_status(self, mode):
        """Update status label"""
        if hasattr(self, 'status_label'):
            self.status_label.configure(text=f"Points: {len(self.ctrlPoints)} | Mode: {mode}")
    
    def on_key_press(self, event):
        """Handle keyboard events"""
        key = event.keysym.lower()
        if key == 'i':
            self.activate_insert_mode()
        elif key == 'm':
            self.activate_move_mode()
        elif key == 'd':
            self.activate_delete_mode()
        elif key == 'escape':
            self.finish_editing()
    
    def on_canvas_click(self, event):
        """Handle canvas click events"""
        # Convert canvas coordinates to image coordinates
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # Scale coordinates back to original image size
        orig_x = int(x / self.canvas_viewer.imscale)
        orig_y = int(y / self.canvas_viewer.imscale)
        
        if self.INSERT_FLAG:
            # Add new point
            self.ctrlPoints.append([orig_x, orig_y])
            self.redraw_points()
            self.update_status("Insert")
            
        elif self.MOVE_FLAG:
            # Find point to move
            self.ind = self.get_id_under_point(orig_x, orig_y)
            if self.ind is not None:
                self.ACTIVE_MOV_FLAG = True
                
        elif self.DELETE_FLAG:
            # Delete point
            self.ind = self.get_id_under_point(orig_x, orig_y)
            if self.ind is not None:
                self.ctrlPoints.pop(self.ind)
                self.redraw_points()
                self.update_status("Delete")
    
    def on_canvas_drag(self, event):
        """Handle canvas drag events"""
        if self.ACTIVE_MOV_FLAG and self.ind is not None:
            # Convert canvas coordinates to image coordinates
            x = self.canvas.canvasx(event.x)
            y = self.canvas.canvasy(event.y)
            
            # Scale coordinates back to original image size
            orig_x = int(x / self.canvas_viewer.imscale)
            orig_y = int(y / self.canvas_viewer.imscale)
            
            # Update point position
            self.ctrlPoints[self.ind] = [orig_x, orig_y]
            self.redraw_points()
    
    def on_canvas_release(self, event):
        """Handle canvas release events"""
        self.ACTIVE_MOV_FLAG = False
        self.ind = None
    
    def get_id_under_point(self, x, y):
        """Find the index of the point under the given coordinates"""
        if not self.ctrlPoints:
            return None
            
        xy = np.array(self.ctrlPoints)
        xt, yt = xy[:, 0], xy[:, 1]
        d = np.hypot(xt - x, yt - y)
        (indseq,) = np.nonzero(d == d.min())
        ind = indseq[0]
        
        # Scale epsilon for current zoom level
        scaled_epsilon = self.epsilon / self.canvas_viewer.imscale
        if d[ind] >= scaled_epsilon:
            ind = None
        return ind
    
    def redraw_points(self):
        """Redraw all points and lines on the canvas"""
        # Check if canvas exists
        if not hasattr(self, 'canvas') or not self.canvas:
            return
            
        # Clear previous drawings
        self.canvas.delete("control_points")
        self.canvas.delete("control_lines")
        
        if not self.ctrlPoints:
            return
            
        # Draw lines between points
        if len(self.ctrlPoints) > 1:
            for i in range(len(self.ctrlPoints) - 1):
                x1, y1 = self.ctrlPoints[i]
                x2, y2 = self.ctrlPoints[i + 1]
                
                # Scale coordinates for display
                display_x1 = x1 * self.canvas_viewer.imscale
                display_y1 = y1 * self.canvas_viewer.imscale
                display_x2 = x2 * self.canvas_viewer.imscale
                display_y2 = y2 * self.canvas_viewer.imscale
                
                self.canvas.create_line(
                    display_x1, display_y1, display_x2, display_y2,
                    fill=self.ORANGE, width=2, tags="control_lines"
                )
        
        # Draw points
        for i, (x, y) in enumerate(self.ctrlPoints):
            # Scale coordinates for display
            display_x = x * self.canvas_viewer.imscale
            display_y = y * self.canvas_viewer.imscale
            
            # Draw point
            self.canvas.create_oval(
                display_x - self.thickness//2, display_y - self.thickness//2,
                display_x + self.thickness//2, display_y + self.thickness//2,
                fill=self.RED, outline=self.WHITE, width=2, tags="control_points"
            )
            
            # Draw point number
            self.canvas.create_text(
                display_x, display_y - self.thickness - 5,
                text=str(i + 1), fill=self.WHITE, font=("Arial", 8, "bold"), tags="control_points"
            )
        
        # Update status
        if hasattr(self, 'update_status'):
            self.update_status(f"Insert ({len(self.ctrlPoints)} points)")
    
    def finish_editing(self):
        """Finish editing and return results"""
        print(f"Finishing editing with {len(self.ctrlPoints)} control points: {self.ctrlPoints}")
        
        # Call the completion callback if provided
        if hasattr(self, 'completion_callback') and self.completion_callback:
            self.completion_callback(self.ctrlPoints)
        
        # Clean up after callback
        self.cleanup()
    
    def cleanup(self):
        """Clean up the editor"""
        # Remove toolbar
        if hasattr(self, 'toolbar_frame'):
            self.toolbar_frame.destroy()
        
        # Remove bindings
        if self.canvas:
            self.canvas.unbind("<Button-1>")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")
        
        # Remove control points from canvas
        if self.canvas:
            self.canvas.delete("control_points")
            self.canvas.delete("control_lines")
        
        # Remove keyboard bindings
        if self.parent:
            self.parent.unbind("<KeyPress>")
    
    def run(self, completion_callback=None):
        """Run the editor with a completion callback"""
        self.completion_callback = completion_callback
        # The editor is already running since it's integrated
        return self.ctrlPoints

class File:
    """
    This class creates a file object.
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

class Root(ctk.CTk):
    """
    Apple-like modern main window with integrated canvas interface.
    """
    FLAG_DOUBLE_CLICK = False
    FLAG_IMAGE_LOADED = False
    FLAG_ROI_LOADED = False
    FLAG_READY_FOR_EXPORT = False
    FLAG_AXIS_SAVED = False

    def __init__(self):
        super().__init__()
        
        # Window configuration
        self.title("Meandros")
        self.geometry("1600x1000")
        self.minsize(1400, 900)
        self.configure(fg_color=DARK_BLACK)
        
        # Load and set icon
        try:
            icon = Image_pil.open("meandros_logo.ico")
            photo = ImageTk.PhotoImage(icon)
            self.wm_iconphoto(True, photo)
        except Exception as e:
            print(f"Error loading icon: {e}")

        # Configure main grid
        self.grid_columnconfigure(0, weight=0)  # Left sidebar
        self.grid_columnconfigure(1, weight=1)  # Main canvas
        self.grid_columnconfigure(2, weight=0)  # Right project files
        self.grid_rowconfigure(0, weight=0)     # Top bar
        self.grid_rowconfigure(1, weight=1)     # Main content

        # Initialize resize variables (needed before create_layout)
        self.resizing = False
        self.resize_start_x = 0
        self.resize_start_width = 0
        self.project_panel_width = 280
        self.manual_resize_active = False  # Flag to prevent auto-sizing during manual resize

        # Create layout
        self.create_layout()
        self.create_widgets()
        
        # Initialize variables
        self.FLAG_FIRST_EXPORT = True
        self.curves = []
        self.object_list = []
        self.selected_indices = []
        self.group_1 = None
        self.group_2 = None
        self.model = None
        self.current_canvas_tool = None

    def create_layout(self):
        """Create the main Apple-like layout"""
        # Configure main grid - now we have 3 columns
        self.grid_columnconfigure(0, weight=0, minsize=320)    # Left sidebar - fixed
        self.grid_columnconfigure(1, weight=1, minsize=600)    # Main canvas - expandable
        self.grid_columnconfigure(2, weight=0, minsize=120)    # Right project files - resizable but fixed weight
        
        # Top bar for project management
        self.top_bar = ctk.CTkFrame(self, height=60, corner_radius=0, fg_color=DARK_PANEL)
        self.top_bar.grid(row=0, column=0, columnspan=3, sticky="ew", padx=0, pady=0)
        self.top_bar.grid_propagate(False)
        
        # Left sidebar for analysis tools
        self.sidebar = ctk.CTkFrame(self, width=320, corner_radius=0, fg_color=DARK_GRAY)
        self.sidebar.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar.grid_propagate(False)
        
        # Main canvas area
        self.canvas_frame = ctk.CTkFrame(self, corner_radius=0, fg_color=DARK_BLACK)
        self.canvas_frame.grid(row=1, column=1, sticky="nsew", padx=0, pady=0)
        self.canvas_frame.grid_columnconfigure(0, weight=1)
        self.canvas_frame.grid_rowconfigure(0, weight=0)  # Header row
        self.canvas_frame.grid_rowconfigure(1, weight=0)  # Toolbar row (dynamic)
        self.canvas_frame.grid_rowconfigure(2, weight=1)  # Main content row
        
        # Right panel for project files (resizable)
        self.project_panel = ctk.CTkFrame(self, width=self.project_panel_width, corner_radius=0, fg_color=DARK_GRAY)
        self.project_panel.grid(row=1, column=2, sticky="nsew", padx=0, pady=0)
        self.project_panel.grid_propagate(False)
        
        # Add resize handle with better positioning
        self.resize_handle = ctk.CTkFrame(self.project_panel, width=8, corner_radius=0, fg_color=ACCENT_PURPLE)
        self.resize_handle.place(x=0, y=0, relheight=1.0)
        
        # Make sure the resize handle is on top
        self.resize_handle.lift()
        
        # Bind events only to the resize handle with better event handling
        self.resize_handle.bind("<Button-1>", self.start_resize, add='+')
        self.resize_handle.bind("<B1-Motion>", self.do_resize, add='+')  
        self.resize_handle.bind("<ButtonRelease-1>", self.stop_resize, add='+')
        self.resize_handle.bind("<Double-Button-1>", self.reset_to_auto_size, add='+')  # Double-click to auto-size
        self.resize_handle.bind("<Enter>", lambda e: self.resize_handle.configure(fg_color=ACCENT_LIGHT_PURPLE))
        self.resize_handle.bind("<Leave>", lambda e: self.resize_handle.configure(fg_color=ACCENT_PURPLE))
        
        # Prevent event propagation
        self.resize_handle.bind("<Button>", lambda e: "break", add='+')
        self.resize_handle.bind("<Motion>", lambda e: "break", add='+')
        
        # Make sure the handle stays on top when other widgets are added
        self.after(100, lambda: self.resize_handle.lift())
        
        # Prevent resize handle from interfering with other events
        self.resize_handle.bind("<Button-1>", lambda e: "break", add="+")
        
        # Configure cursor for resize handle (Windows compatible)
        try:
            self.resize_handle.configure(cursor="size_we")
        except:
            # Fallback for other platforms
            try:
                self.resize_handle.configure(cursor="sb_h_double_arrow")
            except:
                pass

    def create_widgets(self):
        """Create all widgets with Apple-like styling"""
        self.create_top_bar()
        self.create_sidebar()
        self.create_canvas_area()
        self.create_project_panel()

    def create_top_bar(self):
        """Create Apple-like top bar with project management"""
        # App title
        title_label = ctk.CTkLabel(
            self.top_bar, 
            text="Meandros", 
            font=ctk.CTkFont(size=18, weight="normal"),
            text_color=TEXT_WHITE
        )
        title_label.pack(side="left", padx=20, pady=15)
        
        # Project management buttons
        button_frame = ctk.CTkFrame(self.top_bar, fg_color="transparent")
        button_frame.pack(side="right", padx=20, pady=10)
        
        # Dark theme button styling
        button_style = {
            "height": 32,
            "corner_radius": 6,
            "font": ctk.CTkFont(size=12, weight="normal"),
            "fg_color": DARK_BORDER,
            "hover_color": ACCENT_PURPLE,
            "text_color": TEXT_WHITE,
            "border_width": 1,
            "border_color": ACCENT_PURPLE
        }
        
        self.save_project_btn = ctk.CTkButton(
            button_frame, 
            text="Save Project", 
            command=self.export_all,
            width=110,
            **button_style
        )
        self.save_project_btn.pack(side="left", padx=5)
        
        self.load_project_btn = ctk.CTkButton(
            button_frame, 
            text="Load Project", 
            command=self.import_all,
            width=110,
            **button_style
        )
        self.load_project_btn.pack(side="left", padx=5)
        
        self.export_axis_btn = ctk.CTkButton(
            button_frame, 
            text="Export CSV", 
            command=self.export_axis,
            width=100,
            **button_style
        )
        self.export_axis_btn.pack(side="left", padx=5)

    def create_sidebar(self):
        """Create Apple-like sidebar with analysis tools"""
        # Sidebar scroll frame
        self.sidebar_scroll = ctk.CTkScrollableFrame(
            self.sidebar, 
            width=300,
            fg_color="transparent",
            scrollbar_button_color=ACCENT_PURPLE,
            scrollbar_button_hover_color=ACCENT_LIGHT_PURPLE
        )
        self.sidebar_scroll.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Section styling
        section_style = {
            "corner_radius": 12,
            "fg_color": DARK_PANEL,
            "border_width": 1,
            "border_color": ACCENT_PURPLE
        }
        
        button_style = {
            "height": 36,
            "corner_radius": 6,
            "font": ctk.CTkFont(size=13, weight="normal"),
            "fg_color": ACCENT_PURPLE,
            "hover_color": ACCENT_LIGHT_PURPLE,
            "text_color": TEXT_WHITE,
            "border_width": 0
        }
        
        small_button_style = {
            "height": 32,
            "corner_radius": 6,
            "font": ctk.CTkFont(size=11, weight="normal"),
            "fg_color": DARK_BORDER,
            "hover_color": ACCENT_PURPLE,
            "text_color": TEXT_WHITE,
            "border_width": 0
        }
        
        # Model Selection
        model_section = ctk.CTkFrame(self.sidebar_scroll, **section_style)
        model_section.pack(fill="x", pady=(0, 15))
        
        ctk.CTkLabel(
            model_section, 
            text="Model Selection", 
            font=ctk.CTkFont(size=15, weight="normal"),
            text_color=TEXT_WHITE
        ).pack(pady=(15, 10))
        
        MODEL_LIST = ["Axolotl Model", "Gastruloids Model"]
        self.model_var = ctk.StringVar(value=MODEL_LIST[0])
        
        self.model_dropdown = ctk.CTkOptionMenu(
            model_section, 
            values=MODEL_LIST, 
            variable=self.model_var,
            width=250,
            height=35,
            corner_radius=8,
            fg_color=DARK_BORDER,
            button_color=ACCENT_PURPLE,
            button_hover_color=ACCENT_LIGHT_PURPLE,
            text_color=TEXT_WHITE
        )
        self.model_dropdown.pack(pady=(0, 10))
        
        self.load_model_btn = ctk.CTkButton(
            model_section, 
            text="Load Model", 
            command=self.load_model_selected,
            width=250,
            **button_style
        )
        self.load_model_btn.pack(pady=(0, 15))
        
        # File Input
        input_section = ctk.CTkFrame(self.sidebar_scroll, **section_style)
        input_section.pack(fill="x", pady=(0, 15))
        
        ctk.CTkLabel(
            input_section, 
            text="File Input", 
            font=ctk.CTkFont(size=15, weight="normal"),
            text_color=TEXT_WHITE
        ).pack(pady=(15, 10))
        
        input_btn_frame = ctk.CTkFrame(input_section, fg_color="transparent")
        input_btn_frame.pack(pady=(0, 15))
        
        self.signal_btn = ctk.CTkButton(
            input_btn_frame, 
            text="Signal Channel", 
            command=self.filedialog1,
            width=120,
            **small_button_style
        )
        self.signal_btn.pack(side="left", padx=5)
        
        self.brightfield_btn = ctk.CTkButton(
            input_btn_frame, 
            text="Bright Field", 
            command=self.filedialog2,
            width=120,
            **small_button_style
        )
        self.brightfield_btn.pack(side="right", padx=5)
        
        # Threshold Settings
        threshold_section = ctk.CTkFrame(self.sidebar_scroll, **section_style)
        threshold_section.pack(fill="x", pady=(0, 15))
        
        ctk.CTkLabel(
            threshold_section, 
            text="Threshold Settings", 
            font=ctk.CTkFont(size=15, weight="normal"),
            text_color=TEXT_WHITE
        ).pack(pady=(15, 10))
        
        # Channel selection
        channel_frame = ctk.CTkFrame(threshold_section, fg_color="transparent")
        channel_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(
            channel_frame, 
            text="Channel:", 
            font=ctk.CTkFont(size=12, weight="normal"),
            text_color=TEXT_WHITE
        ).pack(side="left", padx=(15, 5))
        
        self.channel_var = ctk.StringVar(value="Red")
        self.channel_dropdown = ctk.CTkOptionMenu(
            channel_frame, 
            values=["Red", "Green"], 
            variable=self.channel_var,
            width=100,
            height=30,
            corner_radius=6,
            fg_color=DARK_BORDER,
            text_color=TEXT_WHITE
        )
        self.channel_dropdown.pack(side="left", padx=5)
        
        # Threshold slider
        slider_frame = ctk.CTkFrame(threshold_section, fg_color="transparent")
        slider_frame.pack(fill="x", padx=15, pady=(0, 10))
        
        ctk.CTkLabel(
            slider_frame, 
            text="Threshold Value:", 
            font=ctk.CTkFont(size=12, weight="normal"),
            text_color=TEXT_WHITE
        ).pack(anchor="w")
        
        self.threshold_slider = ctk.CTkSlider(
            slider_frame, 
            from_=0, 
            to=255, 
            number_of_steps=255,
            width=220,
            height=16,
            progress_color=ACCENT_PURPLE,
            button_color=ACCENT_PURPLE,
            button_hover_color=ACCENT_LIGHT_PURPLE
        )
        self.threshold_slider.pack(fill="x", pady=5)
        self.threshold_slider.set(23)
        
        self.threshold_value_label = ctk.CTkLabel(
            slider_frame, 
            text="23", 
            font=ctk.CTkFont(size=11, weight="normal"),
            text_color=ACCENT_PURPLE
        )
        self.threshold_value_label.pack()
        
        self.threshold_slider.configure(command=self.update_threshold_label)
        
        self.apply_btn = ctk.CTkButton(
            threshold_section, 
            text="Apply Threshold", 
            command=self.run,
            width=250,
            **button_style
        )
        self.apply_btn.pack(pady=(0, 15))
        
        # Analysis Tools
        analysis_section = ctk.CTkFrame(self.sidebar_scroll, **section_style)
        analysis_section.pack(fill="x", pady=(0, 15))
        
        ctk.CTkLabel(
            analysis_section, 
            text="Analysis Tools", 
            font=ctk.CTkFont(size=15, weight="normal"),
            text_color=TEXT_WHITE
        ).pack(pady=(15, 10))
        
        analysis_buttons = [
            ("ROI Detection", self.roi_detection),
            ("Amputation Plane", self.amputation_plane),
            ("Landmarks", self.landmarks),
            ("Exclude Regions", self.exclude_regions),
            ("Axis Detection", self.draw_axis),
        ]
        
        for btn_text, btn_command in analysis_buttons:
            ctk.CTkButton(
                analysis_section, 
                text=btn_text, 
                command=btn_command,
                width=250,
                **button_style
            ).pack(pady=(0, 8))
        
        # Add some padding at the bottom
        ctk.CTkFrame(analysis_section, height=10, fg_color="transparent").pack()
        
        # Plotting & Reports
        plotting_section = ctk.CTkFrame(self.sidebar_scroll, **section_style)
        plotting_section.pack(fill="x", pady=(0, 15))
        
        ctk.CTkLabel(
            plotting_section, 
            text="Plotting & Reports", 
            font=ctk.CTkFont(size=15, weight="normal"),
            text_color=TEXT_WHITE
        ).pack(pady=(15, 10))
        
        # Bins slider
        bins_frame = ctk.CTkFrame(plotting_section, fg_color="transparent")
        bins_frame.pack(fill="x", padx=15, pady=(0, 10))
        
        ctk.CTkLabel(
            bins_frame, 
            text="Bins:", 
            font=ctk.CTkFont(size=12, weight="normal"),
            text_color=TEXT_WHITE
        ).pack(anchor="w")
        
        self.bins_slider = ctk.CTkSlider(
            bins_frame, 
            from_=0, 
            to=100, 
            number_of_steps=100,
            width=220,
            height=16,
            progress_color=ACCENT_PURPLE,
            button_color=ACCENT_PURPLE
        )
        self.bins_slider.pack(fill="x", pady=5)
        self.bins_slider.set(10)
        
        plot_btn_frame = ctk.CTkFrame(plotting_section, fg_color="transparent")
        plot_btn_frame.pack(pady=(0, 15))
        
        self.single_plot_btn = ctk.CTkButton(
            plot_btn_frame, 
            text="Single Plot", 
            command=self.reports,
            width=120,
            **small_button_style
        )
        self.single_plot_btn.pack(side="left", padx=5)
        
        self.multi_plot_btn = ctk.CTkButton(
            plot_btn_frame, 
            text="Multi-Plot", 
            command=self.get_profile,
            width=120,
            **small_button_style
        )
        self.multi_plot_btn.pack(side="right", padx=5)
        
        # Remove project files from sidebar - they're now in the right panel
        self.file_items = []

    def create_canvas_area(self):
        """Create the main canvas area for displaying images and tools"""
        # Canvas header
        canvas_header = ctk.CTkFrame(self.canvas_frame, height=50, corner_radius=0, fg_color=DARK_PANEL)
        canvas_header.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        canvas_header.grid_propagate(False)
        
        self.canvas_title = ctk.CTkLabel(
            canvas_header, 
            text="Canvas", 
            font=ctk.CTkFont(size=16, weight="normal"),
            text_color=TEXT_WHITE
        )
        self.canvas_title.pack(side="left", padx=20, pady=15)
        
        # Main canvas
        self.main_canvas = ctk.CTkFrame(self.canvas_frame, corner_radius=8, fg_color=DARK_PANEL)
        self.main_canvas.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        self.main_canvas.grid_columnconfigure(0, weight=1)
        self.main_canvas.grid_columnconfigure(1, weight=0)  # For scrollbar
        self.main_canvas.grid_rowconfigure(0, weight=1)
        self.main_canvas.grid_rowconfigure(1, weight=0)      # For scrollbar
        
        # Welcome message
        self.welcome_frame = ctk.CTkFrame(self.main_canvas, fg_color="transparent")
        self.welcome_frame.grid(row=0, column=0, columnspan=2, rowspan=2, sticky="nsew")
        self.welcome_frame.grid_columnconfigure(0, weight=1)
        self.welcome_frame.grid_rowconfigure(0, weight=1)
        self.welcome_frame.grid_rowconfigure(1, weight=0)
        self.welcome_frame.grid_rowconfigure(2, weight=1)
        
        # Spacer
        ctk.CTkFrame(self.welcome_frame, fg_color="transparent", height=1).grid(row=0, column=0)
        
        # Welcome text container
        welcome_container = ctk.CTkFrame(self.welcome_frame, fg_color="transparent")
        welcome_container.grid(row=1, column=0, sticky="ew")
        
        ctk.CTkLabel(
            welcome_container, 
            text="Welcome to Meandros", 
            font=ctk.CTkFont(size=24, weight="normal"),
            text_color=TEXT_WHITE
        ).pack(pady=(0, 10))
        
        ctk.CTkLabel(
            welcome_container, 
            text="Load a model and select files to start analyzing", 
            font=ctk.CTkFont(size=14, weight="normal"),
            text_color=TEXT_GRAY
        ).pack()
        
        # Spacer
        ctk.CTkFrame(self.welcome_frame, fg_color="transparent", height=1).grid(row=2, column=0)

    def add_file_to_list(self, filename):
        """Add a file to the visual list with dark theme styling"""
        item_frame = ctk.CTkFrame(
            self.file_list_frame, 
            height=32,
            corner_radius=6,
            fg_color=DARK_BORDER,
            border_width=0
        )
        item_frame.pack(fill="x", padx=3, pady=1)
        item_frame.pack_propagate(False)
        
        current_index = len(self.file_items)
        
        # Checkbox
        checkbox_var = ctk.BooleanVar()
        checkbox = ctk.CTkCheckBox(
            item_frame, 
            text="", 
            variable=checkbox_var,
            width=18,
            height=18,
            corner_radius=3,
            checkmark_color=TEXT_WHITE,
            fg_color=ACCENT_PURPLE,
            hover_color=ACCENT_LIGHT_PURPLE,
            command=lambda idx=current_index: self.toggle_selection(idx)
        )
        checkbox.pack(side="left", padx=8, pady=7)
        
        # File name
        name_label = ctk.CTkLabel(
            item_frame, 
            text=filename, 
            font=ctk.CTkFont(size=12, weight="normal"),
            text_color=TEXT_WHITE,
            anchor="w"
        )
        name_label.pack(side="left", fill="x", expand=True, padx=5, pady=7)
        
        # Click to display (single click only, no double-click)
        name_label.bind("<Button-1>", lambda e, idx=current_index: self.display_file(idx))
        
        self.file_items.append({
            'frame': item_frame,
            'checkbox': checkbox,
            'checkbox_var': checkbox_var,
            'label': name_label,
            'filename': filename
        })
        
        # Auto-size the project panel
        self.auto_size_project_panel()

    def display_file(self, index):
        """Display file in the main canvas"""
        if index < len(self.object_list):
            self.display_in_canvas(self.object_list[index])

    def display_in_canvas(self, file_obj):
        """Display image in the main canvas area"""
        # Clear existing canvas content
        for widget in self.main_canvas.winfo_children():
            widget.destroy()
        
        # Update canvas title
        self.canvas_title.configure(text=f"Viewing: {file_obj.name}")
        
        # Use thresholded image if available, otherwise use original
        if hasattr(file_obj, 'thresholded_image') and file_obj.thresholded_image:
            # Use the stored thresholded image
            display_img = file_obj.thresholded_image
        else:
            # Use original image
            display_img = file_obj.image.convert("RGB")
        
        # Create the custom image viewer directly in main_canvas
        file_obj.canvas_viewer = self.create_image_viewer(self.main_canvas, display_img)
        
        # Store reference for updates
        self.current_file = file_obj
        
        # Ensure resize handle stays functional after canvas changes
        self.after(100, self.ensure_resize_handle_visible)
    
    def create_image_viewer(self, parent, image):
        """Create a custom image viewer that works within a frame"""
        # Create canvas for image display directly in parent
        canvas = tk.Canvas(
            parent,
            bg=DARK_PANEL,
            highlightthickness=0
        )
        canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Create scrollbars
        v_scrollbar = ctk.CTkScrollbar(
            parent, 
            orientation="vertical",
            command=canvas.yview,
            width=16
        )
        v_scrollbar.grid(row=0, column=1, sticky="ns", padx=(0, 5), pady=5)
        
        h_scrollbar = ctk.CTkScrollbar(
            parent, 
            orientation="horizontal",
            command=canvas.xview,
            height=16
        )
        h_scrollbar.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 5))
        
        # Configure canvas scrolling
        canvas.configure(
            xscrollcommand=h_scrollbar.set,
            yscrollcommand=v_scrollbar.set
        )
        
        # Create viewer object to store data
        viewer = type('Viewer', (), {})()
        viewer.image = image
        viewer.canvas = canvas
        viewer.width, viewer.height = image.size
        viewer.imscale = 1.0
        viewer.delta = 1.3
        
        # Create container rectangle
        viewer.container = canvas.create_rectangle(
            0, 0, viewer.width, viewer.height, width=0
        )
        
        # Bind events
        canvas.bind("<Configure>", lambda e: self.show_image_in_viewer(viewer))
        canvas.bind("<ButtonPress-1>", lambda e: self.move_from(e, viewer))
        canvas.bind("<B1-Motion>", lambda e: self.move_to(e, viewer))
        canvas.bind("<MouseWheel>", lambda e: self.wheel_zoom(e, viewer))
        
        # Show initial image
        self.show_image_in_viewer(viewer)
        
        return viewer
    
    def show_image_in_viewer(self, viewer):
        """Show image in the custom viewer"""
        canvas = viewer.canvas
        
        # Get canvas dimensions
        canvas.update_idletasks()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # Set minimum canvas size if too small
        if canvas_width < 200:
            canvas_width = 600
        if canvas_height < 200:
            canvas_height = 400
        
        # Calculate image size with reasonable limits
        image_width = max(10, int(viewer.width * viewer.imscale))
        image_height = max(10, int(viewer.height * viewer.imscale))
        
        # Limit maximum size to prevent memory issues
        max_size = 2000
        if image_width > max_size:
            scale_factor = max_size / image_width
            image_width = max_size
            image_height = int(image_height * scale_factor)
        if image_height > max_size:
            scale_factor = max_size / image_height
            image_height = max_size
            image_width = int(image_width * scale_factor)
        
        # Resize image
        if viewer.imscale == 1.0 and image_width == viewer.width and image_height == viewer.height:
            resized_image = viewer.image
        else:
            resized_image = viewer.image.resize((image_width, image_height), Image.LANCZOS)
        
        # Convert to PhotoImage
        viewer.photo = ImageTk.PhotoImage(resized_image)
        
        # Update canvas
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=viewer.photo)
        
        # Set scroll region to the actual image size
        canvas.configure(scrollregion=(0, 0, image_width, image_height))
    
    def move_from(self, event, viewer):
        """Remember previous coordinates for scrolling"""
        viewer.canvas.scan_mark(event.x, event.y)
    
    def move_to(self, event, viewer):
        """Drag canvas to the new position"""
        viewer.canvas.scan_dragto(event.x, event.y, gain=1)
    
    def wheel_zoom(self, event, viewer):
        """Zoom with mouse wheel"""
        if event.delta > 0:
            viewer.imscale *= viewer.delta
        else:
            viewer.imscale /= viewer.delta
        
        # Limit zoom
        viewer.imscale = max(0.1, min(viewer.imscale, 10.0))
        
        self.show_image_in_viewer(viewer)

    def toggle_selection(self, index):
        """Toggle selection with dark theme visual feedback"""
        if index in self.selected_indices:
            self.selected_indices.remove(index)
            self.file_items[index]['frame'].configure(fg_color=DARK_BORDER)
        else:
            self.selected_indices.append(index)
            self.file_items[index]['frame'].configure(fg_color=ACCENT_LIGHT_PURPLE)

    def get_index(self):
        """Get selected indices"""
        return self.selected_indices if self.selected_indices else []

    def update_threshold_label(self, value):
        """Update threshold value label"""
        self.threshold_value_label.configure(text=f"{int(value)}")
    
    def show_message(self, title, message):
        """Show dark theme message dialog"""
        dialog = ctk.CTkToplevel(self)
        dialog.title(title)
        dialog.geometry("400x150")
        dialog.resizable(False, False)
        dialog.configure(fg_color=DARK_PANEL)
        
        # Center the dialog
        dialog.transient(self)
        dialog.grab_set()
        
        # Configure grid
        dialog.grid_columnconfigure(0, weight=1)
        dialog.grid_rowconfigure(0, weight=1)
        
        # Message label
        message_label = ctk.CTkLabel(
            dialog, 
            text=message, 
            font=ctk.CTkFont(size=13, weight="normal"),
            text_color=TEXT_WHITE,
            wraplength=350
        )
        message_label.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        
        # OK button
        ok_button = ctk.CTkButton(
            dialog,
            text="OK",
            command=dialog.destroy,
            width=80,
            height=30,
            corner_radius=6,
            font=ctk.CTkFont(size=12, weight="normal"),
            fg_color=ACCENT_PURPLE,
            hover_color=ACCENT_LIGHT_PURPLE,
            text_color=TEXT_WHITE
        )
        ok_button.grid(row=1, column=0, pady=(0, 20))
        
        dialog.focus()
        dialog.wait_window()

    def apply_threshold_to_file(self, index):
        """Apply threshold to a specific file"""
        if index >= len(self.object_list):
            return
            
        file_obj = self.object_list[index]
        if not file_obj.roi:
            return
            
        # Get threshold settings from sidebar
        threshold_value = int(self.threshold_slider.get())
        channel = self.channel_var.get()
        color_channel = {"Red": 0, "Green": 1}[channel]
        
        # Create threshold image from original image
        current_img = file_obj.image.convert("RGB").copy()
        pixels = current_img.load()
        
        # Apply threshold to ROI pixels
        for roi_point in file_obj.roi:
            x, y = roi_point[1], roi_point[0]  # Note: roi points are (y, x)
            try:
                pixel_value = pixels[int(x), int(y)][color_channel]
                if pixel_value >= threshold_value:
                    if channel == "Red":
                        pixels[int(x), int(y)] = (255, 0, 0)  # Red highlight
                    else:
                        pixels[int(x), int(y)] = (0, 255, 0)  # Green highlight
            except (IndexError, TypeError):
                continue
        
        # Store the thresholded image
        file_obj.thresholded_image = current_img
        file_obj.threshold = threshold_value
        file_obj.channel = {"Red": 2, "Green": 1}[channel]
        
        # Update the canvas viewer if it exists
        if hasattr(file_obj, 'canvas_viewer') and file_obj.canvas_viewer:
            file_obj.canvas_viewer.image = current_img
            self.show_image_in_viewer(file_obj.canvas_viewer)
            
        # Update the file list display
        if index < len(self.file_items):
            self.file_items[index]['label'].configure(
                text=f"{file_obj.name}: {threshold_value}"
            )
            
        self.FLAG_READY_FOR_EXPORT = True

    def run(self):
        """Apply threshold using sidebar controls"""
        if not self.get_index():
            self.show_message("Alert!", "No element has been selected")
            return

        index = self.get_index()[0]

        if not self.FLAG_IMAGE_LOADED:
            self.show_message("Alert!", "No image loaded")
            return
            
        if not self.FLAG_ROI_LOADED:
            self.show_message("Alert!", "You need to generate ROI first")
            return
        
        # If file is not already displayed, display it first
        if not hasattr(self.object_list[index], 'canvas_viewer') or not self.object_list[index].canvas_viewer:
            self.display_in_canvas(self.object_list[index])
        
        # Apply threshold to the selected file
        self.apply_threshold_to_file(index)
        
        self.show_message("Information", "Threshold applied successfully")

    def load_list_files(self, option_list, index):
        """Load and process files with threshold"""
        current_img = self.object_list[index].image.convert("RGB")
        current_obj = self.object_list[index]
        color = option_list[self.channel_var.get()]
        
        if current_obj.roi is not None:
            pixels = current_img.load()
            self.object_list[index].mask_th = pixels
            threshold_value = int(self.threshold_slider.get())
            
            for i in current_obj.roi:
                x = i[1]
                y = i[0]
                if pixels[int(y), int(x)][color] >= threshold_value:
                    if self.channel_var.get() == "Red":
                        pixels[int(y), int(x)] = (255, 0, 0)
                    else:
                        pixels[int(y), int(x)] = (0, 255, 0)

            # Update canvas viewer if available
            if hasattr(current_obj, 'canvas_viewer') and current_obj.canvas_viewer:
                current_obj.canvas_viewer.image = current_img
                self.show_image_in_viewer(current_obj.canvas_viewer)
            
            current_obj.threshold = threshold_value
            self.FLAG_READY_FOR_EXPORT = True

            # Update the file list display
            self.file_items[index]['label'].configure(
                text=f"{current_obj.name}: {threshold_value}"
            )
            self.object_list[index].threshold = threshold_value
            self.show_message("Information", "Threshold Saved")
            self.object_list[index].channel = {"Red": 2, "Green": 1}[self.channel_var.get()]
            
        else:
            self.show_message("Alert!", "You need to generate ROI first")

        return None

    def draw_axis(self):
        """Draw axis"""
        if not self.get_index():
            self.show_message("Alert!", "No element has been selected")
            return
            
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
            
        try:
            print(f"Starting axis detection with {len(approx_pd)} control points")
            print(f"Control points: {approx_pd}")
            
            # Display the image in canvas if not already displayed
            if not hasattr(self.object_list[index], 'canvas_viewer') or not self.object_list[index].canvas_viewer:
                self.display_in_canvas(self.object_list[index])
            
            # Use integrated canvas editor instead of popup window
            def axis_completion_callback(ctrl_points):
                """Handle axis completion"""
                if ctrl_points and len(ctrl_points) >= 2:
                    try:
                        # Create BezierPD instance for processing
                        bezier_editor = bezierPD.BezierPD(
                            filename=image_tif, 
                            roi=self.object_list[index].roi, 
                            ctrlPoints=ctrl_points,
                            parent=self
                        )
                        
                        # Process the axis using the bezier points
                        # This simulates what bezierPD.BezierPD().run() does internally
                        axis_pd, intersect_list = bezier_editor.return_worker()
                        
                        print(f"Generated axis with {len(axis_pd) if axis_pd is not None else 0} points")
                        
                        if axis_pd is not None and len(axis_pd) > 0:
                            self.object_list[index].intersect_list = intersect_list
                            self.object_list[index].curve = set(map(tuple, axis_pd))
                            self.object_list[index].axis_list = ctrl_points
                            self.FLAG_AXIS_SAVED = True
                            self.show_message("Information", "Axis Saved")
                        else:
                            self.show_message("Alert!", "No axis generated - please add more control points")
                    except Exception as e:
                        print(f"Error in axis generation: {e}")
                        import traceback
                        traceback.print_exc()
                        self.show_message("Alert!", f"Error generating axis: {str(e)}")
                else:
                    self.show_message("Alert!", "At least 2 control points required for axis")
            
            # Create integrated canvas editor
            self.current_canvas_tool = IntegratedCanvasEditor(
                self.object_list[index].canvas_viewer, 
                image_tif, 
                approx_pd, 
                self
            )
            self.current_canvas_tool.run(axis_completion_callback)
            
        except Exception as e:
            print(f"Error in axis generation: {e}")
            import traceback
            traceback.print_exc()
            self.show_message("Alert!", f"Error generating axis: {str(e)}")

    def exclude_regions(self):
        """Exclude regions"""
        if not self.get_index():
            self.show_message("Alert!", "No element has been selected")
            return
            
        index = self.get_index()[0]
        filename = self.object_list[index].path
        
        # Display the image in canvas if not already displayed
        if not hasattr(self.object_list[index], 'canvas_viewer') or not self.object_list[index].canvas_viewer:
            self.display_in_canvas(self.object_list[index])
        
        # Use integrated canvas editor instead of popup window
        def exclusion_completion_callback(ctrl_points):
            """Handle exclusion completion"""
            if ctrl_points:
                # Create exclusion polygons from control points
                try:
                    # Create exclusion instance for processing
                    excl = exclusion.Exclusion(filename=filename)
                    excl.ctrlPoints = ctrl_points
                    
                    # Process exclusion using the control points
                    # This simulates what exclusion.Exclusion().run() does internally
                    polygons = excl.return_worker()
                    
                    self.object_list[index].exclude = polygons
                    
                    if self.object_list[index].exclude is not None:
                        self.show_message("Information", "Exclusion Regions Saved")
                    else:
                        self.show_message("Alert!", "Failed to create exclusion regions")
                except Exception as e:
                    print(f"Error in exclusion processing: {e}")
                    self.show_message("Alert!", f"Error processing exclusion: {str(e)}")
            else:
                self.show_message("Information", "No exclusion regions defined")
        
        # Get existing exclusion points if any
        existing_points = []
        if hasattr(self.object_list[index], 'exclude') and self.object_list[index].exclude:
            # If exclusion already exists, we could try to extract points from it
            # For now, start with empty points
            existing_points = []
        
        # Create integrated canvas editor
        self.current_canvas_tool = IntegratedCanvasEditor(
            self.object_list[index].canvas_viewer, 
            filename, 
            existing_points, 
            self
        )
        self.current_canvas_tool.run(exclusion_completion_callback)

    def roi_detection(self):
        """ROI detection"""
        if not self.get_index():
            self.show_message("Alert!", "No element has been selected")
            return
            
        index = self.get_index()[0]
        
        if self.model:
            filename = self.object_list[index].path
            if filename[-3:] == "tif":
                image_tif = fix_tif(filename)
            else:
                image_tif = filename
                
            # Display the image in canvas if not already displayed
            if not hasattr(self.object_list[index], 'canvas_viewer') or not self.object_list[index].canvas_viewer:
                self.display_in_canvas(self.object_list[index])
                
            if not self.object_list[index].ctrlPoints:
                approx_roi, class_id = roi_detect.model_worker(image_tif, self.model)
                self.object_list[index].class_id = class_id
            else:
                approx_roi = self.object_list[index].ctrlPoints
                
            # Use integrated canvas editor instead of popup window
            def roi_completion_callback(ctrl_points):
                """Handle ROI completion"""
                if ctrl_points:
                    # Create ROI from control points
                    roi_array = np.array(ctrl_points, dtype=np.int32)
                    
                    # Use image shape to create mask
                    img_shape = self.object_list[index].image.size[::-1]  # PIL uses (width, height), cv2 uses (height, width)
                    
                    # Create mask from ROI
                    mask = np.transpose(
                        np.nonzero(
                            cv2.fillPoly(
                                np.zeros(img_shape, np.uint8), [roi_array], (255, 0, 0)
                            )
                        )
                    )
                    
                    # Create result set
                    roi = set(map(tuple, mask[:, [1, 0]]))
                    
                    self.object_list[index].roi = roi
                    self.object_list[index].ctrlPoints = ctrl_points
                    
                    if (self.object_list[index].roi is not None and 
                        self.object_list[index].ctrlPoints is not None):
                        self.FLAG_ROI_LOADED = True
                        self.show_message("Information", "ROI Saved")
                else:
                    self.show_message("Alert!", "No ROI points defined")
            
            # Create integrated canvas editor
            self.current_canvas_tool = IntegratedCanvasEditor(
                self.object_list[index].canvas_viewer, 
                image_tif, 
                approx_roi, 
                self
            )
            self.current_canvas_tool.run(roi_completion_callback)
            
        else:
            self.show_message("Alert!", "You must select a model first")

    def get_profile(self, indexes=None):
        """Get profile analysis"""
        if self.FLAG_AXIS_SAVED:
            try:
                if not indexes:
                    if not self.get_index():
                        self.show_message("Alert!", "No element has been selected")
                        return
                    index = self.get_index()[0]
                    filename = self.object_list[index].path
                    
                    # Validate required data
                    if self.object_list[index].curve is None:
                        self.show_message("Alert!", "No axis curve found. Please generate axis first.")
                        return
                    if self.object_list[index].roi is None:
                        self.show_message("Alert!", "No ROI found. Please detect ROI first.")
                        return
                    
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
                    
                    print(f"Profile analysis for index {index}:")
                    print(f"  Curve points: {len(p_d) if p_d else 0}")
                    print(f"  ROI points: {len(conj_mask) if conj_mask else 0}")
                    print(f"  Channel: {channel}")
                    print(f"  Threshold: {threshold}")
                    print(f"  Landmarks: {landmarks}")
                    
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
                    
                    print(f"  Profile generated: {self.object_list[index].profile is not None}")
                    if self.object_list[index].profile is not None:
                        print(f"  Profile shape: {self.object_list[index].profile.shape}")
                    
                else:
                    for index in indexes:
                        filename = self.object_list[index].path
                        
                        # Validate required data
                        if self.object_list[index].curve is None:
                            self.show_message("Alert!", f"No axis curve found for {self.object_list[index].name}. Please generate axis first.")
                            continue
                        if self.object_list[index].roi is None:
                            self.show_message("Alert!", f"No ROI found for {self.object_list[index].name}. Please detect ROI first.")
                            continue
                        
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
                        
                        print(f"Profile analysis for index {index} ({self.object_list[index].name}):")
                        print(f"  Curve points: {len(p_d) if p_d else 0}")
                        print(f"  ROI points: {len(conj_mask) if conj_mask else 0}")
                        print(f"  Channel: {channel}")
                        print(f"  Threshold: {threshold}")
                        
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
                        
                        print(f"  Profile generated: {self.object_list[index].profile is not None}")
                        if self.object_list[index].profile is not None:
                            print(f"  Profile shape: {self.object_list[index].profile.shape}")
                        
            except Exception as e:
                import traceback
                print(f"Error in get_profile: {e}")
                traceback.print_exc()
                self.show_message("Alert!", f"Profiling Failed: {str(e)}")
        else:
            self.show_message("Alert!", "You need to generate Axis first")

    def landmarks(self):
        """Process landmarks"""
        if not self.get_index():
            self.show_message("Alert!", "No element has been selected")
            return
            
        index = self.get_index()[0]
        filename = self.object_list[index].path
        
        # Display the image in canvas if not already displayed
        if not hasattr(self.object_list[index], 'canvas_viewer') or not self.object_list[index].canvas_viewer:
            self.display_in_canvas(self.object_list[index])
        
        # Use integrated canvas editor instead of popup window
        def landmarks_completion_callback(ctrl_points):
            """Handle landmarks completion"""
            if ctrl_points:
                # Create landmarks instance for processing
                try:
                    landmarks_inst = landmarks.Landmarks(
                        filename=filename, ctrl_points=ctrl_points
                    )
                    
                    # Process landmarks using the control points
                    # This simulates what landmarks.Landmarks().run() does internally
                    landmark_result = landmarks_inst.return_worker()
                    
                    self.object_list[index].landmarks = landmark_result
                    
                    if self.object_list[index].landmarks is not None:
                        self.show_message("Information", "Landmarks Saved")
                    else:
                        self.show_message("Alert!", "Failed to create landmarks")
                except Exception as e:
                    print(f"Error in landmarks processing: {e}")
                    self.show_message("Alert!", f"Error processing landmarks: {str(e)}")
            else:
                self.show_message("Information", "No landmarks defined")
        
        # Get existing landmark points if any
        if self.object_list[index].landmarks is not None:
            landmark_pts = self.object_list[index].landmarks
        else:
            landmark_pts = []
            
        # Create integrated canvas editor
        self.current_canvas_tool = IntegratedCanvasEditor(
            self.object_list[index].canvas_viewer, 
            filename, 
            landmark_pts, 
            self
        )
        self.current_canvas_tool.run(landmarks_completion_callback)

    def amputation_plane(self):
        """Process amputation plane"""
        if not self.get_index():
            self.show_message("Alert!", "No element has been selected")
            return
            
        index = self.get_index()[0]
        filename = self.object_list[index].path
        
        if filename[-3:] == "tif":
            image_tif = fix_tif(filename)
        else:
            image_tif = filename

        if self.object_list[index].amp_plane is None:
            amp_plane = []
        else:
            amp_plane = list(self.object_list[index].amp_plane)
            
        try:
            print(f"Starting amputation plane with {len(amp_plane)} control points")
            
            # Display the image in canvas if not already displayed
            if not hasattr(self.object_list[index], 'canvas_viewer') or not self.object_list[index].canvas_viewer:
                self.display_in_canvas(self.object_list[index])
            
            # Use integrated canvas editor instead of popup window
            def amputation_completion_callback(ctrl_points):
                """Handle amputation plane completion"""
                if ctrl_points and len(ctrl_points) >= 2:
                    try:
                        # Create BezierPD instance for processing
                        bezier_editor = bezierPD.BezierPD(
                            filename=image_tif, 
                            roi=self.object_list[index].roi, 
                            ctrlPoints=ctrl_points,
                            parent=self
                        )
                        
                        # Process the amputation plane using the bezier points
                        # This simulates what bezierPD.BezierPD().run() does internally
                        ap_plane, intersect_ap_list = bezier_editor.return_worker()
                        
                        print(f"Generated amputation plane with {len(ap_plane) if ap_plane is not None else 0} points")
                        
                        if ap_plane is not None and len(ap_plane) > 0:
                            self.object_list[index].amp_plane = set(map(tuple, ap_plane))
                            self.show_message("Information", "Amputation Plane Saved")
                        else:
                            self.show_message("Alert!", "No amputation plane generated")
                    except Exception as e:
                        print(f"Error in amputation plane: {e}")
                        import traceback
                        traceback.print_exc()
                        self.show_message("Alert!", f"Error generating amputation plane: {str(e)}")
                else:
                    self.show_message("Alert!", "At least 2 control points required for amputation plane")
            
            # Create integrated canvas editor
            self.current_canvas_tool = IntegratedCanvasEditor(
                self.object_list[index].canvas_viewer, 
                image_tif, 
                amp_plane, 
                self
            )
            self.current_canvas_tool.run(amputation_completion_callback)
            
        except Exception as e:
            print(f"Error in amputation plane: {e}")
            import traceback
            traceback.print_exc()
            self.show_message("Alert!", f"Error generating amputation plane: {str(e)}")

    def delete(self):
        """Delete selected items"""
        if not self.selected_indices:
            self.show_message("Alert!", "No elements selected")
            return
            
        # Sort indices in reverse order to avoid index shifting issues
        for index in sorted(self.selected_indices, reverse=True):
            if index < len(self.object_list):
                try:
                    # Close canvas viewer if it exists
                    if hasattr(self.object_list[index], 'canvas_viewer') and self.object_list[index].canvas_viewer:
                        # Clean up the viewer object (no cerrar method needed for our custom viewer)
                        self.object_list[index].canvas_viewer = None
                except (AttributeError, Exception):
                    pass
                del self.object_list[index]
                
                # Remove from visual list
                self.file_items[index]['frame'].destroy()
                del self.file_items[index]
        
        self.selected_indices.clear()
        
        # Auto-size the project panel
        self.auto_size_project_panel()
        
        # Clear canvas if no files remain
        if not self.object_list:
            self.clear_canvas()
    
    def clear_canvas(self):
        """Clear the canvas and show welcome message"""
        # Clear existing canvas content
        for widget in self.main_canvas.winfo_children():
            widget.destroy()
        
        # Update canvas title
        self.canvas_title.configure(text="Canvas")
        
        # Reset grid configuration
        self.main_canvas.grid_columnconfigure(0, weight=1)
        self.main_canvas.grid_columnconfigure(1, weight=0)
        self.main_canvas.grid_rowconfigure(0, weight=1)
        self.main_canvas.grid_rowconfigure(1, weight=0)
        
        # Show welcome message
        self.welcome_frame = ctk.CTkFrame(self.main_canvas, fg_color="transparent")
        self.welcome_frame.grid(row=0, column=0, columnspan=2, rowspan=2, sticky="nsew")
        self.welcome_frame.grid_columnconfigure(0, weight=1)
        self.welcome_frame.grid_rowconfigure(0, weight=1)
        self.welcome_frame.grid_rowconfigure(1, weight=0)
        self.welcome_frame.grid_rowconfigure(2, weight=1)
        
        # Spacer
        ctk.CTkFrame(self.welcome_frame, fg_color="transparent", height=1).grid(row=0, column=0)
        
        # Welcome text container
        welcome_container = ctk.CTkFrame(self.welcome_frame, fg_color="transparent")
        welcome_container.grid(row=1, column=0, sticky="ew")
        
        ctk.CTkLabel(
            welcome_container, 
            text="Welcome to Meandros", 
            font=ctk.CTkFont(size=24, weight="normal"),
            text_color=TEXT_WHITE
        ).pack(pady=(0, 10))
        
        ctk.CTkLabel(
            welcome_container, 
            text="Load a model and select files to start analyzing", 
            font=ctk.CTkFont(size=14, weight="normal"),
            text_color=TEXT_GRAY
        ).pack()
        
        # Spacer
        ctk.CTkFrame(self.welcome_frame, fg_color="transparent", height=1).grid(row=2, column=0)
    
    def create_project_panel(self):
        """Create the right panel for project files"""
        # Panel header
        project_header = ctk.CTkFrame(self.project_panel, height=50, corner_radius=0, fg_color=DARK_PANEL)
        project_header.pack(fill="x", padx=(8, 0), pady=0)  # Left padding for resize handle
        project_header.pack_propagate(False)
        
        header_content = ctk.CTkFrame(project_header, fg_color="transparent")
        header_content.pack(fill="x", padx=10, pady=12)
        
        ctk.CTkLabel(
            header_content, 
            text="Project Files", 
            font=ctk.CTkFont(size=15, weight="normal"),
            text_color=TEXT_WHITE
        ).pack(side="left")
        
        self.delete_btn = ctk.CTkButton(
            header_content, 
            text="Delete", 
            command=self.delete,
            width=55,
            height=26,
            corner_radius=6,
            font=ctk.CTkFont(size=11, weight="normal"),
            fg_color=DARK_RED,
            hover_color="#D70015",
            text_color="white"
        )
        self.delete_btn.pack(side="right")
        
        # File list
        self.file_list_frame = ctk.CTkScrollableFrame(
            self.project_panel, 
            fg_color="transparent",
            scrollbar_button_color=ACCENT_PURPLE,
            scrollbar_button_hover_color=ACCENT_LIGHT_PURPLE
        )
        self.file_list_frame.pack(fill="both", expand=True, padx=(8, 10), pady=(0, 10))  # Left padding for resize handle
    
    def filedialog1(self):
        """File dialog for signal channel"""
        try:
            filename = filedialog.askopenfilename(
                title="Select Signal Channel",
                multiple=True,
                filetypes=(
                    ("jpg files", "*.jpg"),
                    ("tif files", "*.tif"),
                    ("all files", "*.*"),
                ),
            )
            if len(filename):
                for k in enumerate(filename):
                    finalname = os.path.splitext(os.path.split(k[1])[1])[0]
                    if k[1][-3:] == "tif":
                        if not os.path.exists("jpg_images"):
                            os.makedirs("jpg_images")
                        image = tif.imread(k[1])
                        if image.ndim > 2:
                            bright = image[1, :, :]
                            marker = image[0, :, :]
                        io.imsave(f"""jpg_images/{finalname}_brightfield.jpg""", bright)
                        io.imsave(f"""jpg_images/{finalname}_marker.jpg""", marker)
                    
                    self.object_list.append(File(finalname, k[1]))
                    self.add_file_to_list(finalname)
                    
                self.FLAG_IMAGE_LOADED = True
            else:
                return
        except AttributeError:
            pass

    def filedialog2(self):
        """File dialog for bright field"""
        if not self.get_index():
            self.show_message("Alert!", "No element has been selected")
            return

        index = self.get_index()[0]
        filename = filedialog.askopenfilename(
            title="Select Bright Field",
            multiple=True,
            filetypes=(
                ("tif files", "*.tif"),
                ("jpg files", "*.jpg"),
                ("all files", "*.*"),
            ),
        )
        if len(filename):
            self.object_list[index].alpha = filename

    def export_axis(self):
        """Export axis to CSV"""
        if not self.get_index():
            self.show_message("Alert!", "No element has been selected")
            return
            
        index = self.get_index()[0]
        f = open("Axis_Extractor.csv", "a+")
        writer = csv.writer(f)
        name = self.object_list[index].name
        
        for x in self.object_list[index].ctrlPoints:
            r = [x[0], x[1], name]
            writer.writerow(r)
        f.close()

    def export_all(self):
        """Export project"""
        if not self.get_index():
            self.show_message("Alert!", "No element has been selected")
            return
            
        index = self.get_index()[0]
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
        """Import project"""
        try:
            open_path = filedialog.askopenfilename(
                initialdir="", initialfile="data", defaultextension=".p"
            )
            list_files_import = pickle.load(open(open_path, "rb"))
            
            for f in list_files_import:
                self.object_list.append(f)
                self.add_file_to_list(f.name)
                
                if f.roi is not None:
                    self.FLAG_ROI_LOADED = True
                if f.curve is not None:
                    self.FLAG_AXIS_SAVED = True
                    
            self.FLAG_IMAGE_LOADED = True
        except Exception as e:
            print(e)

    def load_model_selected(self):
        """Load selected model"""
        self.model = weight_loader(self.model_var.get())
        
        if self.model is not None:
            self.show_message("Information", f"{self.model_var.get()} loaded")

    def statistic_group_1(self, indexes):
        """Statistics for group 1"""
        # Validate that all selected items have profiles
        valid_profiles = []
        for index in indexes:
            if self.object_list[index].profile is None:
                self.show_message("Alert!", f"No profile found for {self.object_list[index].name}. Please generate profile first.")
                return None, None, None, None
            valid_profiles.append(self.object_list[index].profile)
        
        if len(indexes) == 1:
            group = valid_profiles[0].sort_values(
                by=["PD"], ascending=True
            )
        else:
            group = pd.concat(
                valid_profiles,
                ignore_index=True,
                sort=False,
            ).sort_values(by=["PD"], ascending=True)
            
        # Update visual selection
        for i, item in enumerate(self.file_items):
            if i in indexes:
                item['frame'].configure(fg_color=ACCENT_LIGHT_PURPLE)  # Highlight selected
            else:
                item['frame'].configure(fg_color="transparent")
                
        self.group_1 = group
        self.group_1.to_csv("ctrl-group.csv")
        
        elbows = [self.object_list[i].elbow_ratio for i in indexes]
        wrists = [self.object_list[i].wrist_ratio for i in indexes]
        
        # Filter out None values for mean calculation
        valid_elbows = [e for e in elbows if e is not None]
        valid_wrists = [w for w in wrists if w is not None]
        
        elbow_mean = np.mean(valid_elbows) if valid_elbows else None
        wrist_mean = np.mean(valid_wrists) if valid_wrists else None
        elbow_sem = sem(valid_elbows) if valid_elbows else None
        wrist_sem = sem(valid_wrists) if valid_wrists else None
        
        return elbow_mean, elbow_sem, wrist_mean, wrist_sem

    def reports(self):
        """Generate reports"""
        indexes = self.get_index()
        if len(indexes) >= 1:
            self.get_profile(indexes)
            print(indexes)
            e, e_sem, w, w_sem = self.statistic_group_1(indexes)
            
            # Check if statistics were successfully calculated
            if e is not None or w is not None:
                return sdp.mean_sd_plots(self.group_1, e, e_sem, w, w_sem)
            else:
                self.show_message("Alert!", "Failed to calculate statistics. Please ensure all selected items have valid profiles.")
        else:
            self.show_message("Alert!", "No element has been selected")

    def start_resize(self, event):
        """Start resizing the project panel"""
        print(f"Starting resize from width {self.project_panel_width}")
        self.resizing = True
        self.manual_resize_active = True  # Prevent auto-sizing
        self.resize_start_x = event.x_root
        self.resize_start_width = self.project_panel_width

    def do_resize(self, event):
        """Resize the project panel"""
        if self.resizing:
            delta_x = event.x_root - self.resize_start_x
            new_width = self.resize_start_width - delta_x  # Subtract because we're dragging left edge
            # Set reasonable bounds for usability
            if new_width >= 120 and new_width <= 800:  # Minimum and maximum for usability
                self.project_panel_width = new_width
                try:
                    # Use update_idletasks to ensure proper layout
                    self.project_panel.configure(width=new_width)
                    self.update_idletasks()
                    
                    # Force geometry update
                    self.geometry(f"{self.winfo_width()}x{self.winfo_height()}")
                except Exception as e:
                    print(f"Resize error: {e}")
                    # Fallback to basic configuration
                    self.project_panel.configure(width=new_width)

    def stop_resize(self, event):
        """Stop resizing the project panel"""
        print(f"Stopping resize at width {self.project_panel_width}")
        self.resizing = False
        # Keep manual_resize_active = True to prevent auto-sizing until reset
        self.resize_start_x = 0
        self.resize_start_width = 0
        
        # Force a final update to ensure the layout is stable
        try:
            self.update_idletasks()
            # Ensure resize handle stays visible
            self.after(50, self.ensure_resize_handle_visible)
        except Exception as e:
            print(f"Error in stop_resize: {e}")
    
    def reset_to_auto_size(self, event):
        """Reset to auto-sizing (double-click on resize handle)"""
        self.manual_resize_active = False
        self.auto_size_project_panel()
        self.show_message("Information", "Panel reset to auto-size. It will now resize automatically with file list.")
    
    def ensure_resize_handle_visible(self):
        """Ensure the resize handle remains visible and functional"""
        if hasattr(self, 'resize_handle') and self.resize_handle:
            try:
                # Make sure the handle is on top
                self.resize_handle.lift()
                # Ensure it's properly placed
                self.resize_handle.place(x=0, y=0, relheight=1.0)
                print("Resize handle visibility ensured")
            except Exception as e:
                print(f"Error ensuring resize handle visibility: {e}")
            
    def calculate_optimal_panel_width(self):
        """Calculate optimal width based on filename lengths"""
        if not self.object_list:
            return 280  # Default width
        
        filename_lengths = [len(obj.name) for obj in self.object_list]
        if not filename_lengths:
            return 280
            
        # Calculate median length
        filename_lengths.sort()
        n = len(filename_lengths)
        median_length = filename_lengths[n // 2] if n % 2 == 1 else (filename_lengths[n // 2 - 1] + filename_lengths[n // 2]) / 2
        
        # Estimate width: ~8 pixels per character + margins
        estimated_width = int(median_length * 8 + 100)  # 100px for margins, checkboxes, and UI elements
        
        # Set reasonable minimum but no maximum constraint
        return max(180, estimated_width)
    
    def auto_size_project_panel(self):
        """Auto-size the project panel based on current filenames"""
        # Don't auto-size if user has manually resized
        if self.manual_resize_active:
            return
            
        optimal_width = self.calculate_optimal_panel_width()
        self.project_panel_width = optimal_width
        self.project_panel.configure(width=optimal_width)


def main():
    """Main function"""
    if os.environ.get("DISPLAY", "") == "":
        os.environ.__setitem__("DISPLAY", ":0.0")
    
    root = Root()
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    return root


if __name__ == "__main__":
    main() 