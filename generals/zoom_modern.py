import random
import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk


class AutoScrollbar(ctk.CTkScrollbar):
    """A modern scrollbar that hides itself if it's not needed."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
            super().set(lo, hi)


class Zoom_Advanced(ctk.CTkFrame):
    """Advanced zoom of the image with modern styling"""

    open_windows = []

    def __init__(self, mainframe, im):
        """Initialize the main Frame"""
        self.open_windows.append(self)

        super().__init__(master=mainframe)
        
        # Configure the main frame
        self.master.title("Image Viewer")
        self.master.geometry("1000x600")
        self.master.minsize(800, 500)
        self.master.protocol("WM_DELETE_WINDOW", self.cerrar)
        
        # Configure grid weights
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)
        
        # Create main container
        self.main_container = ctk.CTkFrame(self.master)
        self.main_container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Configure container grid
        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)
        
        # Create toolbar
        self.create_toolbar()
        
        # Create canvas container
        self.canvas_container = ctk.CTkFrame(self.main_container)
        self.canvas_container.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Configure canvas container grid
        self.canvas_container.grid_rowconfigure(0, weight=1)
        self.canvas_container.grid_columnconfigure(0, weight=1)
        
        # Create canvas (using tk.Canvas as CTk doesn't have a canvas widget)
        self.canvas = tk.Canvas(
            self.canvas_container,
            bg='#2b2b2b',
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        
        # Create scrollbars
        self.v_scrollbar = ctk.CTkScrollbar(
            self.canvas_container, 
            orientation="vertical",
            command=self.canvas.yview
        )
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")
        
        self.h_scrollbar = ctk.CTkScrollbar(
            self.canvas_container, 
            orientation="horizontal",
            command=self.canvas.xview
        )
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        # Configure canvas scrolling
        self.canvas.configure(
            xscrollcommand=self.h_scrollbar.set,
            yscrollcommand=self.v_scrollbar.set
        )
        
        # Initialize image variables
        self.image = im
        self.width, self.height = self.image.size
        self.imscale = 1.0
        self.delta = 1.3
        
        # Create container rectangle
        self.container = self.canvas.create_rectangle(
            0, 0, self.width, self.height, width=0
        )
        
        # Bind events
        self.bind_events()
        
        # Show initial image
        self.show_image()

    def create_toolbar(self):
        """Create modern toolbar with controls"""
        toolbar = ctk.CTkFrame(self.main_container, height=50)
        toolbar.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        toolbar.grid_columnconfigure(2, weight=1)
        
        # Zoom controls
        zoom_frame = ctk.CTkFrame(toolbar)
        zoom_frame.grid(row=0, column=0, padx=5, pady=5)
        
        ctk.CTkLabel(zoom_frame, text="Zoom:", font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=5)
        
        self.zoom_in_btn = ctk.CTkButton(
            zoom_frame, 
            text="➕", 
            command=self.zoom_in,
            width=30,
            height=30,
            font=ctk.CTkFont(size=16)
        )
        self.zoom_in_btn.pack(side="left", padx=2)
        
        self.zoom_out_btn = ctk.CTkButton(
            zoom_frame, 
            text="➖", 
            command=self.zoom_out,
            width=30,
            height=30,
            font=ctk.CTkFont(size=16)
        )
        self.zoom_out_btn.pack(side="left", padx=2)
        
        self.reset_btn = ctk.CTkButton(
            zoom_frame, 
            text="🔄", 
            command=self.reset_zoom,
            width=30,
            height=30,
            font=ctk.CTkFont(size=16)
        )
        self.reset_btn.pack(side="left", padx=2)
        
        # Image info
        info_frame = ctk.CTkFrame(toolbar)
        info_frame.grid(row=0, column=1, padx=5, pady=5)
        
        self.info_label = ctk.CTkLabel(
            info_frame, 
            text=f"Size: {self.width}x{self.height} | Scale: {self.imscale:.2f}x",
            font=ctk.CTkFont(size=12)
        )
        self.info_label.pack(padx=10, pady=5)
        
        # Close button
        close_frame = ctk.CTkFrame(toolbar)
        close_frame.grid(row=0, column=3, padx=5, pady=5)
        
        self.close_btn = ctk.CTkButton(
            close_frame, 
            text="✖", 
            command=self.cerrar,
            width=30,
            height=30,
            font=ctk.CTkFont(size=16),
            fg_color="#d32f2f",
            hover_color="#b71c1c"
        )
        self.close_btn.pack(padx=5)

    def bind_events(self):
        """Bind mouse and keyboard events"""
        self.canvas.bind("<Configure>", self.show_image)
        self.canvas.bind("<ButtonPress-1>", self.move_from)
        self.canvas.bind("<B1-Motion>", self.move_to)
        self.canvas.bind("<MouseWheel>", self.wheel)
        self.canvas.bind("<Button-5>", self.wheel)
        self.canvas.bind("<Button-4>", self.wheel)
        self.canvas.bind("<Double-Button-1>", self.double_click)
        self.canvas.bind("<B3-Motion>", self.get_coord)
        
        # Keyboard shortcuts
        self.master.bind("<Control-plus>", lambda e: self.zoom_in())
        self.master.bind("<Control-minus>", lambda e: self.zoom_out())
        self.master.bind("<Control-0>", lambda e: self.reset_zoom())
        self.master.focus_set()

    def zoom_in(self):
        """Zoom in"""
        self.imscale *= self.delta
        self.show_image()
        self.update_info()

    def zoom_out(self):
        """Zoom out"""
        self.imscale /= self.delta
        self.show_image()
        self.update_info()

    def reset_zoom(self):
        """Reset zoom to fit"""
        self.imscale = 1.0
        self.show_image()
        self.update_info()

    def update_info(self):
        """Update info label"""
        self.info_label.configure(
            text=f"Size: {self.width}x{self.height} | Scale: {self.imscale:.2f}x"
        )

    def double_click(self, event):
        """Handle double click"""
        index = self.open_windows.index(self)
        if index + 1 < len(self.open_windows):
            self.open_windows[index + 1].image = self.image
            self.open_windows[index + 1].show_image()

    def get_coord(self, event):
        """Get coordinates and mark pixel"""
        # Convert canvas coordinates to image coordinates
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # Convert to image coordinates
        x_img = int(x / self.imscale)
        y_img = int(y / self.imscale)
        
        if 0 <= x_img < self.width and 0 <= y_img < self.height:
            curr = self.image.copy().convert("RGB")
            pixels = curr.load()
            pixels[x_img, y_img] = (255, 255, 255)
            self.image = curr
            self.show_image()

    def move_from(self, event):
        """Remember previous coordinates for scrolling"""
        self.canvas.scan_mark(event.x, event.y)

    def move_to(self, event):
        """Drag (move) canvas to the new position"""
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.show_image()

    def wheel(self, event):
        """Zoom with mouse wheel"""
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # Respond to Linux or Windows wheel event
        if event.num == 4 or event.delta > 0:
            self.imscale *= self.delta
        if event.num == 5 or event.delta < 0:
            self.imscale /= self.delta
            
        # Restrict scale
        self.imscale = max(0.1, min(10.0, self.imscale))
        
        # Update scroll region
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.show_image()
        self.update_info()

    def scroll_y(self, *args):
        """Scroll canvas vertically"""
        self.canvas.yview(*args)
        self.show_image()

    def scroll_x(self, *args):
        """Scroll canvas horizontally"""
        self.canvas.xview(*args)
        self.show_image()

    def show_image(self, event=None):
        """Show image on the Canvas with modern styling"""
        bbox1 = self.canvas.bbox(self.container)
        bbox2 = (self.canvas.canvasx(0), self.canvas.canvasy(0),
                 self.canvas.canvasx(self.canvas.winfo_width()),
                 self.canvas.canvasy(self.canvas.winfo_height()))
        
        bbox = [min(bbox1[0], bbox2[0]), min(bbox1[1], bbox2[1]),
                max(bbox1[2], bbox2[2]), max(bbox1[3], bbox2[3])]
        
        if bbox[0] == bbox2[0] and bbox[2] == bbox2[2]:
            bbox[0] = bbox1[0]
            bbox[2] = bbox1[2]
        if bbox[1] == bbox2[1] and bbox[3] == bbox2[3]:
            bbox[1] = bbox1[1]
            bbox[3] = bbox1[3]
            
        self.canvas.delete("image")
        
        x1 = max(bbox2[0] - bbox1[0], 0)
        y1 = max(bbox2[1] - bbox1[1], 0)
        x2 = min(bbox2[2], bbox1[2]) - bbox1[0]
        y2 = min(bbox2[3], bbox1[3]) - bbox1[1]
        
        if int(x2 - x1) > 0 and int(y2 - y1) > 0:
            x = min(int(x2 / self.imscale), self.width)
            y = min(int(y2 / self.imscale), self.height)
            
            image = self.image.crop((
                int(x1 / self.imscale), 
                int(y1 / self.imscale), 
                x, 
                y
            ))
            
            imagetk = ImageTk.PhotoImage(
                image.resize((int(x2 - x1), int(y2 - y1)), Image.Resampling.LANCZOS)
            )
            
            imageid = self.canvas.create_image(
                max(bbox2[0], bbox1[0]),
                max(bbox2[1], bbox1[1]),
                anchor="nw",
                image=imagetk,
                tags="image"
            )
            
            self.canvas.lower(imageid)
            self.canvas.imagetk = imagetk
            
        # Update scroll region
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def cerrar(self):
        """Close window"""
        if self in self.open_windows:
            self.open_windows.remove(self)
        self.master.destroy() 