"""
This module provides an interactive interface for control point manipulation on images.

Control Point Operations:

    Addition:
        Activate insertion mode by pressing 'i' key or clicking Insert Point button
        Left-click desired image coordinates to place control points
        Multiple points may be added sequentially

    Translation:
        Activate movement mode by pressing 'm' key or clicking Move Point button
        Click and drag control points to new positions using left mouse button

    Deletion:
        Activate deletion mode by pressing 'd' key or clicking Delete Point button
        Click on a control point to remove it

    Termination:
        Press ESC key or close the window to exit the application

The interface allows precise placement and adjustment of control points for image analysis tasks.
"""

import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image as PILImage, ImageTk
import tkinter as tk


class App(ctk.CTkToplevel):
    """
    Modern customtkinter interface for control point manipulation on images.
    Args:
        filename: str
            Path to the image file.
        parent: ctk.CTk
            Parent window for the toplevel
    """

    def __init__(self, filename=None, parent=None):
        # Initialize basic attributes first for backward compatibility
        self.filename = filename
        self.ctrlPoints = []
        self.finalpoints = []
        self.INSERT_FLAG = False
        self.MOVE_FLAG = False
        self.DELETE_FLAG = False
        self.ACTIVE_MOV_FLAG = False
        self.SHOW_LINES = False
        self.ind = None
        self.roi = None
        self.result = None
        
        # Colors (RGB for PIL/tkinter)
        self.RED = "#FF0000"
        self.WHITE = "#FFFFFF"
        self.BLUE = "#0000FF"
        self.ORANGE = "#FFA500"
        self.YELLOW = "#FFFF00"
        self.GREEN = "#00FF00"

        # Configuration
        self.thickness = 8
        self.epsilon = 15
        self.scale_factor = 1.0
        
        # Only initialize the GUI if this is being used as a standalone window
        # (not when being inherited by other classes)
        if self.__class__ == App:
            self._initialize_gui(parent)

    def _initialize_gui(self, parent):
        """Initialize the GUI components"""
        super().__init__(parent)
        
        # Window configuration
        self.title("Control Point Editor")
        self.geometry("1000x700")
        self.minsize(800, 600)

        # Load image first
        if self.filename:
            self.load_image()

        # Create UI
        self.create_widgets()
        
        # Load image into canvas after widgets are created
        if hasattr(self, 'photo') and hasattr(self, 'canvas') and self.photo:
            try:
                self.canvas_image = self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
                self.canvas.configure(scrollregion=self.canvas.bbox("all"))
                # Force update
                self.canvas.update_idletasks()
            except Exception as e:
                print(f"Error displaying image: {e}")
        elif self.filename:
            # Try to load image again
            self.load_image()
        
        # Draw existing control points if any
        if self.ctrlPoints:
            self.redraw_points()
        
        # Bind keyboard events
        self.bind("<KeyPress>", self.on_key_press)
        self.focus_set()

    def load_image(self):
        """Load and prepare the image"""
        if not self.filename:
            return
            
        try:
            # Load with OpenCV then convert to PIL
            cv_img = cv2.imread(self.filename)
            if cv_img is None:
                print(f"Could not load image: {self.filename}")
                return
            
            # Convert BGR to RGB
            cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            self.original_image = PILImage.fromarray(cv_img_rgb)
            self.current_image = self.original_image.copy()
            
            # Calculate scale factor to fit in window
            max_width, max_height = 800, 500
            img_width, img_height = self.original_image.size
            
            scale_x = max_width / img_width
            scale_y = max_height / img_height
            self.scale_factor = min(scale_x, scale_y, 1.0)
            
            # Resize image for display
            display_width = int(img_width * self.scale_factor)
            display_height = int(img_height * self.scale_factor)
            
            self.display_image = self.original_image.resize((display_width, display_height), PILImage.Resampling.LANCZOS)
            # Keep a reference to prevent garbage collection
            self.photo = ImageTk.PhotoImage(self.display_image)
            
            # Update canvas if it exists
            if hasattr(self, 'canvas') and self.canvas:
                # Clear any existing image
                if hasattr(self, 'canvas_image'):
                    self.canvas.delete(self.canvas_image)
                    
                self.canvas_image = self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
                self.canvas.configure(scrollregion=self.canvas.bbox("all"))
                
        except Exception as e:
            print(f"Error loading image: {e}")
            return

    def create_widgets(self):
        """Create the modern UI layout"""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Toolbar
        self.create_toolbar()
        
        # Canvas frame
        canvas_frame = ctk.CTkFrame(self)
        canvas_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        canvas_frame.grid_columnconfigure(0, weight=1)
        canvas_frame.grid_rowconfigure(0, weight=1)

        # Canvas for image display
        self.canvas = tk.Canvas(
            canvas_frame,
            bg="#2b2b2b",
            highlightthickness=0
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        
        # Scrollbars
        v_scrollbar = ctk.CTkScrollbar(canvas_frame, orientation="vertical", command=self.canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=v_scrollbar.set)

        h_scrollbar = ctk.CTkScrollbar(canvas_frame, orientation="horizontal", command=self.canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(xscrollcommand=h_scrollbar.set)

        # Display image if loaded
        if hasattr(self, 'photo') and self.photo:
            try:
                self.canvas_image = self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
                self.canvas.configure(scrollregion=self.canvas.bbox("all"))
                print(f"Image displayed successfully: {self.display_image.size}")
            except Exception as e:
                print(f"Error displaying image in canvas: {e}")
        elif self.filename:
            # Load image if not already loaded
            print(f"Loading image: {self.filename}")
            self.load_image()
            if hasattr(self, 'photo') and self.photo:
                try:
                    self.canvas_image = self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
                    self.canvas.configure(scrollregion=self.canvas.bbox("all"))
                    print(f"Image loaded and displayed: {self.display_image.size}")
                except Exception as e:
                    print(f"Error displaying newly loaded image: {e}")
            else:
                print("Failed to load image")

        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

        # Status bar
        self.create_status_bar()
        
        # Draw existing control points if any (after canvas is fully set up)
        if self.ctrlPoints:
            print(f"Found {len(self.ctrlPoints)} existing control points, drawing them...")
            self.canvas.after(100, self.redraw_points)  # Small delay to ensure canvas is ready

    def create_toolbar(self):
        """Create the modern toolbar with buttons"""
        toolbar = ctk.CTkFrame(self, height=60)
        toolbar.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew")
        toolbar.grid_columnconfigure(4, weight=1)

        # Title
        title_label = ctk.CTkLabel(toolbar, text="🎯 Control Point Editor", font=ctk.CTkFont(size=16, weight="bold"))
        title_label.grid(row=0, column=0, padx=10, pady=10)

        # Insert Point Button
        self.insert_btn = ctk.CTkButton(
            toolbar,
            text="📍 Insert Point (I)",
            command=self.activate_insert_mode,
            width=130,
            height=32,
            fg_color="#1f538d",
            hover_color="#2d5aa0"
        )
        self.insert_btn.grid(row=0, column=1, padx=5, pady=10)

        # Move Point Button
        self.move_btn = ctk.CTkButton(
            toolbar,
            text="🔄 Move Point (M)",
            command=self.activate_move_mode,
            width=130,
            height=32,
            fg_color="#1f538d",
            hover_color="#2d5aa0"
        )
        self.move_btn.grid(row=0, column=2, padx=5, pady=10)

        # Delete Point Button
        self.delete_btn = ctk.CTkButton(
            toolbar,
            text="🗑️ Delete Point (D)",
            command=self.activate_delete_mode,
            width=130,
            height=32,
            fg_color="#d32f2f",
            hover_color="#b71c1c"
        )
        self.delete_btn.grid(row=0, column=3, padx=5, pady=10)

        # Finish Button
        self.finish_btn = ctk.CTkButton(
            toolbar,
            text="✅ Finish",
            command=self.finish_editing,
            width=100,
            height=32,
            fg_color="#2e7d32",
            hover_color="#1b5e20"
        )
        self.finish_btn.grid(row=0, column=5, padx=10, pady=10)

    def create_status_bar(self):
        """Create status bar"""
        self.status_frame = ctk.CTkFrame(self, height=30)
        self.status_frame.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="ew")
        
        self.status_label = ctk.CTkLabel(self.status_frame, text="Ready - Press I to insert points", font=ctk.CTkFont(size=12))
        self.status_label.pack(pady=5)

    def activate_insert_mode(self):
        """Activate insert point mode"""
        self.INSERT_FLAG = True
        self.MOVE_FLAG = False
        self.DELETE_FLAG = False
        self.update_button_states()
        if hasattr(self, 'status_label'):
            self.status_label.configure(text="Insert Mode - Click to add points")

    def activate_move_mode(self):
        """Activate move point mode"""
        self.INSERT_FLAG = False
        self.MOVE_FLAG = True
        self.DELETE_FLAG = False
        self.update_button_states()
        if hasattr(self, 'status_label'):
            self.status_label.configure(text="Move Mode - Click and drag points")

    def activate_delete_mode(self):
        """Activate delete point mode"""
        self.INSERT_FLAG = False
        self.MOVE_FLAG = False
        self.DELETE_FLAG = True
        self.update_button_states()
        if hasattr(self, 'status_label'):
            self.status_label.configure(text="Delete Mode - Click on points to remove them")

    def update_button_states(self):
        """Update button appearance based on active mode"""
        # Only update if GUI components exist
        if not hasattr(self, 'insert_btn'):
            return
            
        # Reset all buttons
        self.insert_btn.configure(fg_color="#1f538d")
        self.move_btn.configure(fg_color="#1f538d")
        self.delete_btn.configure(fg_color="#d32f2f")

        # Highlight active button
        if self.INSERT_FLAG:
            self.insert_btn.configure(fg_color="#2d5aa0")
        elif self.MOVE_FLAG:
            self.move_btn.configure(fg_color="#2d5aa0")
        elif self.DELETE_FLAG:
            self.delete_btn.configure(fg_color="#b71c1c")

    def on_key_press(self, event):
        """Handle keyboard shortcuts"""
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
        orig_x = int(x / self.scale_factor)
        orig_y = int(y / self.scale_factor)
        
        if self.INSERT_FLAG:
            # Add new point
            self.ctrlPoints.append([orig_x, orig_y])
            self.redraw_points()
            
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

    def on_canvas_drag(self, event):
        """Handle canvas drag events"""
        if self.ACTIVE_MOV_FLAG and self.ind is not None:
            # Convert canvas coordinates to image coordinates
            x = self.canvas.canvasx(event.x)
            y = self.canvas.canvasy(event.y)
            
            # Scale coordinates back to original image size
            orig_x = int(x / self.scale_factor)
            orig_y = int(y / self.scale_factor)
            
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
            
        xy = np.asarray(self.ctrlPoints)
        xt, yt = xy[:, 0], xy[:, 1]
        d = np.hypot(xt - x, yt - y)
        (indseq,) = np.nonzero(d == d.min())
        ind = indseq[0]
        
        # Scale epsilon for current zoom level
        scaled_epsilon = self.epsilon / self.scale_factor
        if d[ind] >= scaled_epsilon:
            ind = None
        return ind

    def redraw_points(self):
        """Redraw all points and lines on the canvas"""
        # Check if canvas exists
        if not hasattr(self, 'canvas') or not self.canvas:
            return
            
        # Clear previous drawings
        self.canvas.delete("points")
        self.canvas.delete("lines")
        
        if not self.ctrlPoints:
            return
            
        # Draw lines between points
        if len(self.ctrlPoints) > 1:
            for i in range(len(self.ctrlPoints) - 1):
                x1, y1 = self.ctrlPoints[i]
                x2, y2 = self.ctrlPoints[i + 1]
                
                # Scale coordinates for display
                display_x1 = x1 * self.scale_factor
                display_y1 = y1 * self.scale_factor
                display_x2 = x2 * self.scale_factor
                display_y2 = y2 * self.scale_factor
                
                self.canvas.create_line(
                    display_x1, display_y1, display_x2, display_y2,
                    fill=self.ORANGE, width=2, tags="lines"
                )
        
        # Draw points
        for i, (x, y) in enumerate(self.ctrlPoints):
            # Scale coordinates for display
            display_x = x * self.scale_factor
            display_y = y * self.scale_factor
            
            # Draw point
            self.canvas.create_oval(
                display_x - self.thickness//2, display_y - self.thickness//2,
                display_x + self.thickness//2, display_y + self.thickness//2,
                fill=self.RED, outline=self.WHITE, width=2, tags="points"
            )
            
            # Draw point number
            self.canvas.create_text(
                display_x, display_y - self.thickness - 5,
                text=str(i + 1), fill=self.WHITE, font=("Arial", 8, "bold"), tags="points"
            )
        
        # Update status
        if hasattr(self, 'status_label'):
            self.status_label.configure(text=f"Points: {len(self.ctrlPoints)}")

    def finish_editing(self):
        """Finish editing and return results"""
        if self.ctrlPoints:
            self.result = np.asarray([self.ctrlPoints])
        else:
            self.result = np.asarray([])
        self.destroy()

    def run(self, windows_title=None):
        """Run the control point editor"""
        # Check if this is a GUI-enabled instance
        if hasattr(self, 'tk') and self.tk:
            # Modern GUI mode
            if windows_title:
                self.title(windows_title)
            
            # Start with insert mode
            self.activate_insert_mode()
            
            # Draw existing control points if any
            if self.ctrlPoints:
                self.after(50, self.redraw_points)  # Small delay to ensure everything is ready
            
            # Show the window and wait for it to close
            self.wait_window()
            
            # Return the result
            return self.result if self.result is not None else np.asarray([])
        else:
            # Legacy mode for inherited classes (like RoiSelection)
            # Initialize GUI on demand
            try:
                # Find the main tkinter root window
                import tkinter as tk
                
                # Try to get existing root window or create new one
                try:
                    root = tk._default_root
                    if root is None:
                        root = tk.Tk()
                        root.withdraw()
                except:
                    root = tk.Tk()
                    root.withdraw()
                
                # Initialize GUI components
                self._initialize_gui(root)
                
                if windows_title:
                    self.title(windows_title)
                
                # Start with insert mode
                self.activate_insert_mode()
                
                # Draw existing control points if any
                if self.ctrlPoints:
                    self.after(50, self.redraw_points)  # Small delay to ensure everything is ready
                
                # Show the window and wait for it to close
                self.wait_window()
                
                # Return the result - handle different return types for inherited classes
                if hasattr(self, 'return_worker'):
                    return self.return_worker()
                else:
                    return self.result if self.result is not None else np.asarray([])
                    
            except Exception as e:
                # If GUI initialization fails, return empty result
                print(f"GUI initialization failed: {e}")
                if hasattr(self, 'return_worker'):
                    return self.return_worker()
                else:
                    return np.asarray([])
