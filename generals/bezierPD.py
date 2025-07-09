import cv2
import numpy as np
import bezier
import generals.math_aux as math_aux
import math
import generals.windows_App as windows_App
import customtkinter as ctk
import tkinter as tk


class BezierPD(windows_App.App):
    """
    BezierPD class
    This class is used to create a Bezier curve and calculate the perpendicular distance of the curve to the ROI.
    Args:
        filename: string, image path
        roi: numpy array, region of interest
        ctrlPoints: list, control points
    Output:
        pd: numpy array, perpendicular distance
        intersection: list, intersection points
    """

    thickness = 5
    epsilon = 10

    def __init__(self, filename=None, roi=None, ctrlPoints=[], parent=None):
        # Initialize the parent class first
        super().__init__(filename=filename, parent=parent)
        
        # Override with specific values for BezierPD
        self.img = cv2.imread(filename)
        self.ctrlPoints = ctrlPoints if ctrlPoints else []
        self.pd = None
        self.SHOW_LINES = False  # key: 'r'
        self.roi = roi
        self.intersection = []
        self.thickness = 8
        self.parent = parent

    def line_control(self, k=None, draw_on_canvas=True):
        if len(self.ctrlPoints) < 2:
            print("Not enough control points to create axis")
            return
            
        try:
            nodes = np.asarray(self.ctrlPoints).T
            curve = bezier.Curve(nodes, (len(self.ctrlPoints) - 1))
            pd = curve.evaluate_multi(np.linspace(0.0, 1.0, 5000)).T
            pd = np.unique(pd.astype(int), axis=0)

            if self.ctrlPoints[0][0] < self.img.shape[0]:
                self.pd = pd
            else:
                self.pd = pd[np.argsort(pd[:, 0])][::-1]
            
            print(f"Axis curve calculated successfully with {len(self.pd)} points")
            
            # Only draw on canvas if requested and canvas is available and valid
            if draw_on_canvas and hasattr(self, 'canvas') and self.canvas:
                try:
                    # Check if canvas is still valid (not destroyed)
                    self.canvas.winfo_exists()
                    self.draw_axis_curve()
                except tk.TclError:
                    print("Canvas no longer available for drawing (window closed)")
                except Exception as e:
                    print(f"Error drawing on canvas: {e}")
            
            # Calculate intersections for analysis (without drawing)
            self.calculate_intersections(draw_on_canvas=draw_on_canvas)
            
        except Exception as e:
            print(f"Error creating axis curve: {e}")
            # Don't reset pd here - keep the calculated result
            import traceback
            traceback.print_exc()
            
    def draw_axis_curve(self):
        """Draw the axis curve on the canvas"""
        if not hasattr(self, 'canvas') or not self.canvas or self.pd is None:
            return
            
        # Clear previous axis
        self.canvas.delete("axis")
        
        # Draw axis curve
        if len(self.pd) > 1:
            for i in range(len(self.pd) - 1):
                x1, y1 = self.pd[i]
                x2, y2 = self.pd[i + 1]
                
                # Scale coordinates for display
                display_x1 = x1 * self.scale_factor
                display_y1 = y1 * self.scale_factor
                display_x2 = x2 * self.scale_factor
                display_y2 = y2 * self.scale_factor
                
                self.canvas.create_line(
                    display_x1, display_y1, display_x2, display_y2,
                    fill=self.GREEN, width=3, tags="axis"
                )
    
    def calculate_intersections(self, draw_on_canvas=True):
        """Calculate perpendicular intersections with ROI"""
        if self.pd is None or len(self.pd) == 0:
            return
            
        self.intersection = []
        
        for k in range(0, len(self.pd), 20):
            y = []
            a = []
            try:
                if k < 20:
                    for i in range(0, k + 20):
                        y.append([self.pd[i][1]])
                        a.append([1, self.pd[i][0]])
                elif k >= 20 and abs(len(self.pd) - k) >= 20:
                    for i in range(k - 20, k + 20):
                        y.append([self.pd[i][1]])
                        a.append([1, self.pd[i][0]])
                else:
                    for i in range(k - 20, len(self.pd)):
                        y.append([self.pd[i][1]])
                        a.append([1, self.pd[i][0]])
            except IndexError:
                print("Index Error", k)
                continue
                
            try:
                slope = np.linalg.lstsq(y, a, rcond=None)[1][1]
                b = np.linalg.lstsq(y, a, rcond=None)[0][0][0]
                pt_y = b + slope * (self.pd[k][0])
                intersection = math_aux.f(self.pd[k][0], int(pt_y), -(1 / slope)) & self.roi
                intersection_list = list(intersection)
                self.intersection.append(intersection_list)
                
                # Only draw intersection lines if requested and canvas is available and valid
                if draw_on_canvas and hasattr(self, 'canvas') and self.canvas:
                    try:
                        self.canvas.winfo_exists()
                        self.draw_intersection_lines(intersection_list)
                    except tk.TclError:
                        # Canvas no longer available, skip drawing
                        pass
                    except Exception as e:
                        print(f"Error drawing intersection lines: {e}")
                    
            except Exception as e:
                print(f"Error calculating intersection at {k}: {e}")
                continue
        
        print(f"Calculated {len(self.intersection)} intersection groups")
    
    def draw_intersection_lines(self, intersection_list):
        """Draw intersection lines on the canvas"""
        if not intersection_list or len(intersection_list) < 2:
            return
            
        try:
            for i in range(len(intersection_list) - 1):
                pt1 = (intersection_list[i][0], intersection_list[i][1])
                pt2 = (intersection_list[i + 1][0], intersection_list[i + 1][1])
                
                # Scale coordinates for display
                display_x1 = pt1[0] * self.scale_factor
                display_y1 = pt1[1] * self.scale_factor
                display_x2 = pt2[0] * self.scale_factor
                display_y2 = pt2[1] * self.scale_factor
                
                self.canvas.create_line(
                    display_x1, display_y1, display_x2, display_y2,
                    fill=self.BLUE, width=2, tags="axis"
                )
        except IndexError:
            print("Index Error in drawing intersection lines")

    def redraw_points(self):
        """Override redraw_points to also show axis curve"""
        # Call parent method to draw control points
        super().redraw_points()
        
        # Clear existing axis
        if hasattr(self, 'canvas') and self.canvas:
            try:
                self.canvas.winfo_exists()
                self.canvas.delete("axis")
            except tk.TclError:
                # Canvas no longer available
                return
        
        # Recalculate and draw the axis curve when points change (only during interactive editing)
        if len(self.ctrlPoints) >= 2:
            try:
                self.line_control(draw_on_canvas=True)  # Always draw during interactive editing
                print(f"Interactive axis calculated with {len(self.ctrlPoints)} control points")
            except Exception as e:
                print(f"Error calculating interactive axis: {e}")
        else:
            print(f"Need at least 2 points for axis (currently have {len(self.ctrlPoints)})")
    
    def create_toolbar(self):
        """Override toolbar to add axis-specific controls"""
        # Call parent toolbar creation
        super().create_toolbar()
        
        # Add axis calculation button
        if hasattr(self, 'finish_btn'):
            axis_btn = ctk.CTkButton(
                self.finish_btn.master,
                text="🔄 Recalculate Axis",
                command=self.recalculate_axis,
                width=130,
                height=32,
                fg_color="#2e7d32",
                hover_color="#1b5e20"
            )
            axis_btn.grid(row=0, column=4, padx=5, pady=10)
    
    def recalculate_axis(self):
        """Manually recalculate the axis"""
        if len(self.ctrlPoints) >= 2:
            self.line_control()
            if hasattr(self, 'status_label'):
                self.status_label.configure(text=f"Axis recalculated with {len(self.ctrlPoints)} points")
        else:
            if hasattr(self, 'status_label'):
                self.status_label.configure(text="Need at least 2 points to create axis")
    
    def run(self, windows_title=None):
        """Override run method to show GUI and handle axis creation"""
        # Don't calculate axis immediately - let the user interact first
        
        # Call parent run method to show the GUI
        result = super().run(windows_title)
        
        # After GUI closes, calculate final axis if we have enough points
        # Don't try to draw on canvas since window is closed
        if len(self.ctrlPoints) >= 2:
            print(f"Final axis calculation with {len(self.ctrlPoints)} control points")
            self.line_control(draw_on_canvas=False)
        else:
            print(f"Not enough control points for final axis calculation: {len(self.ctrlPoints)}")
        
        # Return the curve and intersections
        return self.return_worker()
    
    def return_worker(self):
        # If no curve was generated, return empty results
        if self.pd is None:
            print("No axis curve generated (pd is None)")
            return np.array([]), []
        
        # Check if pd is empty array
        if hasattr(self.pd, '__len__') and len(self.pd) == 0:
            print("No axis curve generated (pd is empty)")
            return np.array([]), []
            
        print(f"Returning axis curve with {len(self.pd)} points and {len(self.intersection) if self.intersection else 0} intersections")
        return (self.pd, self.intersection if self.intersection else [])
