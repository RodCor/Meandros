import cv2
import numpy as np
import generals.windows_App as windows_App


class RoiSelection(windows_App.App):

    def __init__(self, filename=None, approx=None):
        # Initialize the parent class first
        super().__init__(filename=filename, parent=None)
        
        # Override with specific values for RoiSelection
        self.img = cv2.imread(filename)
        self.ctrlPoints = approx if approx is not None else []
        self.SHOW_LINES = None

    def return_worker(self):
        # If no control points, return empty result
        if not self.ctrlPoints:
            return set(), []
            
        # Create ROI from control points
        self.roi = np.array(self.ctrlPoints, dtype=np.int32)
        
        # Use img if img2 doesn't exist
        img_shape = self.img.shape[:2] if hasattr(self, 'img') and self.img is not None else (100, 100)
        
        # Create mask from ROI
        mask2 = np.transpose(
            np.nonzero(
                cv2.fillPoly(
                    np.zeros(img_shape, np.uint8), [self.roi], (255, 0, 0)
                )
            )
        )
        
        # Create result set
        result = set(map(tuple, mask2[:, [1, 0]]))
        return result, self.ctrlPoints
