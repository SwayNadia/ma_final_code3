import threading
import numpy as np
import viser
import viser.transforms as tf
from typing import Dict, Optional
import torch

class Viewer3D:
    """3D Viewer for visualizing the rendered scene using viser.
    
    This viewer takes the 3D data generated during rendering and creates an
    interactive 3D visualization that can be viewed from any angle.
    """
    
    def __init__(self, port: int = 8080):
        self.server = viser.ViserServer(port=port)
        self.ready = True
        self.scene_elements = {}
        self._setup_gui()
        
    def _setup_gui(self):
        """Setup the basic GUI elements"""
        # Add basic controls
        self.control_panel = self.server.gui.add_folder("Control Panel")
        with self.control_panel:
            self.point_size = self.server.gui.add_slider(
                "Point Size", 
                min=0.001, 
                max=0.1, 
                step=0.001, 
                initial_value=0.01
            )
            self.opacity = self.server.gui.add_slider(
                "Opacity",
                min=0.0,
                max=1.0,
                step=0.1,
                initial_value=1.0
            )
            self.show_rays = self.server.gui.add_checkbox(
                "Show Rays",
                initial_value=False
            )
            
            @self.point_size.on_update
            def _(_) -> None:
                self.update_visualization()
                
            @self.opacity.on_update
            def _(_) -> None:
                self.update_visualization()
                
            @self.show_rays.on_update
            def _(_) -> None:
                self.update_visualization()

    def visualize_3d_data(self, three_d_data: Dict[str, torch.Tensor]):
        """Visualize the 3D data in the viewer.
        
        Args:
            three_d_data: Dictionary containing the 3D data including:
                - sample_points: Tensor of 3D points
                - colors: Tensor of RGB colors
                - ray_origins: Tensor of ray origins
                - ray_directions: Tensor of ray directions
                - densities: Tensor of density values
        """
        # Clear existing scene elements
        for element in self.scene_elements.values():
            element.remove()
        self.scene_elements.clear()
        
        # Convert tensors to numpy arrays
        points = three_d_data["sample_points"].detach().cpu().numpy()
        colors = three_d_data["colors"].detach().cpu().numpy()
        
        if "densities" in three_d_data:
            densities = three_d_data["densities"].detach().cpu().numpy()
            # Normalize densities to use as alpha values
            alpha = (densities - densities.min()) / (densities.max() - densities.min())
            alpha = alpha * self.opacity.value
        else:
            alpha = np.ones(len(points)) * self.opacity.value
            
        # Add point cloud
        self.scene_elements["points"] = self.server.scene.add_point_cloud(
            "/point_cloud",
            points=points,
            colors=colors,
            point_size=self.point_size.value,
            alpha=alpha
        )
        
        if self.show_rays.value and "ray_origins" in three_d_data and "ray_directions" in three_d_data:
            ray_origins = three_d_data["ray_origins"].detach().cpu().numpy()
            ray_directions = three_d_data["ray_directions"].detach().cpu().numpy()
            
            # Sample a subset of rays to visualize (to avoid cluttering)
            ray_sample_idx = np.random.choice(len(ray_origins), size=min(1000, len(ray_origins)), replace=False)
            
            # Create line segments for rays
            ray_starts = ray_origins[ray_sample_idx]
            ray_ends = ray_origins[ray_sample_idx] + ray_directions[ray_sample_idx]
            
            self.scene_elements["rays"] = self.server.scene.add_line_segments(
                "/rays",
                vertices=np.stack([ray_starts, ray_ends], axis=1),
                colors=np.array([1.0, 1.0, 0.0]),  # Yellow color for rays
                line_width=1.0
            )
    
    def update_visualization(self):
        """Update the visualization based on current control settings"""
        if hasattr(self, "last_three_d_data"):
            self.visualize_3d_data(self.last_three_d_data)
            
    def close(self):
        """Close the viewer"""
        self.server.close() 