import numpy as np
import torch
import viser
import viser.transforms as vtf
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from pathlib import Path

@dataclass
class CameraState:
    """Camera state for visualization"""
    fov: float
    aspect: float
    c2w: torch.Tensor
    camera_type: str = "perspective"

class Viewer:
    """Viewer class for 3D visualization using viser"""
    
    def __init__(self, port: int = 8080):
        self.viser_server = viser.ViserServer(host="localhost", port=port)
        self.ready = False
        self.camera_handles: Dict[int, viser.CameraFrustumHandle] = {}
        self.render_statemachines: Dict[int, RenderStateMachine] = {}
        
        # Initialize GUI elements
        self._setup_gui()
        
    def _setup_gui(self):
        """Setup the viewer GUI elements"""
        # Add control panel
        tabs = self.viser_server.gui.add_tab_group()
        control_tab = tabs.add_tab("Control", viser.Icon.SETTINGS)
        
        with control_tab:
            # Add visualization controls
            self.point_size = self.viser_server.gui.add_slider(
                "Point Size", 0.1, 10.0, 0.1, 1.0
            )
            self.point_opacity = self.viser_server.gui.add_slider(
                "Point Opacity", 0.0, 1.0, 0.1, 0.8
            )
            self.show_depth = self.viser_server.gui.add_checkbox(
                "Show Depth", True
            )
            
    def visualize_3d_data(self, three_d_data: Dict[str, torch.Tensor]):
        """Visualize 3D data in the viewer
        
        Args:
            three_d_data: Dictionary containing 3D data including:
                - ray_origins: Ray origins
                - ray_directions: Ray directions
                - sample_points: Sampled points along rays
                - features: Feature vectors
                - colors: Color values
                - densities: Density values
                - weights: Weights
                - z_values: Depth values
        """
        if not self.ready:
            self.ready = True
            
        # Clear existing points
        self.viser_server.scene.clear()
        
        # Get point size and opacity from GUI
        point_size = self.point_size.value
        point_opacity = self.point_opacity.value
        
        # Convert tensors to numpy arrays
        sample_points = three_d_data["sample_points"].cpu().numpy()
        colors = three_d_data["colors"].cpu().numpy()
        weights = three_d_data["weights"].cpu().numpy()
        
        # Create point cloud
        for i in range(len(sample_points)):
            point = sample_points[i]
            color = colors[i]
            weight = weights[i]
            
            # Create point with color and size based on weight
            self.viser_server.scene.add_point(
                position=point,
                color=color,
                radius=point_size * weight,
                opacity=point_opacity
            )
            
        # Add depth visualization if enabled
        if self.show_depth.value and "z_values" in three_d_data:
            z_values = three_d_data["z_values"].cpu().numpy()
            ray_origins = three_d_data["ray_origins"].cpu().numpy()
            ray_directions = three_d_data["ray_directions"].cpu().numpy()
            
            # Create depth visualization
            for i in range(len(ray_origins)):
                origin = ray_origins[i]
                direction = ray_directions[i]
                z = z_values[i]
                
                # Calculate end point
                end_point = origin + direction * z
                
                # Add line for depth visualization
                self.viser_server.scene.add_line(
                    start=origin,
                    end=end_point,
                    color=(0, 1, 0),  # Green color for depth lines
                    opacity=0.5
                )
                
    def get_camera_state(self, client: viser.ClientHandle) -> CameraState:
        """Get current camera state from client"""
        R = vtf.SO3(wxyz=client.camera.wxyz)
        R = R @ vtf.SO3.from_x_radians(np.pi)
        R = torch.tensor(R.as_matrix())
        pos = torch.tensor(client.camera.position, dtype=torch.float64)
        c2w = torch.concatenate([R, pos[:, None]], dim=1)
        
        return CameraState(
            fov=client.camera.fov,
            aspect=client.camera.aspect,
            c2w=c2w
        )
        
    def handle_new_client(self, client: viser.ClientHandle):
        """Handle new client connection"""
        self.render_statemachines[client.client_id] = RenderStateMachine(self, client)
        
        @client.camera.on_update
        def _(_: viser.CameraHandle) -> None:
            if not self.ready:
                return
            camera_state = self.get_camera_state(client)
            self.render_statemachines[client.client_id].action(RenderAction("move", camera_state))
            
    def handle_disconnect(self, client: viser.ClientHandle):
        """Handle client disconnect"""
        if client.client_id in self.render_statemachines:
            self.render_statemachines[client.client_id].running = False
            del self.render_statemachines[client.client_id]

class RenderStateMachine:
    """Render state machine for handling viewer updates"""
    
    def __init__(self, viewer: Viewer, client: viser.ClientHandle):
        self.viewer = viewer
        self.client = client
        self.running = True
        
    def action(self, action: RenderAction):
        """Handle render action"""
        # For now, just update the visualization
        if action.action == "move":
            self.viewer.visualize_3d_data(self.viewer.current_3d_data)
            
@dataclass
class RenderAction:
    """Render action for state machine"""
    action: str
    camera_state: CameraState 