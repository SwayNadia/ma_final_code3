import viser
import numpy as np
import threading
import queue
import time
from typing import Optional, Dict, Any
from render_state_machine import RenderStateMachine, RenderOutput

class Viewer:
    def __init__(self, host: str = "localhost", port: int = 8080):
        """Initialize the viewer.
        
        Args:
            host: Host address for the viser server
            port: Port for the viser server
        """
        self.server = viser.ViserServer(host=host, port=port)
        self.state_machine = RenderStateMachine()
        self.is_running = True
        
        # Start the visualization thread
        self.vis_thread = threading.Thread(target=self._visualization_loop)
        self.vis_thread.daemon = True
        self.vis_thread.start()
        
        # Initialize scene elements
        self._init_scene()
        
    def _init_scene(self):
        """Initialize the scene with basic elements."""
        # Add world coordinate frame
        self.server.scene.add_frame(
            name="world_frame",
            show_axes=True,
            axes_length=1.0,
            axes_radius=0.02
        )
        
        # Configure environment lighting
        self.server.scene.configure_environment_map(
            hdri="warehouse",
            background=True,
            background_intensity=1.0
        )
        
        # Add transform controls
        self.server.scene.add_transform_controls(
            name="transform_controls",
            scale=1.0,
            line_width=2.5
        )
        
        # Add GUI elements
        with self.server.gui.add_folder("Visualization Controls"):
            self.point_size = self.server.gui.add_slider(
                "Point Size", 
                min=0.001, 
                max=0.1, 
                step=0.001, 
                initial_value=0.01
            )
            self.show_normals = self.server.gui.add_checkbox(
                "Show Normals",
                initial_value=True
            )
            
    def update_point_cloud(self, point_cloud_data: Dict[str, np.ndarray]):
        """Update the point cloud visualization.
        
        Args:
            point_cloud_data: Dictionary containing:
                - points: 3D point cloud coordinates (N, 3)
                - colors: RGB colors for each point (N, 3)
                - depths: Depth values for each point (N,)
                - normals: Normal vectors for each point (N, 3) if available
        """
        # Start new render in state machine
        self.state_machine.start_render()
        # Process render output
        self.state_machine.process_render_output(point_cloud_data)
        
    def _visualization_loop(self):
        """Main visualization loop that processes data from the queue."""
        while self.is_running:
            try:
                # Get next output from state machine
                output = self.state_machine.get_next_output()
                if output is None:
                    continue
                    
                # Remove existing point cloud if it exists
                try:
                    self.server.scene.remove_by_name("scene_points")
                except:
                    pass
                
                # Add new point cloud
                self.server.scene.add_point_cloud(
                    name="scene_points",
                    points=output.points,
                    colors=output.colors,
                    point_size=self.point_size.value,
                    point_shape="circle"
                )
                
                # Add normals if available and enabled
                if output.normals is not None and self.show_normals.value:
                    try:
                        self.server.scene.remove_by_name("normals")
                    except:
                        pass
                        
                    # Calculate normal endpoints
                    normal_endpoints = output.points + output.normals * 0.1
                    normal_lines = np.stack([
                        output.points,
                        normal_endpoints
                    ], axis=1)
                    
                    self.server.scene.add_line_segments(
                        name="normals",
                        points=normal_lines,
                        colors=(255, 0, 0),
                        line_width=1.0
                    )
                
                # Update status in GUI
                if hasattr(output, 'metadata'):
                    self.server.gui.add_text(
                        f"Points: {output.metadata['num_points']}",
                        label="Status"
                    )
                
            except Exception as e:
                print(f"Error in visualization loop: {e}")
                time.sleep(0.1)
                
    def close(self):
        """Close the viewer and clean up resources."""
        self.is_running = False
        self.state_machine.reset()
        if self.vis_thread.is_alive():
            self.vis_thread.join()
        self.server.stop()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 