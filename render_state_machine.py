import numpy as np
from typing import Dict, Optional, Any
from dataclasses import dataclass
import threading
import queue
from enum import Enum, auto

class RenderState(Enum):
    IDLE = auto()
    RENDERING = auto()
    PROCESSING = auto()
    VISUALIZING = auto()
    ERROR = auto()

@dataclass
class RenderOutput:
    """Container for render outputs."""
    points: np.ndarray  # (N, 3) point cloud coordinates
    colors: np.ndarray  # (N, 3) RGB colors
    depths: np.ndarray  # (N,) depth values
    normals: Optional[np.ndarray] = None  # (N, 3) normal vectors if available
    metadata: Optional[Dict[str, Any]] = None

class RenderStateMachine:
    def __init__(self):
        """Initialize the render state machine."""
        self.state = RenderState.IDLE
        self.current_output: Optional[RenderOutput] = None
        self.output_queue = queue.Queue()
        self.lock = threading.Lock()
        
    def start_render(self) -> None:
        """Start a new render."""
        with self.lock:
            if self.state == RenderState.IDLE:
                self.state = RenderState.RENDERING
                
    def process_render_output(self, render_data: Dict[str, np.ndarray]) -> None:
        """Process render output data.
        
        Args:
            render_data: Dictionary containing render outputs from eval
        """
        with self.lock:
            if self.state == RenderState.RENDERING:
                try:
                    self.state = RenderState.PROCESSING
                    
                    # Create RenderOutput object
                    output = RenderOutput(
                        points=render_data["points"],
                        colors=render_data["colors"],
                        depths=render_data["depths"],
                        normals=render_data.get("normals", None),
                        metadata={
                            "timestamp": np.datetime64('now'),
                            "num_points": len(render_data["points"])
                        }
                    )
                    
                    # Store output and put in queue
                    self.current_output = output
                    self.output_queue.put(output)
                    
                    self.state = RenderState.VISUALIZING
                    
                except Exception as e:
                    print(f"Error processing render output: {e}")
                    self.state = RenderState.ERROR
                    
    def get_latest_output(self) -> Optional[RenderOutput]:
        """Get the latest render output."""
        with self.lock:
            return self.current_output
            
    def get_next_output(self, timeout: float = 0.1) -> Optional[RenderOutput]:
        """Get next output from the queue.
        
        Args:
            timeout: Time to wait for next output
            
        Returns:
            RenderOutput object or None if queue is empty
        """
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def reset(self) -> None:
        """Reset the state machine."""
        with self.lock:
            self.state = RenderState.IDLE
            self.current_output = None
            # Clear queue
            while not self.output_queue.empty():
                try:
                    self.output_queue.get_nowait()
                except queue.Empty:
                    break
                    
    @property
    def is_rendering(self) -> bool:
        """Check if currently rendering."""
        return self.state == RenderState.RENDERING
        
    @property
    def is_processing(self) -> bool:
        """Check if currently processing."""
        return self.state == RenderState.PROCESSING
        
    @property
    def is_visualizing(self) -> bool:
        """Check if currently visualizing."""
        return self.state == RenderState.VISUALIZING
        
    @property
    def has_error(self) -> bool:
        """Check if in error state."""
        return self.state == RenderState.ERROR 