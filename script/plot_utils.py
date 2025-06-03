import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt backend for better performance
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class RealTimePlotter:
    def __init__(self, edges, roads, buildings, crossings, railway, green, minx, miny, maxx, maxy):
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.setup_background(edges, roads, buildings, crossings, railway, green)
        self.setup_axes(minx, miny, maxx, maxy)
        
        # Initialize trajectory lines
        self.line_est, = self.ax.plot([], [], 'forestgreen', label='Estimated Trajectory', lw=2, visible=True)
        self.line_corr, = self.ax.plot([], [], 'dodgerblue', label='Corrected Trajectory', lw=2)
        
        # Data storage
        self.est_x = []
        self.est_y = []
        self.corr_x = []
        self.corr_y = []
        
        # Animation control
        self.animation = None
        self.is_running = False

        # Downsampler Factor
        self.downsample_factor = 5  # Plot every 5th point
        self.counter = 0

        # Add temporary plot elements
        self.fov_patch = None
        self.intersection_points = None
        self.current_frame_count = 0

    def setup_background(self, edges, roads, buildings, crossings, railway, green):
        """Plot static map elements"""
        green.plot(ax=self.ax, color="palegreen", alpha=0.6)
        roads.plot(ax=self.ax, color="slategray", alpha=0.7)
        railway.plot(ax=self.ax, color="black", alpha=0.7)
        crossings.plot(ax=self.ax, color="lawngreen", alpha=0.8)
        buildings.plot(ax=self.ax, color="sienna", alpha=0.8)
        self.ax.legend(loc='upper left')

    def setup_axes(self, minx, miny, maxx, maxy):
        """Configure plot axes"""
        self.ax.set_xlim(minx, maxx)
        self.ax.set_ylim(miny, maxy)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X Coordinate (m)')
        self.ax.set_ylabel('Y Coordinate (m)')
        self.ax.set_title('Real-Time Trajectory Visualization')

    def update_plot(self, frame):
        """Animation update function"""
        if self.is_running:
            self.current_frame_count += 1
            elements = [self.line_est, self.line_corr]
            
            # Only update temporary elements every 5 frames
            if self.current_frame_count % 5 == 0 and hasattr(self, 'last_fov_box'):
                temp_elements = self.update_temporary_elements(
                    self.last_fov_box,
                    self.last_intersecting_points
                )
                elements.extend(temp_elements)
            
            # Auto-scale if needed
            if len(self.est_x) > 0:
                self.ax.relim()
                self.ax.autoscale_view(scalex=False, scaley=False)
                
        return elements

    def update_temporary_elements(self, fov_box=None, intersecting_points=None):
        """Update temporary visualization elements"""
        # Clear previous elements
        if self.fov_patch:
            self.fov_patch.remove()
        if self.intersection_points:
            self.intersection_points.remove()
        
        # Plot new elements if provided
        if fov_box is not None:
            self.fov_patch = plt.Polygon(
                list(fov_box.exterior.coords),
                closed=True,
                fill=False,
                color='lightblue',
                linewidth=1,
                alpha=0.5
            )
            self.ax.add_patch(self.fov_patch)
        
        if intersecting_points is not None and len(intersecting_points) > 0:
            x, y = zip(*[(p.x, p.y) for p in intersecting_points])
            self.intersection_points, = self.ax.plot(
                x, y,
                'ro',  # Red circles
                markersize=8,
                alpha=0.7,
                label='Intersections'
            )
        
        return self.line_est, self.line_corr, self.fov_patch, self.intersection_points

    def start_animation(self):
        """Start the real-time animation"""
        self.is_running = True
        self.animation = FuncAnimation(
            self.fig,
            self.update_plot,
            frames=None,
            interval=100,  # Update every 100ms
            blit=True,
            cache_frame_data=False
        )
        plt.show(block=False)

    def add_est_point(self, x, y):
        """Add new point with downsampling"""
        self.counter += 1
        if self.counter % self.downsample_factor == 0:
            self.est_x.append(x)
            self.est_y.append(y)
            return True  # Point was added
        return False  # Point was skipped
        

    def add_corr_point(self, x, y):
        """Add new corrected trajectory point"""
        self.corr_x.append(x)
        self.corr_y.append(y)

    def close(self):
        """Clean up"""
        self.is_running = False
        if self.animation:
            self.animation.event_source.stop()
        plt.close(self.fig)