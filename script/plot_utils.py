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
            self.line_est.set_data(self.est_x, self.est_y)
            self.line_corr.set_data(self.corr_x, self.corr_y)

            # Auto-scale if needed (optional)
            if len(self.est_x) > 0:
                self.ax.relim()
                self.ax.autoscale_view(scalex=False, scaley=False)
                
        return self.line_est, self.line_corr

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