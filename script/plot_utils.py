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
        self.line_corr, = self.ax.plot([], [], 'dodgerblue', label='Corrected Trajectory', lw=2, visible=True)
        
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
        self.current_elements = []
        self.fov_patch = None
        self.intersection_points = None
        self.current_frame_count = 0
        self._artists = []  # Track all managed artists
        self._init_artists()

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
        active_artists = []
    
        # Update permanent artists
        if len(self.est_x) > 0:
            self.line_est.set_data(self.est_x, self.est_y)
            active_artists.append(self.line_est)
        
        if len(self.corr_x) > 0:
            self.line_corr.set_data(self.corr_x, self.corr_y)
            active_artists.append(self.line_corr)
        
        # Handle temporary elements
        if hasattr(self, 'fov_patch') and self.fov_patch:
            active_artists.append(self.fov_patch)
        if hasattr(self, 'intersection_points') and self.intersection_points:
            active_artists.append(self.intersection_points)
        
        # Verify all artists before return
        valid_artists = []
        for artist in active_artists:
            if hasattr(artist, 'axes') and artist.axes is not None:
                valid_artists.append(artist)
            else:
                print(f"Warning: Discarding invalid artist {artist}")
        
        return valid_artists

    def add_temporary_elements(self, fov_box, intersections):
        """Clear and redraw temporary elements"""
        # Clear old elements
        for artist in [a for a in [self.fov_patch, self.intersection_points] if a]:
            try:
                artist.remove()
            except:
                pass
        
        # Create new FOV patch
        if fov_box and hasattr(fov_box, 'exterior'):
            self.fov_patch = plt.Polygon(
                list(fov_box.exterior.coords),
                closed=True, fill=False, color='red', alpha=0.5
            )
            self.ax.add_patch(self.fov_patch)
        
        # Create intersection points
        if intersections:
            ix, iy = zip(*[(p[0], p[1]) for p in intersections])
            self.intersection_points, = self.ax.plot(
                ix, iy, 'ro', markersize=8, alpha=0.7
            )
        
        # Verify new artists
        for artist in [self.fov_patch, self.intersection_points]:
            if artist and not hasattr(artist, 'axes'):
                print(f"Warning: Failed to create proper artist for {artist}")

    def start_animation(self):
        """Start the real-time animation"""
        self.is_running = True
        self.animation = FuncAnimation(
            self.fig,
            self.update_plot,
            frames=None,
            interval=100,
            blit=True,
            cache_frame_data=False,
            init_func=self._init_draw  # Use dedicated init
        )
        plt.show(block=False)
    
    def _init_draw(self):
        """Initial draw for blitting"""
        for artist in self._artists:
            artist.set_visible(True)
        return self._artists

    def add_est_point(self, x, y):
        """Add new point with downsampling"""
        self.counter += 1
        if self.counter % self.downsample_factor == 0:
            self.est_x.append(x)
            self.est_y.append(y)
            return True  # Point was added
        return False  # Point was skipped
        

    def add_corr_point(self, x, y):
        """Add new corrected point with downsampling"""
        self.counter += 1
        if self.counter % self.downsample_factor == 0:
            self.corr_x.append(x)
            self.corr_y.append(y)
            return True  # Point was added
        return False  # Point was skipped

    def _init_artists(self):
        """Initialize all artists and ensure they have axes"""
        # Clear existing artists
        for artist in self._artists:
            try:
                artist.remove()
            except:
                pass
        
        # Create fresh artists
        self.line_est, = self.ax.plot([], [], 'forestgreen', label='Estimated', lw=2)
        self.line_corr, = self.ax.plot([], [], 'dodgerblue', label='Corrected', lw=2)
        self.fov_patch = None
        self.intersection_points = None
        
        # Register permanent artists
        self._artists = [self.line_est, self.line_corr]
        
        # Verify artists
        for artist in self._artists:
            assert hasattr(artist, 'axes'), f"Artist {artist} has no axes"

    def close(self):
        """Clean up"""
        self.is_running = False
        if self.animation:
            self.animation.event_source.stop()
        plt.close(self.fig)