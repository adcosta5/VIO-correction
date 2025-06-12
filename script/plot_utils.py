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
        self.downsample_factor = 1  # Plot every 5th point
        self.counter = 0

        # Add temporary plot elements
        self.current_elements = []
        self.fov_patch = None
        self.intersection_points = None
        self.point_cloud_points = None
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
        if hasattr(self, 'point_cloud_points') and self.point_cloud_points:
            active_artists.append(self.point_cloud_points)
        
        # Verify all artists before return
        valid_artists = []
        for artist in active_artists:
            if hasattr(artist, 'axes') and artist.axes is not None:
                valid_artists.append(artist)
            else:
                print(f"Warning: Discarding invalid artist {artist}")
        
        return valid_artists

    def add_temporary_elements(self, fov_box, intersections, point_cloud):
        """Clear and redraw temporary elements"""
        # Clear old elements
        for artist in [self.fov_patch, self.intersection_points, self.point_cloud_points]:
            if artist:
                try:
                    artist.remove()
                except:
                    pass
        
        # Create new FOV patch
        if fov_box and hasattr(fov_box, 'exterior'):
            self.fov_patch = plt.Polygon(
                list(fov_box.exterior.coords),
                closed=True, fill=False, color='red', alpha=0.5, 
                label='FOV Boundary', zorder=5
            )
            self.ax.add_patch(self.fov_patch)
        
        # Update intersection points
        if intersections.any():
            ix, iy = zip(*[(p[0], p[1]) for p in intersections])
            self.intersection_points, = self.ax.plot(
                ix, iy, 'ro', markersize=3, alpha=0.7, 
                label='Cartographic Points', zorder=8
            )
        
        # Update point cloud
        if point_cloud:
            ix, iy = zip(*[(p[0], p[1]) for p in point_cloud])
            self.point_cloud_points, = self.ax.plot(
                ix, iy, 'gx', markersize=3, alpha=0.7, 
                label='ZED Point Cloud', zorder=10
            )

        # Update the legend to include all elements
        self.update_legend()

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
        self.point_cloud_points, = self.ax.plot([], [], 'gx', markersize=3, alpha=0.7, zorder=10)  # Initialize empty
    
        # Register ALL artists that need updating
        self._artists = [self.line_est, self.line_corr, self.point_cloud_points]

    def update_legend(self):
        """Update the legend with all current elements"""
        handles, labels = self.ax.get_legend_handles_labels()
        
        # Get handles from all possible artists
        all_handles = [
            self.line_est,
            self.line_corr,
            self.fov_patch,
            self.intersection_points,
            self.point_cloud_points
        ]
        
        # Filter out None values and get their labels
        valid_handles = [h for h in all_handles if h is not None]
        valid_labels = [h.get_label() for h in valid_handles]
        
        # Update legend
        self.ax.legend(handles=valid_handles, labels=valid_labels, loc='upper left')



    def close(self):
        """Clean up"""
        self.is_running = False
        if self.animation:
            self.animation.event_source.stop()
        plt.close(self.fig)