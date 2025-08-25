from collections import deque
from matplotlib.figure import Figure 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk) 
import tkinter as tk
import customtkinter
import numpy as np
customtkinter.set_appearance_mode("Light")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"
import sys, os
sys.path.append(os.path.abspath(""))
from src.briann.python.network import components as bpnc
from src.briann.python.training import data_management as bptdm
import networkx as nx
from customtkinter import filedialog    
import tkinter as tk
from typing import List, Tuple
from CTkMenuBar import *
import json, torch
from src.briann.python.utilities import utilities as bpuu

def get_dpi():
    # Create a hidden root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Get the DPI
    dpi = root.winfo_fpixels('1i')  # '1i' means 1 inch
    root.destroy()  # Destroy the hidden window
    return dpi

class Animator(customtkinter.CTk):

    @property
    def briann(self):
        return self._briann

    @property
    def selected_area(self):
        return self._selected_area
    
    @selected_area.setter
    def selected_area(self, new_value):
        # Ensure input validity
        if (not new_value == None) and (not isinstance(new_value, bpnc.Area)): raise TypeError(f"The selected area was expected to be of type Area but was {type(new_value)}.")
        
        # Set property
        self._selected_area = new_value

    def __init__(self, briann: bpnc.BrIANN):
        # Call super
        super().__init__()
        
        # Set properties
        self._briann = briann
        self.selected_area = None

        # Start simulation
        self.briann.start_next_trial_batch()

        # Configure window
        self.title("BrIANN Animator")
        self.dpi = get_dpi()
        self.geometry(f"{(int)(18*self.dpi)}x{(int)(10*self.dpi)}")

        # Controller
        self.controller_frame = ControllerFrame(briann=briann, master=self, corner_radius=0)
        self.controller_frame.pack(fill='x', expand=False, side=tk.BOTTOM)
        
        # Canvas
        self.canvas = Canvas(briann=briann, master=self, xscrollincrement=1, yscrollincrement=1)
        self.canvas.pack(expand=True, fill='both')
        
        # Time Label
        self.time_label = customtkinter.CTkLabel(self.canvas, text=f"Time: {self.briann.current_simulation_time:.3f} s", anchor=tk.CENTER, font=("Arial", 16))
        self.time_label.place(relx=0.5, rely=0.01, anchor=tk.N)
        
        # Quit handler
        self.protocol('WM_DELETE_WINDOW', self._on_close_window)  # root is your root window

    def _on_close_window(self):
        
        # Close all matplotlib figures to prevent errors
        plt.close('all')  
        self.after(50, self.destroy)
        
class ControllerFrame(customtkinter.CTkFrame):

    def __init__(self, briann: bpnc.BrIANN, **kwargs):
        
        # Super
        super().__init__(**kwargs)

        # Set proeprties
        self._briann = briann
        self.dpi = get_dpi()

        # Create widgets
        customtkinter.CTkButton(self, text="Previous Stimulus", command=self.on_previous_stimulus_button_click).pack(expand=True, side=tk.LEFT, padx=10, pady=10)
        customtkinter.CTkButton(self, text="First Time-Frame", command=self.on_first_time_frame_button_click).pack(expand=True, side=tk.LEFT,  padx=10, pady=10)
        customtkinter.CTkButton(self, text="Previous Time-Frame", command=self.on_previous_time_frame_button_click).pack(expand=True, side=tk.LEFT,  padx=10, pady=10)
        customtkinter.CTkButton(self, text="Play", command=self.on_play_button_click).pack(expand=True, side=tk.LEFT, padx=10, pady=10)
        customtkinter.CTkButton(self, text="Next Time-Frame", command=self.on_next_time_frame_button_click).pack(expand=True, side=tk.LEFT, padx=10, pady=10)
        customtkinter.CTkButton(self, text="Last Time-Frame", command=self.on_last_time_frame_button_click).pack(expand=True, side=tk.LEFT, padx=10, pady=10)
        customtkinter.CTkButton(self, text="Next Stimulus", command=self.on_next_stimulus_button_click).pack(expand=True, side=tk.LEFT, padx=10, pady=10)
        
    def on_previous_stimulus_button_click(self):
        print("Previous stimulus button clicked")

    def on_first_time_frame_button_click(self):
        print("Reset time button clicked")

    def on_previous_time_frame_button_click(self):
        print("Previous time frame button clicked")

    def on_play_button_click(self):
        print("Play button clicked")

    def on_next_time_frame_button_click(self):
        self._briann.step()
        self.master.canvas.network_visualizer.update()
        self.master.time_label.configure(text=f"Time: {self._briann.current_simulation_time:.3f} s")
        

    def on_last_time_frame_button_click(self):
        print("Next time frame button clicked")

    def on_next_stimulus_button_click(self):
        print("Next time frame button clicked")
  
    
class Canvas(tk.Canvas):

    def __init__(self, briann: bpnc.BrIANN, **kwargs):
        super().__init__(**kwargs)

        # Set properties
        self._briann = briann
        self.dpi = get_dpi()
        
        # Add visualizers
        self.create_reference_grid(x_min=-10, x_max=10, y_min=-10, y_max=10, spacing=1.0)
        #self.create_cross(x=0, y=0, size=0.5)
        self.network_visualizer = NetworkVisualizer(briann=briann, canvas=self, initial_x=0.0, initial_y=0.0, width=4, height=4, area_size=0.5)
        
        self._drag_start_x = None; self._drag_start_y = None # For panning
        self.bind("<ButtonPress-1>", self.left_mouse_press)
        self.bind("<B1-Motion>", self.drag_motion)
        self.bind("<ButtonRelease-1>", lambda event: setattr(self, "_drag_start_x", None) or setattr(self, "_drag_start_y", None))  # Reset mouse position on release
        #self.bind("<MouseWheel>", self.zoom)
        self.bind(sequence="<Configure>", func= self.center_scroll_region)  # Update scroll region to fit all items
        
        self.resized = False

    def create_reference_grid(self, x_min: float, x_max: float, y_min: float, y_max: float, spacing: float = 1.0, color: str = 'lightgray') -> None:
        """Creates a reference grid on the canvas with the given parameters.
        :param x_min: The minimum x-coordinate of the grid in cartesian space (inches).
        :type x_min: float
        :param x_max: The maximum x-coordinate of the grid in cartesian space (inches).
        :type x_max: float
        :param y_min: The minimum y-coordinate of the grid in cartesian space (inches).
        :type y_min: float
        :param y_max: The maximum y-coordinate of the grid in cartesian space (inches).
        :type y_max: float
        :param spacing: The spacing between the grid lines in cartesian space (inches).
        :type spacing: float
        :param color: The color of the grid lines.
        :type color: str"""
        
        # Vertical lines
        x = 0
        while x <= x_max:
            x1, y1 = self.cartesian_to_canvas(x=x, y=y_min)
            x2, y2 = self.cartesian_to_canvas(x=x, y=y_max)
            self.create_line(x1, y1, x2, y2, fill=color, width=1)
            if x != 0:
                x1, y1 = self.cartesian_to_canvas(x=-x, y=y_min)
                x2, y2 = self.cartesian_to_canvas(x=-x, y=y_max)
                self.create_line(x1, y1, x2, y2, fill=color, width=1)
            x += spacing

        # Horizontal lines
        y = 0
        while y <= y_max:
            x1, y1 = self.cartesian_to_canvas(x=x_min, y=y)
            x2, y2 = self.cartesian_to_canvas(x=x_max, y=y)
            self.create_line(x1, y1, x2, y2, fill=color, width=1)
            if y != 0:
                x1, y1 = self.cartesian_to_canvas(x=x_min, y=-y)
                x2, y2 = self.cartesian_to_canvas(x=x_max, y=-y)
                self.create_line(x1, y1, x2, y2, fill=color, width=1)
            y += spacing

    def create_cross(self, x, y, size, color='red'):
        x1, y1 = self.cartesian_to_canvas(x=x-size/2, y=y)
        x2, y2 = self.cartesian_to_canvas(x=x+size/2, y=y)
        self.create_line(x1, y1, x2, y2, fill=color, width=2)
        x1, y1 = self.cartesian_to_canvas(x=x, y=y-size/2)
        x2, y2 = self.cartesian_to_canvas(x=x, y=y+size/2)
        self.create_line(x1, y1, x2, y2, fill=color, width=2)

    def center_scroll_region(self, event):
        """Centers the scroll region of the canvas."""
        if not self.resized:
            # Get the current scroll region
            scroll_region = self.bbox(tk.ALL)
            
            # Calculate the center position
            center_x = scroll_region[0] + (scroll_region[2] - scroll_region[0]) / 2
            center_y = scroll_region[1] + (scroll_region[3] -scroll_region[1]) / 2
            
            # Set the canvas view to center the scroll region
            self.xview_scroll(-(int)(self.winfo_width()/2-center_x), "units")
            self.yview_scroll(-(int)(self.winfo_height()/2-center_y), "units")

            self.resized = True
        
    @property
    def briann(self):
        return self._briann

    def get_area_button(self, label: int) -> customtkinter.CTkButton:
        
        # Fetch
        result = None
        for button in self.area_buttons.values():
            if button.cget('text') == label: result = button

        # Return
        return result

    def left_mouse_press(self, event):
        widget = event.widget
        widget._press_x = event.x
        widget._press_y = event.y
        widget._drag_start_x = event.x
        widget._drag_start_y = event.y
        
    def drag_motion(self, event):
        widget = event.widget
        if widget._drag_start_x != None and widget._drag_start_y != None:
            dx = event.x - widget._drag_start_x
            dy = event.y - widget._drag_start_y 
            widget.xview_scroll(-dx, "units")
            widget.yview_scroll(-dy, "units")
            
            widget._drag_start_x = event.x
            widget._drag_start_y = event.y

    def zoom(self, event):
        x = self.canvasx(event.x)
        y = self.canvasy(event.y)
        factor = 1.001 ** event.delta
        self.scale(tk.ALL, x, y, factor, factor)

    def cartesian_to_canvas(self, x: float, y: float) -> Tuple[int, int]:
        """Converts from a cartesian coordinate system to this canvas's coordinate system.
        - The cartesian coordinate system has its origin in the center of the canvas, with the x-axis pointing to the right and the y-axis pointing upwards and its units are inches.
        - The canvas coordinate system has its origin in the top left corner of the canvas, with the x-axis pointing to the right and the y-axis pointing downwards and its units are pixels.
        
        :param x: The x-coordinate in cartesian coordinates.
        :type x: float
        :param y: The y-coordinate in cartesian coordinates.
        :type y: float
        :return: The x and y coordinates in canvas space.
        :rtype: Tuple[int, int]"""

        return (int)(x*self.dpi), (int)(-y*self.dpi)

    def canvas_to_cartesian(self, x: int, y: int) -> Tuple[float, float]:
        """Converts from this canvas's coordinate system to a cartesian coordinate system.
        - The cartesian coordinate system has its origin in the center of the canvas, with the x-axis pointing to the right and the y-axis pointing upwards and its units are inches.
        - The canvas coordinate system has its origin in the top left corner of the canvas, with the x-axis pointing to the right and the y-axis pointing downwards and its units are pixels.
        
        :param x: The x-coordinate in canvas coordinates.
        :type x: int
        :param y: The y-coordinate in canvas coordinates.
        :type y: int
        :return: The x and y coordinates in cartesian space.
        :rtype: Tuple[float, float]"""

        return (float)(x)/self.dpi, -(float)(y)/self.dpi

class DraggableWidget():
    """Creates a visualizer that visualizes data in a rectangle whose center is at (`x`,`y`) with provided `width` and `height`.
        The visualizer will be drawn on top of all previously drawn elements on the `canvas` and it can be dragged around with the mouse.

        :param canvas: The canvas to draw the visualizer on.
        :type canvas: Canvas
        :param x: The x-coordinate of the visualizer's center in Cartesian space (inches).
        :type x: float
        :param y: The y-coordinate of the visualizer's center in Cartesian space (inches).
        :type y: float
        :param width: The width of the visualizer in inches.
        :type width: float
        :param height: The height of the visualizer in inches.
        :type height: float
        :return: The visualizer that was created.
        :rtype: :py:class:`.Visualizer`"""
        
    def __init__(self, canvas: Canvas, widget: tk.Widget, initial_x: float, initial_y: float) -> "DraggableWidget":
        
        # Set properties
        self._canvas = canvas
        
        # Create window on canvas
        self._initial_x, self._initial_y = self.canvas.cartesian_to_canvas(x=initial_x, y=initial_y) # Convert to canvas space
        self._x, self._y = self._initial_x, self._initial_y # Current position in canvas space
        
        self._window_index = self.canvas.create_window(self._x, self._y, window=widget, anchor=tk.CENTER)
            
        # Position along Z-Axis
        widget.tk.call('lower', widget._w, None)
        widget.tk.call('lower', self.canvas._w, None)

        # Add drag listeners
        widget.bind("<ButtonPress-1>", self._on_drag_start)
        widget.bind("<B1-Motion>", self._on_drag_motion)
        widget.bind("<ButtonRelease-1>", self._on_drag_end)  # Reset mouse position on release
        
    @property
    def canvas(self) -> Canvas:
        """
        :return: The canvas on which this visualizer is drawn.
        :rtype: Canvas"""
        return self._canvas
    
    @property
    def x(self) -> float:
        """
        :return: The x-position of the center of self in cartesian space (inches).
        :rtype: float"""
        
        # Convert to cartesian space
        x, _ = self._canvas.canvas_to_cartesian(x=self._x, y=self._y)
        
        # Output
        return x
    
    @property
    def y(self) -> float:
        """
        :return: The y-position of the center of self in cartesian space (inches)f.
        :rtype: float"""
        
        # Convert to cartesian space
        _, y = self._canvas.canvas_to_cartesian(x=self._x, y=self._y)
        
        # Output
        return y

    def _on_drag_start(self, event) -> None:
        """Begins a drag operation.
        :param event: The event that triggered this method.
        :type event: tk.Event"""
        
        # Save mouse position in canvas space
        self._mouse_x = self._initial_x + event.x
        self._mouse_y = self._initial_y + event.y
        
    def _on_drag_motion(self, event) -> None:
        """Begins a drag operation.
        :param event: The event that triggered this method.
        :type event: tk.Event"""
        
        # Compute delta
        dx = self._initial_x + event.x - self._mouse_x
        dy = self._initial_y + event.y - self._mouse_y
        
        # Move self on canvas
        self.canvas.move(self._window_index, dx, dy)
        self._x += dx
        self._y += dy
        
    def _on_drag_end(self, event) -> None:
        """Ends a drag operation.
        :param event: The event that triggered this method.
        :type event: tk.Event"""
        
        # Reset mouse position
        self._mouse_x = 0
        self._mouse_y = 0
        
        # Update initial position
        self._initial_x, self._initial_y = self._x, self._y

class Area(DraggableWidget):
    
    def __init__(self, area: bpnc.Area, canvas: Canvas, x: float, y: float, size: float) -> "Area":
        """x, y, and size are in inches"""


        # Create button
        self._button = customtkinter.CTkButton(canvas, 
                                         text = area.index, 
                                         fg_color='lightgray', 
                                         border_color="darkgray",#"#325882",# "#14375e", 
                                         border_width=0.05*size*canvas.dpi, 
                                         text_color='black', 
                                         anchor = tk.CENTER, 
                                         width=size*canvas.dpi, # Dynamcially adjusts width based on text length
                                         height=size*canvas.dpi, 
                                         corner_radius=0.5*size*canvas.dpi)
        self._button.bind("<Button-2>", lambda area_index=area.index: self._on_area_click(area_index=area_index))
        self._button.bind("<Button-3>", lambda area_index=area.index: self._on_area_click(area_index=area_index))

        # Call super
        super().__init__(canvas=canvas, widget=self._button, initial_x=x, initial_y=y)
        
        # Set properties
        self._area = area
        self._subscribers = []

    def display_as_active(self) -> None:
        """Displays this area as active."""
        self._button.configure(fg_color='orange', text_color='white')

    def display_as_inactive(self) -> None:
        """Displays this area as inactive."""
        self._button.configure(fg_color='lightgray', text_color='black')

    @property
    def area(self) -> bpnc.Area:
        """
        :return: The area that is visualized by this button.
        :rtype: bpnc.Area"""
        return self._area

    def add_subscriber(self, subscriber):
        self._subscribers.append(subscriber)

    def _on_drag_motion(self, event) -> None:
        super()._on_drag_motion(event=event)
        
        # Redraw connections
        for subscriber in self._subscribers:
            subscriber.on_area_reposition()

    def _on_area_click(self, area_index: str):
        
        # Create a popup
        popup = tk.Toplevel()
        popup.geometry(f"{(int)(2*self.canvas.dpi)}x{(int)(2*self.canvas.dpi)}+{self.canvas.winfo_rootx()+self._button.winfo_x()}+{self.canvas.winfo_rooty()+self._button.winfo_y()}")
        popup.overrideredirect(True) # Prevent window decorations
        
        # Add widgets to popup
        customtkinter.CTkLabel(popup, text="Add Visualizer", anchor=tk.CENTER).pack(expand=True, fill="x", padx=10, pady=10)
        option_menu = customtkinter.CTkOptionMenu(popup, values=["State", "Input", "Output"])
        option_menu.pack(expand=True, fill="x", padx=10, pady=10)
        customtkinter.CTkButton(popup, text="Add", 
                                command=lambda option_menu=option_menu, area_index=area_index, popup=popup: self.add_state_visualizer(option=option_menu.get(), area_index=area_index, popup=popup)
                                ).pack(expand=True, fill="x", padx=10, pady=10)
        
        # Display popup on top of everything else
        popup.lift()
             
    def add_state_visualizer(self, area_index:int, option: str, popup: customtkinter.CTkToplevel):
        StateVisualizer1D(area=self.area, initial_x=self.x, initial_y=self.y, width=3, height=2, canvas=self.canvas)
        popup.destroy()
        
class Connection():
    
    def __init__(self, connection: bpnc.Connection, from_area: Area, to_area: Area, canvas: Canvas, thickness: float = 2.0, bend_by: float = 0.0) -> "Connection":
        
        # Set properties
        self.canvas = canvas
        self.from_area = from_area
        self.to_area = to_area
        self.thickness = thickness
        self.bend_by = bend_by
        
        # Draw connection
        self.draw()

    def draw(self):

        # Get start end points from area as well as midpoint
        x0, y0 = self.from_area.x, self.from_area.y # Starting point
        x1, y1 = self.to_area.x, self.to_area.y # Endpoint
        d01 = (x1-x0, y1-y0 if y1-y0 != 0 else 0.0001) # Prevent division by zero
        xm, ym = x0+0.5*d01[0], y0+0.5*d01[1] # Mid-point
        
        # Compute bend points orthogonal to midpoint
        d_orthogonal = (1, -d01[0]/d01[1])
        d_orthogonal_len = np.sqrt(d_orthogonal[0]**2+d_orthogonal[1]**2)
        d_orthogonal = (d_orthogonal[0]/d_orthogonal_len, d_orthogonal[1]/d_orthogonal_len)
        xmf, ymf = xm+self.bend_by*d_orthogonal[0], ym+self.bend_by*d_orthogonal[1] # Bend point for forward pointing arrow
        xmb, ymb = xm-self.bend_by*d_orthogonal[0], ym-self.bend_by*d_orthogonal[1] # Bend point for backward pointing arrow
        
        # Draw on screen space
        x0, y0 = self.canvas.cartesian_to_canvas(x=x0, y=y0)
        xmf, ymf = self.canvas.cartesian_to_canvas(x=xmf, y=ymf)
        xmb, ymb = self.canvas.cartesian_to_canvas(x=xmb, y=ymb)
        x1, y1 = self.canvas.cartesian_to_canvas(x=x1, y=y1)
        if self.from_area.area.index < self.to_area.area.index: 
            self.first_segment = self.canvas.create_line(x0,y0, xmf,ymf, width=self.thickness, arrow='last'); self.second_segment = self.canvas.create_line(xmf,ymf, x1,y1, width=self.thickness)
        else: 
            self.first_segment = self.canvas.create_line(x0,y0, xmb,ymb, width=self.thickness, arrow='last'); self.second_segment = self.canvas.create_line(xmb,ymb, x1,y1, width=self.thickness)

    def on_area_reposition(self):
        """Re-draws the connection when an area is repositioned."""
        
        # Remove old segments
        self.canvas.delete(self.first_segment)
        self.canvas.delete(self.second_segment)
        
        # Draw new segments
        self.draw()

class NetworkVisualizer():
    """Visualizes the network structure of a given `briann` instance in a rectangle whose center is at (`x`,`y`) with provided `width` and `height`.
        The visualizer will be drawn on top of all previously drawn elements on the `canvas` and it can be dragged around with the mouse.

        :param briann: The BrIANN instance to visualize.
        :type briann: bpnc.BrIANN
        :param canvas: The canvas to draw the visualizer on.
        :type canvas: Canvas
        :param x: The x-coordinate of the visualizer's center in Cartesian space (inches).
        :type x: float
        :param y: The y-coordinate of the visualizer's center in Cartesian space (inches).
        :type y: float
        :param width: The width of the visualizer in inches.
        :type width: float
        :param height: The height of the visualizer in inches.
        :type height: float
        :return: The visualizer that was created.
        :rtype: :py:class:`.NetworkVisualizer`"""

    def __init__(self, briann: bpnc.BrIANN, canvas: Canvas, initial_x: float, initial_y: float, width: float, height: float, area_size: float) -> "NetworkVisualizer":
        
        # Set properties
        self.canvas = canvas
        self._briann = briann
        
        # Override the drag functions from super to extend it to all child widgets

        # Compute positions
        G = self.briann.to_simple_networkx()
        area_to_position = nx.shell_layout(G=G, center=(0,0))

        # Convert from G space to canvas space
        x_min, x_max = min([pos[0] for pos in area_to_position.values()]), max([pos[0] for pos in area_to_position.values()]) # G space
        y_min, y_max = min([pos[1] for pos in area_to_position.values()]), max([pos[1] for pos in area_to_position.values()]) # G space
        x_range, y_range = x_max-x_min, y_max-y_min # G space
        for area, position in area_to_position.items():
            x, y = position[0] / (0.5*x_range), position[1] / (0.5*y_range) # Ensure width and height of whole network is between -1 and 1
            x, y = x*0.5*(width-area_size), y*0.5*(height-area_size) # Scale to visualizer size
            area_to_position[area] = (x+initial_x, y+initial_y)

        # Create area drawables
        self.area_to_drawable = {}
        for area in self.briann.areas:
            x, y = area_to_position[area][0], area_to_position[area][1]
            self.area_to_drawable[area] = Area(area=area, canvas=self.canvas, x=x, y=y, size=area_size)
        
        # Draw the edges
        width = 0.05*area_size
        curved_edges = [edge for edge in G.edges() if reversed(edge) in G.edges()]
        self.edge_to_drawable = {}
        for (u,v) in G.edges():
            # Convert index to area
            u = self.briann.get_area_at_index(index=u.index)
            v = self.briann.get_area_at_index(index=v.index)
            bend_by = 0.1 if (u,v) in curved_edges else 0.0
            self.edge_to_drawable[u.index, v.index] = Connection(connection=None, from_area=self.area_to_drawable[u], to_area=self.area_to_drawable[v], canvas=self.canvas, thickness=width, bend_by=bend_by)
            self.area_to_drawable[u].add_subscriber(self.edge_to_drawable[u.index, v.index])
            self.area_to_drawable[v].add_subscriber(self.edge_to_drawable[u.index, v.index])
        
    def update(self):
        for area, drawable in self.area_to_drawable.items():
            if area in self._briann._due_areas:
                drawable.display_as_active()
            else: drawable.display_as_inactive()
            
    @property
    def briann(self) -> bpnc.BrIANN:
        """
        :return: The BrIANN instance that is visualized by this visualizer.
        :rtype: bpnc.BrIANN"""
        return self._briann

class StateVisualizer(DraggableWidget, bpnc.AreaStateSubscriber):
    """Superclass for a set of classes that create 2D visualizations of a :py:meth:`.TimeFrame.state` on a 1x1 unit square"""

    def __init__(self, area: bpnc.Area, canvas: Canvas, initial_x: float, initial_y: float, width: float, height: float):
    
        # Set proeprties
        self.bpnc = area

        # Create Figure
        self.figure = plt.figure(figsize=(width, height), dpi=canvas.dpi)
        self.figure.set_tight_layout(True)
        widget = FigureCanvasTkAgg(plt.gcf()).get_tk_widget()
        
        # Call super
        super().__init__(canvas=canvas, widget=widget, initial_x=initial_x, initial_y=initial_y)

        # Subscribe to area
        area.add_state_subscriber(subscriber=self)

        # Initial draw
        self.update_plot()
        
    def receive_state(self, area_index: int, time_frame: bpnc.TimeFrame) -> None:
        plt.figure(self.figure.number)
        self.update_plot(time_frame=time_frame)

    def update_plot(self, time_frame: bpnc.TimeFrame = None) -> None:
        pass
        
class StateVisualizer1D(StateVisualizer):
    """Visualizes the input of an area in a 2D plot."""

    def __init__(self, area: bpnc.Area, canvas: Canvas, initial_x: float, initial_y: float, width: float, height: float) -> "StateVisualizer1D":
        super().__init__(area=area, canvas=canvas, initial_x=initial_x, initial_y=initial_y, width=width, height=height)
        

    def update_plot(self, time_frame: bpnc.TimeFrame = None) -> None:
        
        if time_frame is None:
            plt.clf()
            self.ts = []
            self.ys = []
            plt.xlabel("Time (s)")
            plt.ylabel("State")
            plt.title(f"Area {self.bpnc.index}")
            plt.grid(True)
            self.figure.canvas.draw()
        else:
            plt.clf()
            self.ys.append(time_frame.state.cpu().detach().numpy()[0,:]) 
            self.ts.append(np.repeat([time_frame.time_point], time_frame.state.shape[-1]))
            plt.plot(self.ts, self.ys)
            plt.xlabel("Time (s)")
            plt.ylabel("State")
            plt.title(f"Area {self.bpnc.index}")
            plt.grid(True)
            self.figure.canvas.draw()
        

if __name__ == "__main__":
    
    path = bpuu.map_path_to_os(path="tests/briann 1.json")
    with open(path, 'r') as file:
        configuration = json.loads(file.read())

    briann = bpnc.BrIANN(configuration=configuration)
    app = Animator(briann=briann)
    app.mainloop()
    