import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg) 
import tkinter as tk
import customtkinter
import numpy as np

from briann.utilities import callbacks as bpuc
customtkinter.set_appearance_mode("Light")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"
import sys, os
sys.path.append(os.path.abspath(""))
from briann.network import core as bnc
import networkx as nx   
import tkinter as tk
from typing import Tuple, List, Iterator
from CTkMenuBar import *
import threading, time
from abc import ABC, abstractmethod


# Get the DPI
root = tk.Tk()
root.withdraw()  # Hide the root window
DPI = root.winfo_fpixels('1i')  # '1i' means 1 inch
root.destroy()  # Destroy the hidden window

class Animator(customtkinter.CTk):
    """This class can be used to animate :py:class:`~briann.network.core.BrIANN`'s. The animator displays a graph for the different :py:class:`~briann.network.core.Area`'s and :py:class:`~briann.network.core.Connections`'s.
    The layout of the graph can be adjusted with mouse-clicks inside the animator. It is also possible to add visualisations for area states by right-clicking the areas. 
    When stepping through the simulation, the animator indicates with a color-code which area is currently processing its inputs and updates to their states are displayed in the corresponding visualizers.
    
    :param briann: The briann instance to be animated.
    :type briann: :py:class:`~briann.network.core.BrIANN`
    :param data_iterator: An iterator that provides the data (X, y (ignored)) to be streamed to the briann model during the simulation.
    :type data_iterator: Iterator
    """

    @property
    def briann(self):
        """:return: The briann instance that is being animated.
        :rtype: :py:class:`~briann.GUI.model_explorer.Animator`"""

        return self._briann

    def __init__(self, briann: bnc.BrIANN, data_iterator: Iterator) -> None:
        # Call super
        super().__init__()
        
        # Start simulation
        self._data_iterator = data_iterator
        X, y = next(data_iterator)
        briann.load_next_stimulus_batch(X=X)

        # Configure window
        self.title("BrIANN Animator")
        self.geometry(f"{(int)(18*DPI)}x{(int)(10*DPI)}")

        # Canvas
        canvas = Canvas(master=self, xscrollincrement=1, yscrollincrement=1)
        canvas.pack(expand=True, fill='both')
        self._briann = briann
        
        # Network Visualizer
        network_visualizer = NetworkVisualizer(briann=briann, canvas=canvas, initial_x=0.0, initial_y=0.0, width=4, height=4, area_size=0.5)

        # Time Label
        self._time_label = customtkinter.CTkLabel(canvas, text=f"Time: {briann.current_simulation_time:.3f} s", anchor=tk.CENTER, font=("Arial", 16))
        self._time_label.place(relx=0.5, rely=0.01, anchor=tk.N)

        # Add self as subscriber for briann simulation time
        bpuc.CallbackManager.add_callback_to_attribute(target_class=type(briann), target_instance=briann, attribute_name='_current_simulation_time', callback=self.on_current_simulation_time_update)
        
        # Controller
        controller_frame = ControllerFrame(briann=briann, network_visualizer=network_visualizer, master=self, corner_radius=0)
        controller_frame.pack(fill='x', expand=False, side=tk.BOTTOM)
        
        # Quit handler
        self.protocol(name='WM_DELETE_WINDOW', func=self._on_window_close) 
    
    def on_current_simulation_time_update(self, obj, name, value) -> None:
        
        # Set time-label
        self._time_label.configure(text=f"Time: {value:.3f} s")
        
    def _on_window_close(self) -> None:
        """Ensures the animator closes all visualizations properly before quitting the app.
        :rtype: None"""

        # Close all matplotlib figures to prevent errors
        plt.close('all')  
        self.after(50, self.destroy)

class ControllerFrame(customtkinter.CTkFrame):
    """A frame that holds the different buttons used to control the animation.
    
    :param briann: The briann instance for which the animation shall be controlled.
    :type briann: :py:class:`~briann.network.core.BrIANN`
    :param network_visualizer: The visualizer that displays the current state of `briann`.
    :type network_visualizer: :py:class:`~briann.GUI.model_explorer.NetworkVisualizer`
    """

    def __init__(self, briann: bnc.BrIANN, network_visualizer: "NetworkVisualizer", **kwargs) -> None:
        
        # Super
        super().__init__(**kwargs)

        # Set properties
        self._briann = briann
        self._network_visualizer = network_visualizer
        self._is_playing = False

        # Create buttons
        self._play_button = customtkinter.CTkButton(self, text="Play", command=self._on_play_button_click)
        self._play_button.pack(expand=True, side=tk.LEFT, padx=0, pady=10, anchor=tk.CENTER)
        customtkinter.CTkButton(self, text="Pause", command=self._on_pause_button_click).pack(expand=True, side=tk.LEFT, padx=0, pady=10, anchor=tk.CENTER)
        customtkinter.CTkButton(self, text="Next Time-Frame", command=self.on_next_time_frame_button_click).pack(expand=True, side=tk.LEFT, padx=0, pady=10, anchor=tk.CENTER)
        customtkinter.CTkButton(self, text="Next Stimulus", command=self.on_next_stimulus_button_click).pack(expand=True, side=tk.LEFT, padx=0, pady=10, anchor=tk.CENTER)
        
    def _on_play_button_click(self) -> None:
        """Starts a loop that steps through the animation until stopped via :py:meth:`~briann.gui.model_explorer.ControllerFrame._on_pause_button_click`.
        :rtype: None"""

        # Set property
        self._is_playing = True

        # Start loop on separate thread
        threading.Thread(target=self._play).start()

        # Disable button
        self._play_button.configure(state='disabled')
        
    def _play(self) -> None:
        """Steps through the animation for as long as :py:attr:`~Canvas._is_playing` is True.
        :rtype: None"""

        # Loop
        while self._is_playing:

            # Step
            self.on_next_time_frame_button_click()

            # Wait
            time.sleep(1)

    def _on_pause_button_click(self) -> None:
        """Halts the animation, in case it is currently running.
        :rtype: None"""

        # Set property
        self._is_playing = False

        # Enable button
        self._play_button.configure(state='normal')

    def on_next_time_frame_button_click(self) -> None:
        """Loads the next :py:class:`~briann.network.core.TimeFrame` into the :py:class:`~briann.network.core.BrIANN` model.
        :rtype: None"""

        # Step
        self._briann.step() # The area state subscribers will automatically update the UI
        
    def on_next_stimulus_button_click(self) -> None:
        """Loads the next stimulus into the :py:class:`~briann.network.core.BrIANN` model.
        :rtype: None"""

        # Ensure current simulation is paused
        self._is_playing = False
        
        # Load next stimulus
        X, y = next(self._data_iterator)
        self._briann.load_next_stimulus_batch(X=X)

class Canvas(tk.Canvas):
    """This class creates a canvas on which network components will be displayed. The canvas can be dragged around and has a reference grid in the background.
    
    """

    def __init__(self, **kwargs) -> None:
        super(Canvas, self).__init__(**kwargs)
        
        # Add reference grid
        self.create_reference_grid(x_min=-15, x_max=15, y_min=-15, y_max=15, spacing=1.0)
        
        self._drag_start_x = None; self._drag_start_y = None # For panning
        self.bind("<ButtonPress-1>", self._on_drag_start)
        self.bind("<B1-Motion>", self._on_drag_motion)
        self.bind("<ButtonRelease-1>", lambda event: setattr(self, "_drag_start_x", None) or setattr(self, "_drag_start_y", None))  # Reset mouse position on release
        self.bind(sequence="<Configure>", func= self.center_scroll_region)  # Update scroll region to fit all items
        
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
        :type color: str
        :rtype: None"""
        
        # Vertical lines
        x = 0
        while x <= x_max:
            x1, y1 = Canvas.cartesian_to_canvas(x=x, y=y_min)
            x2, y2 = Canvas.cartesian_to_canvas(x=x, y=y_max)
            self.create_line(x1, y1, x2, y2, fill=color, width=1)
            if x != 0:
                x1, y1 = Canvas.cartesian_to_canvas(x=-x, y=y_min)
                x2, y2 = Canvas.cartesian_to_canvas(x=-x, y=y_max)
                self.create_line(x1, y1, x2, y2, fill=color, width=1)
            x += spacing

        # Horizontal lines
        y = 0
        while y <= y_max:
            x1, y1 = Canvas.cartesian_to_canvas(x=x_min, y=y)
            x2, y2 = Canvas.cartesian_to_canvas(x=x_max, y=y)
            self.create_line(x1, y1, x2, y2, fill=color, width=1)
            if y != 0:
                x1, y1 = Canvas.cartesian_to_canvas(x=x_min, y=-y)
                x2, y2 = Canvas.cartesian_to_canvas(x=x_max, y=-y)
                self.create_line(x1, y1, x2, y2, fill=color, width=1)
            y += spacing

    def center_scroll_region(self, event: tk.Event) -> None:
        """Centers the scroll region of the canvas.

        :param event: The event that triggered the function call.
        :type event: :py:class:`tkinter.Event`
        :rtype: None"""
        
        # Get the current scroll region
        scroll_region = self.bbox(tk.ALL)
        
        # Calculate the center position
        center_x = scroll_region[0] + (scroll_region[2] - scroll_region[0]) / 2
        center_y = scroll_region[1] + (scroll_region[3] - scroll_region[1]) / 2
        
        # Set the canvas view to center the scroll region
        self.xview_scroll(-(int)(self.winfo_width()/2-center_x), "units")
        self.yview_scroll(-(int)(self.winfo_height()/2-center_y), "units")

    def _on_drag_start(self, event: tk.Event) -> None:
        """Prepares the drag operation

        :param event: The event that triggered the function call.
        :type event: :py:class:`tkinter.Event`
        :rtype: None"""

        self._press_x = event.x
        self._press_y = event.y
        self._drag_start_x = event.x
        self._drag_start_y = event.y
        
    def _on_drag_motion(self, event: tk.Event) -> None:
        """Performs a step of the drag operation.
        
        :param event: The event that triggered the function call.
        :type event: :py:class:`tkinter.Event`
        :rtype: None"""
        
        if self._drag_start_x != None and self._drag_start_y != None:
            dx = event.x - self._drag_start_x
            dy = event.y - self._drag_start_y 
            self.xview_scroll(-dx, "units")
            self.yview_scroll(-dy, "units")
            
            self._drag_start_x = event.x
            self._drag_start_y = event.y

    @staticmethod
    def cartesian_to_canvas(x: float, y: float) -> Tuple[int, int]:
        """Converts from a cartesian coordinate system to this canvas's coordinate system.
        - The cartesian coordinate system has its origin in the center of the canvas, with the x-axis pointing to the right and the y-axis pointing upwards and its units are inches.
        - The canvas coordinate system has its origin in the top left corner of the canvas, with the x-axis pointing to the right and the y-axis pointing downwards and its units are pixels.
        
        :param x: The x-coordinate in cartesian coordinates.
        :type x: float
        :param y: The y-coordinate in cartesian coordinates.
        :type y: float
        :return: The x and y coordinates in canvas space.
        :rtype: Tuple[int, int]"""

        return (int)(x*DPI), (int)(-y*DPI)

    @staticmethod
    def canvas_to_cartesian(x: int, y: int) -> Tuple[float, float]:
        """Converts from this canvas's coordinate system to a cartesian coordinate system.
        - The cartesian coordinate system has its origin in the center of the canvas, with the x-axis pointing to the right and the y-axis pointing upwards and its units are inches.
        - The canvas coordinate system has its origin in the top left corner of the canvas, with the x-axis pointing to the right and the y-axis pointing downwards and its units are pixels.
        
        :param x: The x-coordinate in canvas coordinates.
        :type x: int
        :param y: The y-coordinate in canvas coordinates.
        :type y: int
        :return: The x and y coordinates in cartesian space.
        :rtype: Tuple[float, float]"""

        return (float)(x)/DPI, -(float)(y)/DPI

class DraggableWidget():
    """Creates a visualizer that visualizes data in a rectangle whose center is at (`x`,`y`).
        The visualizer will be drawn on top of all previously drawn elements on the `canvas` and it can be dragged around with the mouse.

        :param canvas: The canvas to draw the visualizer on.
        :type canvas: Canvas
        :param widget: The widget to be placed on the canvas.
        :type widget: py:class:`tkinter.Widget`
        :param x: The x-coordinate of the visualizer's center in Cartesian space (inches).
        :type x: float, optional, defaults to 0
        :param y: The y-coordinate of the visualizer's center in Cartesian space (inches).
        :type y: float, optional, defaults to 0
        """
        
    def __init__(self, canvas: Canvas, widget: tk.Widget, x: float = 0, y: float = 0) -> None:
        
        # Set properties
        self._canvas = canvas
        self._initial_x, self._initial_y = Canvas.cartesian_to_canvas(x=x, y=y) # Convert to canvas space
        self._x, self._y = self._initial_x, self._initial_y # Current position in canvas space
        self._location_subscribers = []
                
        # Create window on canvas
        self._window_index = self._canvas.create_window(self._x, self._y, window=widget, anchor=tk.CENTER)
        
        # Position along Z-Axis
        widget.tk.call('lower', widget._w, None)
        widget.tk.call('lower', self._canvas._w, None)

        # Add drag listeners
        widget.bind("<ButtonPress-1>", self._on_drag_start)
        widget.bind("<B1-Motion>", self._on_drag_motion)
        widget.bind("<ButtonRelease-1>", self._on_drag_end)  # Reset mouse position on release
        
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

    def _on_drag_start(self, event: tk.Event) -> None:
        """Begins a drag operation.
        
        :param event: The event that triggered the function call.
        :type event: :py:class:`tkinter.Event`
        :rtype: None"""
        
        # Save mouse position in canvas space
        self._mouse_x = self._initial_x + event.x
        self._mouse_y = self._initial_y + event.y
            
    def add_location_subscriber(self, subscriber: "DraggableWidget.LocationSubscriber") -> None:
        """Adds a subscriber that will be notified once the location of this DraggableWidget changes.

        :param subscriber: The subscriber to be added to the subscription list.
        :type subscriber: :py:class:`~briann.GUI.model_explorer.DraggableWidget.LocationSubscriber`
        :rtype: None"""

        self._location_subscribers.append(subscriber)

    def _on_drag_motion(self, event: tk.Event) -> None:
        """Begins a drag operation.
        
        :param event: The event that triggered the function call.
        :type event: :py:class:`tkinter.Event`
        :rtype: None"""
        
        # Compute delta
        dx = self._initial_x + event.x - self._mouse_x
        dy = self._initial_y + event.y - self._mouse_y
        
        # Move self on canvas
        self._canvas.move(self._window_index, dx, dy)

        # Update position of self
        self._x += dx
        self._y += dy

        # Update subscribers
        for subscriber in self._location_subscribers:
            subscriber.on_location_update(draggable_widget=self)
        
    def _on_drag_end(self, event: tk.Event) -> None:
        """Ends a drag operation.
        
        :param event: The event that triggered the function call.
        :type event: :py:class:`tkinter.Event`
        :rtype: None"""
        
        # Reset mouse position
        self._mouse_x = 0
        self._mouse_y = 0
        
        # Update initial position
        self._initial_x, self._initial_y = self._x, self._y

    class LocationSubscriber(ABC):
        """An abstract base class for subscribers that want to receive the location of a :py:class:`~briann.GUI.model_explorer.DraggableWidget` every time it is updated. Subscribers must implement the :py:meth:`~briann.GUI.model_explorer.DraggableWidget.LocationSubscriber.on_location_update` method.
        """

        @abstractmethod
        def on_location_update(self, draggable_widget: "DraggableWidget") -> None:
            """This method will be called every time the :py:class:`~briann.GUI.model_explorer.DraggableWidget`'s :py:meth:`~briann.GUI.model_explorer.DraggableWidget.x` or :py:meth:`~briann.GUI.model_explorer.DraggableWidget.y` are updated.

            :param draggable_widget: The draggable widget that was updated.
            :type draggable_widget: :py:class:`~briann.GUI.model_explorer.DraggableWidget`
            :rtype: None
            """
            pass

class Area(DraggableWidget):
    """A visual representation of a :py:class:`~briann.network.core.BrIANN` that can be placed onto the :py:class:`~briann.GUI.model_explorer.Canvas`.
    It is represented by a button that can be moved around. The button changes color to indicate whether the corresponding BrIANN area is currently being updated.
    When right-clicking the button, an option menu unfolds that allows to add a visualizer for the state of the area.
    
    :param bpnc_area: The BrIANN area to be visualized.
    :type bpnc_area: :py:class:`~briann.network.core.Area`
    :param canvas: The canvas on which the area shall be visualized.
    :type canvas: :py:class:`~briann.GUI.model_explorer.Canvas`
    :param briann: The BrIANN that is being simulated.
    :type briann: :py:class:`~briann.network.core.BrIANN`
    :param x: The x-coordinate of the visualizer's center in Cartesian space (inches).
    :type x: float, optional, defaults to 0
    :param y: The y-coordinate of the visualizer's center in Cartesian space (inches).
    :type y: float, optional, defaults to 0
    """

    def __init__(self, bpnc_area: bnc.Area, canvas: Canvas, briann: bnc.BrIANN, x: float, y: float, size: float) -> None:
       
        # Set properties
        self._bpnc_area = bpnc_area
        self._briann = briann
        self._size = size

        # Link callbacks
        bpuc.CallbackManager.add_callback_to_attribute(target_class=type(bpnc_area), target_instance=bpnc_area, attribute_name='_update_count', callback=self.on_update_count_update)
        bpuc.CallbackManager.add_callback_to_attribute(target_class=type(briann), target_instance=briann, attribute_name='_current_simulation_time', callback=self.on_current_simulation_time_update)
        bpuc.CallbackManager.add_callback_to_method(target_instance=bpnc_area, method_name='reset', callback= lambda caller, selfy=self: selfy.on_current_simulation_time_update(obj=None, name=None, value=None)) # Reset color when area is reset
        
        # Create button
        self._button = customtkinter.CTkButton(canvas, 
                                         text = bpnc_area.index, 
                                         fg_color='lightgray', 
                                         border_color="darkgray",#"#325882",# "#14375e", 
                                         border_width=0.05*size*DPI, 
                                         text_color='black', 
                                         anchor = tk.CENTER, 
                                         width=size*DPI, # Dynamcially adjusts width based on text length
                                         height=size*DPI, 
                                         corner_radius=0.5*size*DPI)
        
        # Add right-click functionality 
        self._button.bind("<Button-2>", self._on_right_click)
        self._button.bind("<Button-3>", self._on_right_click)

        # Add state visualizers array
        self._state_visualizers = []

        # Call super
        super().__init__(canvas=canvas, widget=self._button, x=x, y=y)

    @property
    def size(self) -> float:
        """:return: The size of this area in inches.
        :rtype: float"""
        return self._size
    
    @property
    def state_visualizers(self) -> List["AreaStateVisualizer"]:
        """:return: The list of state visualizers that are currently attached to this area.
        :rtype: List[:py:class:`~briann.GUI.model_explorer.AreaStateVisualizer`]"""
        return self._state_visualizers

    def _on_right_click(self, event: tk.Event) -> None:
        """Displays a pop-up window that allow to choose a :py:class:`~briann.GUI.model_explorer.AreaStateVisualizer`.
        
        :param event: The event that triggered the function call.
        :type event: :py:class:`tkinter.Event`
        :rtype: None
        """
        
        # Create a popup
        popup = tk.Toplevel()
        popup.geometry(f"{(int)(2*DPI)}x{(int)(2*DPI)}+{self._canvas.winfo_rootx()+self._button.winfo_x()}+{self._canvas.winfo_rooty()+self._button.winfo_y()}")
        popup.overrideredirect(True) # Prevent window decorations
        
        # Add widgets to popup
        customtkinter.CTkLabel(popup, text="Add Visualizer", anchor=tk.CENTER).pack(expand=True, fill="x", padx=10, pady=10)
        
        option_values = []
        for visualizer in [StateVisualizerLineChart, StateVisualizerHeatMap]:
            if visualizer.is_compatible_with_state(input_shape=self._bpnc_area.output_time_frame_accumulator._time_frame.state.shape):
                option_values.append(visualizer.NAME_TAG)

        option_menu = customtkinter.CTkOptionMenu(popup, values=option_values)
        option_menu.pack(expand=True, fill="x", padx=10, pady=10)
        customtkinter.CTkButton(popup, text="Add", 
                                command=lambda option_menu=option_menu, popup=popup, area=self: area._add_state_visualizer(option=option_menu.get(), popup=popup)
                                ).pack(expand=True, fill="x", padx=10, pady=10)
        
        # Display popup on top of everything else
        popup.lift()
             
    def _add_state_visualizer(self, option: str, popup: customtkinter.CTkToplevel) -> None:
        """Maps the user's selection for the preferred :py:class:`~briann.GUI.model_explorer.AreaStateVisualizer` to the actual visualizer to be displayed on screen.
        
        :param option: The name of the chosen visualizer. Options are ["Line Chart"].
        :type option: str
        :param popup: The pop-up that allowed the users to make a choice. This pop-up will be deleted by the current method.
        :type popup: :py:class:`customtkinter.CTkTopLevel`
        :rtype: None
        """

        # Map selection option to visualizer
        if option == "Line Chart":
            self._state_visualizers.append(StateVisualizerLineChart(area=self._bpnc_area, canvas=self._canvas, current_simulation_time=self._briann.current_simulation_time, initial_x=self.x, initial_y=self.y, width=3, height=2))
        if option == "Heatmap":
            self._state_visualizers.append(StateVisualizerHeatMap(area=self._bpnc_area, canvas=self._canvas, current_simulation_time=self._briann.current_simulation_time, initial_x=self.x, initial_y=self.y, width=3, height=2))

        # Destroy the popup
        popup.destroy()
    
    def on_current_simulation_time_update(self, obj, name, value) -> None:
        
        # Display self as inactive
        self._button.configure(fg_color='lightgray', text_color='black')

    def on_update_count_update(self, obj, name, value) -> None:
        
        # Display self as active
        self._button.configure(fg_color='orange', text_color='white')

class Connection(DraggableWidget.LocationSubscriber):
    """A visual representation of a :py:class:`~briann.network.core.Connection`. The connection is automatically redraws itself when the area is repositioned.
    
    :param from_area: The area from which the connection starts.
    :type from_area: :py:class:`~briann.GUI.model_explorer.Area`
    :param to_area: The area at which the connection ends.
    :type to_area: :py:class:`~briann.GUI.model_explorer.Area`
    :canvas: The canvas on which the connection shall be drawn.
    :type canvas: :py:class:`~briann.GUI.model_explorer.Canvas`
    :param width: The thickness of the line representing this connection on screen.
    :type width: int, optional, default to 2.
    :param bend_by: The extend by which the connection should be bent (in inches) relative to the mid-point between the `from_area` and the `to_area`.
    :type bend_by: float, optional, defaults to 0.0.  
    """

    def __init__(self, from_area: Area, to_area: Area, canvas: Canvas, width: int = 2, bend_by: float = 0.0) -> None:
        
        # Set properties
        self._canvas = canvas
        self._from_area = from_area
        self._to_area = to_area
        self._width = width
        self._bend_by = bend_by
        
        # Draw connection
        self.draw()

        # Add self as location subscriber to areas
        from_area.add_location_subscriber(subscriber=self)
        to_area.add_location_subscriber(subscriber=self)

    def draw(self) -> None:
        """Draws self on the canvas as two segments. The first segment goes from the from-area to the mid-point with an arrow-head. The second segment goes from the mid-point to the to-area.
        
        :rtype: None"""

        # Get start end points from areas
        x0, y0 = self._from_area.x, self._from_area.y # Starting point
        x1, y1 = self._to_area.x, self._to_area.y # Endpoint
        
        # Self loop
        if x0 == x1 and y0 == y1:
            
            # Bounding box for circle
            top_left = (x0,y0)
            a = self._from_area.size # Step size for x and y from top left to bottom right
            bottom_right = (x0 + a, y0 - a)
            b = np.sqrt((a/2.0)**2+(a/2.0)**2) # length of diagonal from center of box to its corner
            tmp = np.cos(np.radians(45)) * (b + (a/2.0)) # Projection onto one of the sides of the bounding box
            arrow_position = (x0+tmp, y0-tmp)

            # Draw on screen space
            top_left = Canvas.cartesian_to_canvas(x=top_left[0], y=top_left[1])
            bottom_right = Canvas.cartesian_to_canvas(x=bottom_right[0], y=bottom_right[1])
            arrow_position = Canvas.cartesian_to_canvas(x=arrow_position[0], y=arrow_position[1])
            self._first_segment = self._canvas.create_arc(*top_left, *bottom_right, width=self._width, start=0, extent=359, style='arc')
            self._second_segment = self._canvas.create_line(arrow_position[0]-1, arrow_position[1]+1, *arrow_position, arrow='last')
            return

        # Straight line
        if self._bend_by == 0:
            # Draw on screen space
            x0, y0 = Canvas.cartesian_to_canvas(x=x0, y=y0)
            x1, y1 = Canvas.cartesian_to_canvas(x=x1, y=y1)
            xm, ym = (x0+x1)/2, (y0+y1)/2
            self._first_segment = self._canvas.create_line(x0,y0, xm,ym, width=self._width, arrow='last'); 
            self._second_segment = self._canvas.create_line(xm,ym, x1,y1, width=self._width); 
            
            return
        
        else:
            # Arc (tilted and translated ellipse segment)
            height = 2*self._bend_by
            width = np.sqrt((x1-x0)**2+(y1-y0)**2)
            
            y = lambda x: np.sqrt(((height/2)**2)*(1 - (x**2)/((width/2)**2))) # Equation of an ellipse centered at (0,0) with radii width/2 and height/2
            
            n = 20
            arc_points = np.linspace(start=-width/2, stop=width/2, num=n)
            arc_points = np.array([[x, y(x)] for x in arc_points]).T
            angle = np.arctan2(y1-y0, x1-x0) * 180 / np.pi # Angle between the two areas in degrees
            R = np.array([[np.cos(np.radians(angle)), -np.sin(np.radians(angle))], [np.sin(np.radians(angle)), np.cos(np.radians(angle))]]) # Rotation matrix
            arc_points = np.dot(R,arc_points) + np.array([[(x0+x1)/2], [(y0+y1)/2]]) # Apply rotation and translation
            
            # Draw on screen space
            arc_points = [Canvas.cartesian_to_canvas(x=arc_points[0,i], y=arc_points[1,i]) for i in range(arc_points.shape[1])]
            self._first_segment = self._canvas.create_line(*arc_points[:len(arc_points)//2+1], width=self._width, smooth=True, arrow='last')
            self._second_segment = self._canvas.create_line(*arc_points[len(arc_points)//2:], width=self._width, smooth=True)
            
    def on_location_update(self, draggable_widget: DraggableWidget) -> None:
        
        # Remove old segments
        self._canvas.delete(self._first_segment)
        self._canvas.delete(self._second_segment)
        
        # Draw new segments
        self.draw()

class NetworkVisualizer():
    """Visualizes the network structure of a given `briann` instance in a rectangle whose center is at (`initial_x`,`initial_y`) with provided `width` and `height`.
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
        :type height: float"""

    def __init__(self, briann: bnc.BrIANN, canvas: Canvas, initial_x: float, initial_y: float, width: float, height: float, area_size: float) -> None:
        
        # Set properties
        self.canvas = canvas
        self._briann = briann
        
        # Override the drag functions from super to extend it to all child widgets

        # Compute positions
        G = self.briann.get_topology()
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
            self.area_to_drawable[area] = Area(bpnc_area=area, briann=self._briann, canvas=self.canvas, x=x, y=y, size=area_size)
            
        # Draw the edges
        width = 0.05*area_size
        curved_edges = [edge for edge in G.edges() if reversed(edge) in G.edges()]
        self.edge_to_drawable = {}
        for (u,v) in G.edges():
            # Convert index to area
            u = self.briann.get_area_at_index(index=u.index)
            v = self.briann.get_area_at_index(index=v.index)
            bend_by = 0.25 if (u,v) in curved_edges else 0.0
            self.edge_to_drawable[u.index, v.index] = Connection(from_area=self.area_to_drawable[u], to_area=self.area_to_drawable[v], canvas=self.canvas, width=width, bend_by=bend_by)
          
    @property
    def briann(self) -> bnc.BrIANN:
        """
        :return: The BrIANN instance that is visualized by this visualizer.
        :rtype: bpnc.BrIANN"""
        return self._briann

class AreaStateVisualizer(DraggableWidget):
    """Superclass for a set of classes that create 2D visualizations of a :py:meth:`~briann.network.core.TimeFrame.state` on a 1x1 unit square"""

    def __init__(self, area: bnc.Area, canvas: Canvas, current_simulation_time: float, initial_x: float, initial_y: float, width: float, height: float):
    
        # Set proeprties
        self.bpnc = area

        # Create Figure
        self.figure = plt.figure(figsize=(width, height), dpi=DPI)
        
        self.figure.set_tight_layout(True)
        widget = FigureCanvasTkAgg(plt.gcf()).get_tk_widget()
        
        # Call super
        super().__init__(canvas=canvas, widget=widget, x=initial_x, y=initial_y)

        # Subscribe to area
        bpuc.CallbackManager.add_callback_to_method(target_instance=area, method_name='forward', callback=self.on_forward_call)
        bpuc.CallbackManager.add_callback_to_method(target_instance=area, method_name='reset', callback=self.on_reset_call)
        
        # Initial draw
        self._update_plot(time_frame = area.output_time_frame_accumulator.time_frame(current_time=current_simulation_time)) # Draw current time frame
        
    def on_forward_call(self, caller) -> None:
        time_frame = self.bpnc.output_time_frame_accumulator._time_frame
        self._update_plot(time_frame=time_frame)

    def on_reset_call(self, caller) -> None:
        # Reset data of self
        self.ts = np.empty(shape=[0,0])
        self.ys = np.empty(shape=[0,0])
        
        # Redraw
        time_frame = self.bpnc.output_time_frame_accumulator._time_frame
        self._update_plot(time_frame=time_frame)

    def _update_plot(self, time_frame: bnc.TimeFrame = None) -> None:
        plt.figure(self.figure.number)
        
class StateVisualizerLineChart(AreaStateVisualizer):
    """Plots a series of :py:class:`~briann.network.core.TimeFrame`'s in a line-plot where the x-axis represents time, the y-axis represents deflection and there will be one line per channel.
    This class assumes that the time-series is streamed to py:meth:`~briann.GUI.StateVisualizerLineChart._update_plot_` one time-frame at a time.
    Each such time-frame is expected to be of shape (batch-size, channel-count). """

    NAME_TAG = "Line Chart"

    def __init__(self, area: bnc.Area, canvas: Canvas, current_simulation_time: float, initial_x: float, initial_y: float, width: float, height: float) -> None:
        
        # FIRST Set Attributes (needed for initial plot)
        self.ts = np.empty(shape=[0,0])
        self.ys = np.empty(shape=[0,0])
        
        # Then Call super
        super().__init__(area=area, canvas=canvas, current_simulation_time=current_simulation_time, initial_x=initial_x, initial_y=initial_y, width=width, height=height)
        

    @classmethod
    def is_compatible_with_state(cls, input_shape: List[int]) -> bool:
        """Checks whether the given input shape is compatible with this visualizer.
        
        :param input_shape: The time-frame's state's shape to be checked.
        :type input_shape: List[int]
        :return: True if the shae is compatible with this visualizer, False otherwise.
        :rtype: bool"""
        return len(input_shape) == 2 # (batch_size, features)

    def _update_plot(self, time_frame: bnc.TimeFrame = None) -> None:
        
        # Call super
        super()._update_plot(time_frame=time_frame)

        # Update plot
        plt.clf()
                 
        if len(self.ts) == 0: self.ts = np.repeat([time_frame.time_point], time_frame.state.shape[-1])[np.newaxis,:]
        else: self.ts = np.concatenate([self.ts, np.repeat([time_frame.time_point], time_frame.state.shape[-1])[np.newaxis,:]], axis=0)
    
        if len(self.ys) == 0: self.ys = time_frame.state.cpu().detach().numpy()[0,:][np.newaxis,:]
        else: self.ys = np.concatenate([self.ys, time_frame.state.cpu().detach().numpy()[0,:][np.newaxis,:]], axis=0)  
        
        plt.plot(self.ts, self.ys)
        for channel in range(self.ys.shape[1]): plt.scatter(self.ts[:,channel], self.ys[:,channel])
        plt.xlabel("Time (s)")
        plt.ylabel("State")
        plt.title(f"Area {self.bpnc.index}")
        plt.draw()

class StateVisualizerHeatMap(AreaStateVisualizer):
    """Plots a series of :py:class:`~briann.network.core.TimeFrame`'s in an evolving heatmap where the x-axis represents width, the y-axis represents height and the heatmap represents values.
    It assumes that the input state has a shape of (batch-size, channels, height, width) where channels is 1 or 3 (grayscale or RGB) or any other number of channels which will then be averaged across.
    An alternative accepted shape is (batch-size, height, width) for single-channel data.
    The visualizer always plots only the first instance of the batch (index 0)."""

    NAME_TAG = "Heatmap"

    def __init__(self, area: bnc.Area, canvas: Canvas, current_simulation_time: float, initial_x: np.ndarray, initial_y: float, width: float, height: float) -> None:
        
        super().__init__(area=area, canvas=canvas, current_simulation_time=current_simulation_time, initial_x=initial_x, initial_y=initial_y, width=width, height=height)
        
    
    @classmethod
    def is_compatible_with_state(cls, input_shape: List[int]) -> bool:
        """Checks whether the given input shape is compatible with this visualizer.
        
        :param input_shape: The time-frame's state's shape to be checked.
        :type input_shape: List[int]
        :return: True if the shae is compatible with this visualizer, False otherwise.
        :rtype: bool"""
        return len(input_shape) == 3 or (len(input_shape) == 4) # (batch_size, width, height) or (batch_size, channels, width, height)


    def _update_plot(self, time_frame: bnc.TimeFrame = None) -> None:
        
        # Call super
        super()._update_plot(time_frame=time_frame)

        # Update plot
        plt.clf()
        
        if len(time_frame.state.shape) == 3: # (batch_size, height, width)
            plt.imshow(time_frame.state[0].cpu().detach().numpy().T, aspect='auto') # We take the first instance [0]
        else:
            if len(time_frame.state.shape) == 4: # (batch_size, channel-count, height, width)
                state = time_frame.state[0].cpu().detach().numpy()  # We take the first instance [0]
                if state.shape[1] != 1 and state.shape[0] != 3: # Not just 1 or 3 channels, average across channels
                    state = np.mean(state, axis=0, keepdims=True) # Shape now (1 channel, height, width)
                
                state = np.moveaxis(state, 0, -1) # Move channel axis to the end, shape now (height, width, channel-count)
                state = np.moveaxis(state, 0, 1) # transpose width and height, shape now (width, height, channel-count)
                plt.imshow(state, aspect='auto')
        
        plt.title(f"Area {self.bpnc.index}")
        plt.draw()
    