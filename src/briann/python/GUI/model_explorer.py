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
from src.briann.python.network import components as bnc
from src.briann.python.training import data_management as btd
import networkx as nx
from customtkinter import filedialog    
import tkinter as tk
from typing import List
from CTkMenuBar import *
import json

def get_dpi():
    # Create a hidden root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Get the DPI
    dpi = root.winfo_fpixels('1i')  # '1i' means 1 inch
    root.destroy()  # Destroy the hidden window
    return dpi

class App(customtkinter.CTk):

    INITIAL_SIDE_BAR_WIDTH = 3 # Inches
    INITIAL_CANVAS_WIDTH = 7 # Inches
    INITIAL_CANVAS_HEIGHT = 7 # Inches
    INITIAL_WIDTH = INITIAL_CANVAS_WIDTH + 2*INITIAL_SIDE_BAR_WIDTH # Inches
    INITIAL_HEIGHT = 6 # Inches
    
    @property
    def briann(self):
        return self._briann

    @briann.setter
    def briann(self, new_value):
        # Set property
        self._briann = new_value

        # GUI
        if new_value == None:
            self._selected_area = None
            self.disable_left_sidebar()
            self.disable_play_control_panel()
        else:
            self.enable_left_sidebar()
            self.enable_play_control_panel()
            self.canvas.load()
            self.canvas.plot()

    @property
    def selected_area(self):
        return self._selected_area
    
    @selected_area.setter
    def selected_area(self, new_value):
        # Ensure input validity
        if (not new_value == None) and (not isinstance(new_value, bnc.Area)): raise TypeError(f"The selected area was expected to be of type Area but was {type(new_value)}.")
        
        # Set property
        self._selected_area = new_value

        # GUI
        if new_value == None:
            self.right_side_frame.disable()
        else:
            self.right_side_frame.enable()

    def __init__(self):
        # Call super
        super().__init__()

        # Configure window
        self.title("BrIANN Explorer")
        self.dpi = get_dpi()
        self.geometry(f"{App.INITIAL_WIDTH*self.dpi}x{App.INITIAL_HEIGHT*self.dpi}")
                
        # Place Menu bar items
        menu = CTkMenuBar(master=self)
        file_button = menu.add_cascade("File")
        file_dropdown = CustomDropdownMenu(widget=file_button)
        file_dropdown.add_option(option="New", command=self.new_configuration) 
        file_dropdown.add_separator()
        file_dropdown.add_option(option="Open", command=self.open_configuration) 
        file_dropdown.add_separator()
        self.save_button = file_dropdown.add_option(option="Save", command=self.save_configuration) 
        self.save_button.configure(state="disabled")
        file_dropdown.add_separator()

        dataset_button = menu.add_cascade("Dataset")
        dataset_dropdown = CustomDropdownMenu(widget=dataset_button)
        dataset_dropdown.add_option(option="Load", command=self.select_data_directory)
        
        # Body Layout
        self.body = customtkinter.CTkFrame(master=self, width=self.INITIAL_WIDTH*self.dpi, height=self.INITIAL_HEIGHT*self.dpi, corner_radius=0)
        self.body.pack(fill="both", expand=True)
        self.body.grid_columnconfigure(0, weight=0) # Low priority to take up available space
        self.body.grid_columnconfigure(1, weight=1) # High priority to take up available space
        self.body.grid_columnconfigure(2, weight=0) # Low priority to take up available space
        self.body.grid_rowconfigure(0, weight=1)
        self.body.grid_rowconfigure(1, weight=0)

        # Place top level frames
        self.left_sidebar_frame = customtkinter.CTkFrame(self.body)
        self.left_sidebar_frame.grid(row=0, column=0, sticky="nsew")
        
        self.right_side_frame = RightSideFrame(master=self.body, app=self)
        self.right_side_frame.grid(row=0, column=2, sticky="nsew")
        
        self.canvas_frame = customtkinter.CTkFrame(self.body)
        self.canvas_frame.grid(row=0, column=1, sticky="nsew")

        self.bottom_frame = customtkinter.CTkFrame(self.body)
        self.bottom_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")

        # Place widgets in left sidebar
        #self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="BrIANN Model Explorer", font=customtkinter.CTkFont(size=20, weight="bold"))
        #self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.left_sidebar_briann_name_label = customtkinter.CTkLabel(self.left_sidebar_frame, text="BrIANN Name")
        self.left_sidebar_briann_name_label.grid(row=0, column=0, padx=20, pady=10)
        self.left_sidebar_add_area_button = customtkinter.CTkButton(self.left_sidebar_frame, text="Add Area", command=self.add_area)
        self.left_sidebar_add_area_button.grid(row=1, column=0, padx=20, pady=10)
        self.left_sidebar_add_connection_button = customtkinter.CTkButton(self.left_sidebar_frame, text="Add Connection", command=self.add_connection)
        self.left_sidebar_add_connection_button.grid(row=2, column=0, padx=20, pady=10)
        
        # Place widget in right sidebar
        
        # Place widgets in canvas frame
        self.canvas = BrIANNCanvas(app=self, master=self.canvas_frame, width=self.INITIAL_CANVAS_WIDTH*self.dpi, height=self.INITIAL_HEIGHT*self.dpi, bg='white', xscrollincrement=1,yscrollincrement=1)
        self.canvas.pack(expand=True, fill="both")
        
        # Place widgets in bottom frame. The frame uses its own grid layout internally.
        self.time_stamp_label = customtkinter.CTkLabel(self.bottom_frame, text=f"Time: 0.000s")
        self.time_stamp_label.grid(row=0, column=0, padx=20, pady=10)

        self.start_button = customtkinter.CTkButton(self.bottom_frame, state=tk.DISABLED,  text="Start", command=self.on_start_button_click)
        self.start_button.grid(row=0, column=1, padx=20, pady=10) #.grid(row=1, column=0, padx=20, pady=10)
        
        self.step_button = customtkinter.CTkButton(self.bottom_frame, state=tk.DISABLED,  text="Step", command=self.on_step_button_click)
        self.step_button.grid(row=0, column=2, padx=20, pady=10) #.grid(row=1, column=0, padx=20, pady=10)
        
        self.reset_button = customtkinter.CTkButton(self.bottom_frame, state=tk.DISABLED,  text="Reset", command=self.on_reset_button_click)
        self.reset_button.grid(row=0, column=3, padx=20, pady=10) #.grid(row=1, column=0, padx=20, pady=10)

        self.logo_label_2 = customtkinter.CTkLabel(self.bottom_frame, text=" ", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label_2.grid(row=0, column=4)

        self.bottom_frame.grid_columnconfigure((0,4), weight=1)
        self.bottom_frame.grid_columnconfigure((1,2,3), weight=0)

        # Set basic attributes of self
        self.briann_configuration = None
        self.briann_configuration_file_path = None
        self.briann = None
        self.selected_area = None
        
    def enable_left_sidebar(self):
        self.left_sidebar_briann_name_label.configure(text=self.briann.name)
        self.left_sidebar_add_area_button.configure(state='enabled')
        self.left_sidebar_add_connection_button.configure(state='enabled')

    def disable_left_sidebar(self):
        self.left_sidebar_briann_name_label.configure(text="No BrIANN Selected")
        self.left_sidebar_add_area_button.configure(state='disabled')
        self.left_sidebar_add_connection_button.configure(state='disabled')

    def enable_play_control_panel(self):
        self.start_button.configure(state=tk.NORMAL)
        self.step_button.configure(state=tk.NORMAL)
        self.reset_button.configure(state=tk.NORMAL)
      
    def disable_play_control_panel(self):
        self.time_stamp_label.configure(text="Time: 0.000s")
        self.start_button.configure(state=tk.DISABLED)
        self.step_button.configure(state=tk.DISABLED)
        self.reset_button.configure(state=tk.DISABLED)

    def add_area(self):
        if not self.briann == None:
            print("ToDo: Add area")

    def add_connection(self):
        if not self.briann == None:
            print("ToDo: Add connection")

    def new_configuration(self):
        # Load default configuration
        with open(os.path.join("src","briann","python","GUI","default configuration.json"), 'r') as file_handle:
            briann_configuration = file_handle.read() 
        self.briann = bnc.BrIANN(configuration=briann_configuration)

        # GUI
        self.save_button.configure(state="enabled")
        self.canvas.load(briann=self.briann)
        self.canvas.plot()
        self.enable_play_control_panel()

    def open_configuration(self):
        # Ask user for file path
        file_path = os.path.join("C:\\","Users","P70057764","Documents","PhD","Software Development","briann","tests","BrIANN 1.json")# filedialog.askopenfilename()#
        self.set_briann_configuration_path(new_path=file_path)
        
        # If valid, load the file
        if self.briann_configuration_file_path != None:
            try:
                with open(self.briann_configuration_file_path, "r") as json_file:
                    json_string = json_file.read()
                    self.briann = bnc.BrIANN(configuration=json_string)
            except Exception as exception:
                tk.messagebox.showinfo("showinfo", f"The file at {file_path} is not a valid configuration file for a BrIANN model.") 

    def set_briann_configuration_path(self, new_path):
        if new_path != None and new_path != "":
            self.briann_configuration_file_path = new_path
            self.save_button.configure(state="enabled")
        elif new_path == None or new_path == "":
            self.save_button.configure(state="disabled")
        else:
            self.briann_configuration_file_path = None

    def save_configuration(self):
        # If no path exists yet, ask user to choose one
        if self.briann_configuration_file_path == None:
            file_path = filedialog.askopenfilename()
            self.set_briann_configuration_path(new_path=file_path)
        
        # If path exists now, save
        if self.briann_configuration_file_path != None:
            try:
                with open(file=self.briann_configuration_file_path, mode='w') as json_file:
                    json.dump(obj=self.briann_configuration, fp=json_file, indent=4)
            except Exception as exception:
                tk.messagebox.showinfo("showinfo", exception) 

    def select_data_directory(self):
        folder_path = os.path.join("C:\\","Users","P70057764","Downloads","sequences")#filedialog.askdirectory()
        if folder_path != "":
            try:
                mnist = btd.MNIST_Sequence(batch_size=5, folder_path=folder_path, from_train=True, seed=42)
                batch_generator = mnist.batch_generator()
                self._time_frames, y = next(batch_generator)
                self.enable_play_control_panel()
            except:
                tk.messagebox.showinfo("showinfo", f"The folder at {folder_path} does not contain valid data.") 
  
    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def on_start_button_click(self):
        self.briann.start(stimuli=self._time_frames)
        print(self.briann)

    def on_step_button_click(self):
        # Update briann
        new_time_frame = self.briann.step()

        # Update gui
        self.time_stamp_label.configure(text=f"Time: {self.briann.current_simulation_time:.3f} s")
        self.canvas.plot()

        print(self.briann)

    def on_reset_button_click(self):
        print("Reset button click")
        
class RightSideFrame(customtkinter.CTkFrame):

    def __init__(self, master, app, *args, **kwargs):
        super().__init__(master, *args, **kwargs)

        # Properties
        self.app = app

        # Area Index
        self.area_index_label = customtkinter.CTkLabel(master=self, text="Area Index", anchor='w')
        self.area_index_label.grid(row=0, column=0, padx=(20,5), pady=10, stick='ew')
        self.area_index_entry = customtkinter.CTkEntry(master=self, width=0.5*app.dpi, height=0.25*app.dpi, state='disabled')
        self.area_index_entry.grid(row=0, column=1, padx=5, pady=10, stick='ew')
        self.area_index_button = customtkinter.CTkButton(master=self, width=0.5*app.dpi, height=0.25*app.dpi, text='Set', state='disabled', command=self.on_area_index_change)
        self.area_index_button.grid(row=0, column=2, padx=(5,20), pady=10, stick='ew')
        
        # Update Rate
        self.update_rate_label = customtkinter.CTkLabel(master=self, text="Update Rate:", anchor='w')
        self.update_rate_label.grid(row=1, column=0, padx=(20,5), pady=10, stick='ew')
        self.update_rate_entry = customtkinter.CTkEntry(master=self, width=0.5*app.dpi, height=0.25*app.dpi, state='disabled')
        self.update_rate_entry.grid(row=1, column=1, padx=5, pady=10, stick='ew')
        self.update_rate_button = customtkinter.CTkButton(master=self, width=0.5*app.dpi, height=0.25*app.dpi, text='Set', state='disabled', command=self.on_update_rate_change)
        self.update_rate_button.grid(row=1, column=2, padx=(5,20), pady=10, stick='ew')
        
        # Area Type
        customtkinter.CTkLabel(master=self, text="Area Type:", anchor='w').grid(row=2, column=0, padx=(20,5), pady=10)
        self.area_type_menu = customtkinter.CTkOptionMenu(master=self, values=["Source", "Regular", "Target"], command=self.set_area_type, state='disabled')
        self.area_type_menu.set("Regular")
        self.area_type_menu.grid(row=2, column=1, columnspan=2, padx=(5,20), pady=10)
        
        self.area_editor_frame = None # To be replaced by specific frame once area type is known
        
    def enable(self) -> None:
        # Assumes that self.app.selected_area exists and is not None

        # Area index
        index = self.app.selected_area.index
        self.area_index_entry.configure(state="normal")
        self.area_index_entry.delete(first_index=0, last_index="end")
        self.area_index_entry.configure(text_color="black")
        self.area_index_entry.insert(index=0, string=index)
        self.area_index_button.configure(state='normal')

        # Update rate
        update_rate = self.app.selected_area.update_rate
        self.update_rate_entry.configure(state="normal")
        self.update_rate_entry.delete(first_index=0, last_index="end")
        self.update_rate_entry.configure(text_color="black")
        self.update_rate_entry.insert(index=0, string=update_rate)
        self.update_rate_button.configure(state='normal')

        # Area type
        self.area_type_menu.configure(state="normal")
        area_type = "Regular"
        if isinstance(self.app.selected_area, bnc.Source): area_type = "Source"
        elif isinstance(self.app.selected_area, bnc.Target): area_type = "Target"
        
        self.area_type_menu.set(area_type)
        self.set_area_type(choice=area_type)

    def disable(self):
        # Area index
        self.area_index_entry.configure(state='normal')
        self.area_index_entry.delete(first_index=0, last_index='end')
        self.area_index_entry.configure(text_color="black")
        self.area_index_entry.configure(state='disabled')
        self.area_index_button.configure(state='disabled')
        
        # Update rate
        self.update_rate_entry.configure(state='normal')
        self.update_rate_entry.delete(first_index=0, last_index='end')
        self.update_rate_entry.configure(text_color="black")
        self.update_rate_entry.configure(state='disabled')
        self.update_rate_button.configure(state='disabled')

        # Area Type
        self.area_type_menu.configure(state="disabled")
        #self.right_sidebar_area_editor_frame.grid_forget()

    def on_area_index_change(self):
        # Extract index
        try:
            index = (int)(self.area_index_entry.get())
            old_index = self.app.selected_area.index
            if index != old_index and index in self.app.briann.get_area_indices(): raise Exception("Index not available for area")
            else:
                self.app.selected_area.index = index
                self.area_index_entry.configure(text_color="black")
                area_button = self.app.canvas.get_area_button(label=old_index)
                area_button.configure(text=index)
        except:
            self.area_index_entry.configure(text_color="red")
        
    def on_update_rate_change(self):
        # Extract update_rate
        update_rate = (float)(self.update_rate_entry.get())
        try:
            self.app.selected_area.update_rate = update_rate
        except:
            self.update_rate_entry.configure(text_color="red")
        
    def set_area_type(self, choice):
        if self.area_editor_frame != None:
            self.area_editor_frame.grid_forget()
        if choice == "Source":
            pass#self.area_editor_frame = SourceAreaFrame(master=self)
        else:
            pass#self.area_editor_frame - TargetAndRegularAreaFrame(master=self)

        #self.area_editor_frame.grid(row=3, column=0, columnspan=2, sticky='nsew')
                
class TargetAndRegularAreaFrame(customtkinter.CTkFrame):

    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        # Initial state
        
class SourceAreaFrame(customtkinter.CTkFrame):

    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        # Dataset type
        # Dataset frame
        
        self.right_sidebar_source_editor_frame = customtkinter.CTkFrame(self.right_sidebar_frame)
        customtkinter.CTkLabel(self.right_sidebar_source_editor_frame, text="Dataset Type:").grid(row=0, column=0, padx=20, pady=10)
        self.right_sidebar_source_dataset_type_menu = customtkinter.CTkOptionMenu(self.right_sidebar_source_editor_frame, values=["None","SequentialMNIST"], command=self.set_dataset_type)
        self.right_sidebar_source_dataset_type_menu.grid(row=0, column=1, padx=20, pady=10)

        self.right_sidebar_source_dataset_frame = MNISTSequenceEditor(self.right_side_frame)
        self.right_sidebar_source_dataset_frame.grid(row=3, column=0, columnspan=2, sticky='nsew')
        if self.selected_area.data_loader == None:
            dataset_type ="None" 
        else:
            dataset_type = str(type(self.selected_area.data_loader.dataset)).split(".")[-1].replace(">","").replace("'","")
        self.right_sidebar_source_dataset_type_menu.set(dataset_type)
        self.set_dataset_type(choice=dataset_type)
        

    def set_dataset_type(self, choice):
        if choice == "SequentialMNIST":
            self.right_sidebar_source_dataset_frame = MNISTSequenceEditor(master=self.right_sidebar_source_editor_frame)
            self.right_sidebar_source_dataset_frame.grid(row=1, column=0, columnspan=2, sticky='nsew')
     
class MNISTSequenceEditor(customtkinter.CTkFrame):

    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        customtkinter.CTkLabel(master, text="Data Folder").grid(row=1, column=0, padx=0, pady=0)
        customtkinter.CTkButton(master, text="Choose Path", command=self.load_data).grid(row=1, column=1, padx=0, pady=0)

    def load_data(self):
        file_path = filedialog.askopenfilename()

class BrIANNCanvas(tk.Canvas):

    def __init__(self, app: customtkinter.CTk, **kwargs):
        super().__init__(**kwargs)

        self.app = app
        self.dpi = get_dpi()

        self.bind("<ButtonPress-1>", self.left_mouse_press)
        self.bind("<B1-Motion>", self.drag_motion)
        self.bind("<MouseWheel>", self.zoom) # WINDOWS ONLY

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

    def load(self):

        # Create a directed graph
        self.G = nx.DiGraph()
         
        # Add nodes for each area
        for area in self.app.briann.areas:
            self.G.add_node(area.index, label=area.index)
        
        # Add edges for each connection
        edge_list = []
        for connection in self.app.briann.connections:
            edge_list.append((connection.from_area_index, connection.to_area_index, {'label': ''}))
        self.G.add_edges_from(edge_list)

    def plot(self):

        # Draw the areas
        area_positions = nx.shell_layout(self.G)
        area_size = 0.3*self.dpi
        self.area_buttons = {}
        for i, area_position in area_positions.items():
            x0, y0 = self.cartesian_to_canvas(x=area_position[0], y=area_position[1])
            color = 'orange' if self.app.briann.get_area_at_index(index= i) in self.app.briann._due_areas else 'lightgray'
            button = customtkinter.CTkButton(self, text = i, fg_color=color, border_color='lightgray', border_width=0.05*area_size, text_color='black', command = lambda button_identifier=f"id{i}": self.area_click(button_identifier=button_identifier), anchor = tk.W, width = area_size, height=area_size, corner_radius=0.5*area_size)
            self.area_buttons[f"id{i}"] = button
            self.create_window(x0, y0, anchor=tk.CENTER, window=button)
            
        # Draw the edges
        width = 0.05*area_size
        curved_edges = [edge for edge in self.G.edges() if reversed(edge) in self.G.edges()]
        straight_edges = list(set(self.G.edges()) - set(curved_edges))
        for (u,v) in straight_edges:
            x0, y0 = area_positions[u][0], area_positions[u][1] # Starting point
            x1, y1 = area_positions[v][0], area_positions[v][1] # Endpoint
            d01 = (x1-x0, y1-y0)
            xm, ym = x0+0.5*d01[0], y0+0.5*d01[1] # Mid-point

            # Convert to screen space
            x0, y0 = self.cartesian_to_canvas(x=x0, y=y0)
            xm, ym = self.cartesian_to_canvas(x=xm, y=ym)
            x1, y1 = self.cartesian_to_canvas(x=x1, y=y1)
            self.create_line(x0,y0,xm,ym, arrow='last', width=width); self.create_line(x1,y1,xm,ym, width=width)
        
        for (u,v) in curved_edges:
            # Get start end points from area as well as midpoint
            x0, y0 = area_positions[u][0], area_positions[u][1] # Starting point
            x1, y1 = area_positions[v][0], area_positions[v][1] # Endpoint
            d01 = (x1-x0, y1-y0)
            xm, ym = x0+0.5*d01[0], y0+0.5*d01[1] # Mid-point
            
            # Compute bend points orthogonal to midpoint
            d_orthogonal = (1, -d01[0]/d01[1])
            d_orthogonal_len = np.sqrt(d_orthogonal[0]**2+d_orthogonal[1]**2)
            d_orthogonal = (d_orthogonal[0]/d_orthogonal_len, d_orthogonal[1]/d_orthogonal_len)
            xmf, ymf = xm+0.1*d_orthogonal[0], ym+0.1*d_orthogonal[1] # Bend point for forward pointing arrow
            xmb, ymb = xm-0.1*d_orthogonal[0], ym-0.1*d_orthogonal[1] # Bend point for backward pointing arrow
            
            # Draw on screen space
            x0, y0 = self.cartesian_to_canvas(x=x0, y=y0)
            xmf, ymf = self.cartesian_to_canvas(x=xmf, y=ymf)
            xmb, ymb = self.cartesian_to_canvas(x=xmb, y=ymb)
            x1, y1 = self.cartesian_to_canvas(x=x1, y=y1)
            if u < v: 
                self.create_line(x0,y0, xmf,ymf, width=width, arrow='last'); self.create_line(xmf,ymf, x1,y1, width=width)
            else: 
                self.create_line(x0,y0, xmb,ymb, width=width, arrow='last'); self.create_line(xmb,ymb, x1,y1, width=width)
    
    def area_click(self, button_identifier: str):
        
        button = self.area_buttons[button_identifier]
        index = button.cget('text')
        self.app.selected_area = self.app.briann.get_area_at_index(index=index)

    def cartesian_to_canvas(self, x, y):
        return x*self.dpi+0.5*self.winfo_reqwidth(), 0.5*self.winfo_reqheight()-y*self.dpi

    def canvas_to_cartesian(self, x, y):
        return (x-0.5*self.winfo_reqwidth())/self.dpi, -(y-0.5*self.winfo_reqheight())/self.dpi

class StateVisualizer():
    """Superclass for a set of classes that create 2D visualizations of a :py:meth:`.TimeFrame.state` on a 1x1 unit square"""

    def __init__(self):
        pass

    def visualize(self, axes, x, y):
        pass
       

if __name__ == "__main__":
    app = App()
    app.mainloop()