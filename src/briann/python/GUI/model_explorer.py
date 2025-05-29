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


class App(customtkinter.CTk):

    INITIAL_SIDE_BAR_WIDTH = 3 # Inches
    INITIAL_PLOT_WIDTH = 7 # Inches
    INITIAL_WIDTH = INITIAL_PLOT_WIDTH + INITIAL_SIDE_BAR_WIDTH # Inches
    INITIAL_HEIGHT = 5 # Inches
    
    def get_dpi(self):
        # Create a hidden root window
        root = tk.Tk()
        root.withdraw()  # Hide the root window

        # Get the DPI
        dpi = root.winfo_fpixels('1i')  # '1i' means 1 inch
        root.destroy()  # Destroy the hidden window
        return dpi

    def __init__(self):
        super().__init__()

        # Configure data
        self._briann = None
        self._time_frames = None
        
        # Configure window
        dpi = self.get_dpi()

        self.title("BrIANN Model Explorer")
        self.geometry(f"{App.INITIAL_WIDTH*dpi}x{App.INITIAL_HEIGHT*dpi}")
        
        # Layout
        self.grid_columnconfigure(0, weight=0) # Low priority to take up available space
        self.grid_columnconfigure(1, weight=1) # High priority to take up available space
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)

        # Place top level frames
        self.sidebar_frame = customtkinter.CTkFrame(self, width=self.INITIAL_SIDE_BAR_WIDTH*dpi, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        
        self.bottom_frame = customtkinter.CTkFrame(self, width=self.INITIAL_WIDTH*dpi, corner_radius=0)
        self.bottom_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")

        self.content_frame = customtkinter.CTkFrame(self, width=App.INITIAL_PLOT_WIDTH*dpi, corner_radius=0)
        self.content_frame.grid(row=0, column=1, sticky="nsew")
        
        # Place widgets in sidebar
        #self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="BrIANN Model Explorer", font=customtkinter.CTkFont(size=20, weight="bold"))
        #self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sidebar_configuration_button = customtkinter.CTkButton(self.sidebar_frame, text="Open Configuration", command=self.select_configuration_file)
        self.sidebar_configuration_button.grid(row=0, column=0, padx=20, pady=10)
        self.sidebar_data_button = customtkinter.CTkButton(self.sidebar_frame, text="Open Dataset", command=self.select_data_directory)
        self.sidebar_data_button.grid(row=1, column=0, padx=20, pady=10)
        
        # Place widgets in plot frame
        self.plot = Plot(master=self.content_frame, data_object=self)

        # Place widgets in bottom frame. The frame uses its own grid layout internally.
        self.time_stamp_label = customtkinter.CTkLabel(self.bottom_frame, text=f"Time: 0", font=customtkinter.CTkFont(size=20, weight="normal"))
        self.time_stamp_label.grid(row=0, column=0, padx=20, pady=10)

        self.start_button = customtkinter.CTkButton(self.bottom_frame, state=tk.DISABLED,  text="Start", command=self.on_start_button_click)
        self.start_button.grid(row=0, column=1, padx=20, pady=10) #.grid(row=1, column=0, padx=20, pady=10)
        
        self.step_button = customtkinter.CTkButton(self.bottom_frame, state=tk.DISABLED,  text="Step", command=self.on_step_button_click)
        self.step_button.grid(row=0, column=2, padx=20, pady=10) #.grid(row=1, column=0, padx=20, pady=10)
        
        self.reset_button = customtkinter.CTkButton(self.bottom_frame, state=tk.DISABLED,  text="Reset", command=self.on_reset_button_click)
        self.reset_button.grid(row=0, column=3, padx=20, pady=10) #.grid(row=1, column=0, padx=20, pady=10)

        self.logo_label_2 = customtkinter.CTkLabel(self.bottom_frame, text="CustomTkinter", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label_2.grid(row=0, column=4)

        self.bottom_frame.grid_columnconfigure((0,4), weight=1)
        self.bottom_frame.grid_columnconfigure((1,2,3), weight=0)
        
        """
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, command=self.sidebar_button_event)
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=3, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=4, column=0, padx=20, pady=(10, 10))
        
        """
        
    def select_configuration_file(self):
        file_path = os.path.join("C:\\","Users","P70057764","Documents","PhD","Software Development","briann","tests","briann 1.json")# filedialog.askopenfilename()
        if file_path != "":
            try:
                self._briann = bnc.BrIANN(batch_size=1, configuration_file_path=file_path)
                self.plot.plot_graph()
                self.enable_play_control_panel()
            except:
                tk.messagebox.showinfo("showinfo", f"The file at {file_path} is not a valid configuraiton file for a BrIANN model.") 
        
    def select_data_directory(self):
        folder_path = os.path.join("C:\\","Users","P70057764","Downloads","sequences")#filedialog.askdirectory()
        if folder_path != "":
            try:
                mnist = btd.MNIST_Sequence(batch_size=1, folder_path=folder_path, from_train=True, seed=42)
                batch_generator = mnist.batch_generator()
                self._time_frames, y = next(batch_generator)
                self.enable_play_control_panel()
            except:
                tk.messagebox.showinfo("showinfo", f"The folder at {folder_path} does not contain valid data.") 
                
    def enable_play_control_panel(self):
        if self._briann != None and self._time_frames != None:
            self.start_button.configure(state=tk.NORMAL)
            self.step_button.configure(state=tk.NORMAL)
            self.reset_button.configure(state=tk.NORMAL)
        else:
            self.start_button.configure(state=tk.DISABLED)
            self.step_button.configure(state=tk.DISABLED)
            self.reset_button.configure(state=tk.DISABLED)

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def on_start_button_click(self):
        self._briann.start(stimuli=self._time_frames)

    def on_step_button_click(self):
        self._briann.step()

    def on_reset_button_click(self):
        print("Reset button click")

    def on_next_time_frame_button_click(self):
        self.time_stamp_label.setvar(name="text", value=self.briann.simulation_time)

class Plot():
    
    
    def __init__(self, master, data_object):
        self.data_object = data_object
        
        self.fig = Figure(figsize = (App.INITIAL_PLOT_WIDTH, App.INITIAL_HEIGHT)) 
        self.axes = self.fig.add_subplot()
        self.axes.set_axis_off()
        # creating the Tkinter canvas containing the Matplotlib figure 
        self.canvas = FigureCanvasTkAgg(self.fig, master = master) 
        self.canvas.draw() 
        self.canvas.get_tk_widget().pack(expand=True, fill='both') 
    
        # creating the Matplotlib toolbar 
        toolbar = NavigationToolbar2Tk(self.canvas, master) 
        toolbar.update()  
        toolbar.pack(expand=False, fill='x')


    def plot_graph(self):
        # Plotting the graph 
        self.data_object._briann.plot_graph(axes=self.axes)
        self.canvas.draw()
        
        

if __name__ == "__main__":
    app = App()
    app.mainloop()