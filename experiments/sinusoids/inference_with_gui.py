"""Example of using the Briann GUI to explore a model during inference."""
from briann.GUI.model_explorer import Animator

from experiments.sinusoids import model_loader as model_loader

briann = model_loader.inference_configuration["model"]
data_iterator = model_loader.inference_configuration["data_iterator"]

app = Animator(briann=briann, data_iterator=data_iterator)
app.mainloop()
    
