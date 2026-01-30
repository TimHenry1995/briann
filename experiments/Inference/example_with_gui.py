"""Example of using the Briann GUI to explore a model during inference."""
from briann.GUI.model_explorer import Animator

# Either one of the two example configurations shown below can be used
from experiments.configurations import example_briann as briann_loader 
#from experiments.configurations import vggish as briann_loader

briann = briann_loader.inference_configuration["model"]
data_iterator = briann_loader.inference_configuration["data_iterator"]

app = Animator(briann=briann, data_iterator=data_iterator)
app.mainloop()
    
