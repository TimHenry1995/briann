"""Example of using the Briann GUI to explore a model during inference."""
from briann.GUI.model_explorer import Animator
from experiments.vggish import load_weights as load_weights
from experiments.vggish import model_loader as briann_loader

briann = briann_loader.inference_configuration["model"]
load_weights.load_vggish_weights_into_briann(briann, state_dict_path='experiments/vggish/vggish_state_dict.pth')

data_iterator = briann_loader.inference_configuration["data_iterator"]

app = Animator(briann=briann, data_iterator=data_iterator)
app.mainloop()
    
