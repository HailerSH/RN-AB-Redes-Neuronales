import torch
import numpy as np
from .nn_model import NN




# Load the model once at module level
def load_model(path='predictor/best_model.pth', input_size=151, device="cpu"):
    device = device
    model = NN(input_size=input_size)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

# Single prediction (for example, from JSON or form input)
def predict(input_array: np.ndarray, model, device):
    
    
    if len(input_array.shape) == 1:
        input_array = input_array.reshape(1, -1)

    with torch.no_grad():
        input_tensor = torch.tensor(input_array.values, dtype=torch.float32).to(device)
        output = model(input_tensor)
        probabilities = torch.sigmoid(output).cpu().numpy()
        predictions = (probabilities > 0.5).astype(int)
    
    return {
        "probabilities": probabilities.tolist(),
        "predictions": predictions.tolist()
    }


