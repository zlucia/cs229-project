import numpy as np
import torch.nn.functional as F

def run_experiment(model, test_x, test_y):
    predictions = model(test_x)
    error = torch.sqrt(F.mse_loss(predictions, test_y))  # L2 distance between predictions, ground truth
    
    error = error.item()
    predictions = np.array(predictions.detach())
    print('error:', error)
    return error, predictions
