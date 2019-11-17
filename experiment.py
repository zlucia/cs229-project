import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score

def run_experiment(model, test_x, test_y, torch_model=False):
    # not fully functional with torch models yet
    if torch_model:
        predictions = model(test_x)
    else:
        predictions = model.predict(test_x)
        predictions = torch.Tensor(predictions)
        test_y = torch.Tensor(test_y)

    error = torch.sqrt(F.mse_loss(predictions, test_y))  # L2 distance between predictions, ground truth
    
    error = error.item()
    if torch_model:
        predictions = np.array(predictions.detach())

    r2 = r2_score(test_y, predictions)

    stats = {'RMSE': error,
             'R2': r2,
            }

    print(stats)

    return stats, predictions