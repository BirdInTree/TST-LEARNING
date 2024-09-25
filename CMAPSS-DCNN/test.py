import torch
import math
torch.manual_seed(0)
pred = torch.randint(0, 10, (100,1))
true = torch.randint(0, 10, (100,1))

print(pred[1])
print(true[1])

def score(y_pred, y_true):
    ds = y_pred - y_true
    results = sum([ (torch.exp(-d/13)-1).item()  if d < 0 else (torch.exp(d/10)-1).item() for d in ds])
    return results

score= score(pred, true)
print(score,type(score))