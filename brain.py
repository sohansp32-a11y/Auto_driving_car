from torchinfo import summary
import torch
from pydantic import BaseModel
from fastapi import FastAPI
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import uvicorn

app = FastAPI()

class PredInput(BaseModel):
    state: list[float]

class CarModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(5, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CarModel().to(device)

def test_model():
    x = torch.randn(1, 5)
    x = x.to(device)

    start = time.perf_counter()

    for _ in range(10000):
        model(x)

    end = time.perf_counter()

    print("Total:", end - start)
    print("Per inference:", (end - start) / 10000)

def choose_action(car_state):

    state = torch.tensor(
        car_state,
        dtype=torch.float32
    )

    with torch.no_grad():
        q_values = model(state)

    action = torch.argmax(q_values).item()

    return action


@app.post("/predict")
def predict(data: PredInput):
    state = data.state

    model.eval()
    with torch.no_grad():
        q_logits = model(state)

    action = torch.argmax(q_logits, dim=1)

    return {
        "action": action,
        "state": state
    }

if __name__ == '__main__':
    uvicorn.run(
        app=app,
    )
