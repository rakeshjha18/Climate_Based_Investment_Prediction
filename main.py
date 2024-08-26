from scripts.train_models import train_models
from scripts.generate_scenarios import generate_scenarios
from models.generative_model import build_vae

if __name__ == "__main__":
    train_models()
    vae = build_vae((28, 28, 1))
    scenarios = generate_scenarios(vae)
    print("Scenarios generated:", scenarios)
