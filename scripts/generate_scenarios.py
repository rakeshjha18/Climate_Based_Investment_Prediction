import numpy as np


def generate_scenarios(model, num_scenarios=3):
    scenarios = []
    for _ in range(num_scenarios):
        # Generate synthetic data (for example purposes)
        scenario = np.random.normal(size=(1, 28, 28, 1))
        scenarios.append(scenario)

    predictions = [model.predict(scenario) for scenario in scenarios]
    return predictions


if __name__ == "__main__":
    from models.generative_model import build_vae
    model = build_vae((28, 28, 1))
    predictions = generate_scenarios(model)
    print("Generated scenarios:", predictions)
