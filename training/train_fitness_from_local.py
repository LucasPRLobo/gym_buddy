from pathlib import Path

from gymbuddy_engine.datasets.local_clips import load_local_clips_dataset
from training.train_fitness_model import train_fitness_model


def main():
    # Force dataset path to your videos folder
    data_dir = Path("/home/lucas-lobo/Programing/gym_buddy/videos")

    print(f"Loading dataset from {data_dir}")

    samples = load_local_clips_dataset(str(data_dir))

    if not samples:
        raise RuntimeError(
            "No samples loaded. Make sure your videos folder contains:\n"
            "  - MP4 files\n"
            "  - a labels.json with corresponding annotations\n"
        )

    # Save model inside gym_buddy/models/
    project_root = Path("/home/lucas-lobo/Programing/gym_buddy")
    model_path = project_root / "models" / "fitness_model.pkl"
    model_path.parent.mkdir(exist_ok=True)

    print(f"Training fitness model on {len(samples)} samples…")
    train_fitness_model(samples, str(model_path))
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
