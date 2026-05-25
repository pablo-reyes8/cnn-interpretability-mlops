import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="CLI ResNet101 para entrenamiento con MLflow.")
    parser.add_argument("--config", type=str, default="config/model/oxford_pets_binary_resnet101.yaml")
    parser.add_argument("--output-dir", type=str, default="resnet101/model_trained/mlops")
    parser.add_argument("--tracking-uri", type=str, default="file:./resnet101/mlruns")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--register-model-name", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    from resnet101.src.training.train_mlflow import run_training

    run_training(args)


if __name__ == "__main__":
    main()
