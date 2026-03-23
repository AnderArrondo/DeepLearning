
from utils import *
from config import Config
from models import InsuranceModel4

def main():
    config = Config()

    print("# DATA COLLECTION")
    df: pd.DataFrame = pd.read_csv(config.csv_path)
    print(df.head())

    print("\n")

    print("# PREPROCESSING")
    df.info()
    print()
    print(df.nunique())
    print()

    categorical_cols = ["sex", "smoker", "region"]
    for category in categorical_cols:
        df[category] = pd.Categorical(df[category])

    print(df.head())
    print()
    df.info()
    print()

    if config.show_plots:
        print("# DATA VISUALIZATION")
        visualize_distributions(df)

    train_loader, test_loader, val_loader, x_scaler, y_scaler = split_data(df, batch_size=config.batch_size)
    for X, y in train_loader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    loss_fn = nn.L1Loss()
    # loss_fn = nn.MSELoss()
    # scheduler = StepLR(optimizer, step_size=20, gamma=0.005)

    if config.optimize_hyperparams:
        print("# HYPERPARAMETER OPTIMIZATION")
        sampler = optuna.samplers.TPESampler(seed=config.random_seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(make_objective(train_loader, val_loader, loss_fn), n_trials=config.val_trials) # TODO: params de make_objective

        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        top_trials = sorted(complete_trials, key=lambda t: t.value)[:config.val_trials]

        print(f"\n{"Rank":<5} | {"Trial":<6} | {"Value":<10} | {"Model":<10} | {"LR":<10} | {'Epochs':<6}")
        print("-" * 65)
        for i, trial in enumerate(top_trials):
            print(
                f"{i+1:<5} | {trial.number:<6} | {trial.value:<10.4f} | "
                f"{trial.params["model"]:<10} | {trial.params["lr"]:<10.6f} | {trial.last_step:<6}"
            )

        # BEST TRIAL SUMMARY
        best = study.best_trial
        print(f"\nBest trial: #{best.number}")
        print(f"  Model : {best.params["model"]}")
        print(f"  LR    : {best.params["lr"]:.6f}")
        print(f"  Value : {best.value:.4f}")
        print(f"  Epochs: {best.last_step}")


        model4_trials = [(t.params["lr"], t.value, t.last_step) for t in complete_trials
                 if t.params["model"] == "model4" and t.value < 0.12]

        lrs     = np.array([lr     for lr, _,   _      in model4_trials])
        losses  = np.array([val    for _,  val, _      in model4_trials])
        epochs  = np.array([steps  for _,  _,   steps  in model4_trials])

        weights = 1 / losses
        weighted_mean_lr     = np.average(lrs,    weights=weights)
        weighted_mean_epochs = np.average(epochs, weights=weights)

        print(f"LR (weighted mean):     {weighted_mean_lr:.6f}")
        print(f"Epochs (weighted mean): {weighted_mean_epochs:.1f}")

    if config.train_model:
        print("# BEST MODEL PERFORMANCE")
        best_model = config.models[config.best_model]()
        best_lr = config.lr



if __name__ == "__main__":
    main()
