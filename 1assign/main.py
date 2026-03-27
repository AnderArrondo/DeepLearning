
from utils import *
from config import Config
from models import InsuranceModel4
from datetime import datetime
from optuna.visualization import plot_param_importances

import torch.optim as optim

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

    # loss_fn = nn.MSELoss()
    # scheduler = StepLR(optimizer, step_size=20, gamma=0.005)
    config.custom_name=input("Introduce a name for the save(Empty for default):")
    if config.custom_name=="":
        config.model_path=f"models/best_model_({config.best_model}_{config.timestamp}).pth"
    else:
        config.model_path=f"models/{config.custom_name}.pth"
        config.writer=SummaryWriter(f"runs/insurance_best/{config.custom_name}")


    if config.optimize_hyperparams:
        print("# HYPERPARAMETER OPTIMIZATION")
        sampler = optuna.samplers.TPESampler(seed=config.random_seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(make_objective(train_loader, val_loader, config.loss_fn), n_trials=config.val_trials) # TODO: params de make_objective

        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        top_trials = sorted(complete_trials, key=lambda t: t.value)[:config.val_trials]

        print(f"\n{"Rank":<5} | {"Trial":<6} | {"Value":<10} | {"Model":<10} | {"LR":<10} | {'Epochs':<6}")
        print("-" * 65)
        
        ranking=[]
        trial_num=[]
        trial_val=[]
        trial_model=[]
        trial_lr=[]
        for i, trial in enumerate(top_trials):
            ranking.append(i+1)
            trial_num.append(trial.number)
            trial_val.append(trial.value)
            trial_model.append(trial.params["model"])
            trial_lr.append(trial.params["lr"])
            print(
                f"{i+1:<5} | {trial.number:<6} | {trial.value:<10.4f} | "
                f"{trial.params["model"]:<10} | {trial.params["lr"]:<10.6f} | {trial.last_step:<6}"
            )
        top_trials_df=pd.DataFrame({
            "ranking":ranking,
            "trial_num":trial_num,
            "trial_val":trial_val,
            "trial_model": trial_model,
            "trial_lr":trial_lr
        })
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
        config.top_trials_df=top_trials_df

        try:
            fig = plot_param_importances(study)
            fig.write_image("param_importance_plot.png")
        except Exception as e:
            pass

    top_trials_df["loss_inverse"] = 1/top_trials_df["trial_val"]

    #DEFINE BEST MODEL
    model_ranking=config.top_trials_df\
        .groupby("trial_model")\
        .apply(weighted_stats)\
        .sort_values(by=["avg_loss"])
    #print(model_ranking)

    config.best_model=model_ranking["model"].iloc[0]
    config.best_lr=model_ranking["avg_lr"].iloc[0]
    
    if config.train_model:
        print("# BEST MODEL PERFORMANCE")
        best_model: nn.Module = config.models[config.best_model]()#Returns model
        best_model.to(config.device)
        best_lr = config.best_lr
        #print(f"MODEL:{best_model}||| LR:{best_lr}")
        best_model_optimizer=optim.Adam(best_model.parameters(), lr=best_lr)
        train(train_loader,
              best_model,
              config.loss_fn,
              best_model_optimizer,
              config.device,
              config.writer,
              config.epochs)
        
        checkpoint={
            "model_key": config.best_model,
            "model": best_model.state_dict(),
            "lr":config.best_lr
        }
        torch.save(checkpoint, config.model_path)
    
    if config.test_model:
        model_key, model_data, _ = load_model(config.model_path)
        test_model: nn.Module=config.models[model_key]()
        test_model.load_state_dict(model_data)
        test_model.to(config.device)

        test(test_loader,
            test_model,
            config.loss_fn,
            y_scaler,
            config.device,
            config.writer,
            config.epochs)

# probar otros optimizers o buscar parametros de adam
# parameter importance optuna
# paralelizacion optuna
# buscar pruner


if __name__ == "__main__":
    main()
