import optuna

def objective(trial):
 n_estimators = trial.suggest_int("n_estimators", 10, 200)
 max_depth = trial.suggest_int("max_depth", 2, 20)
 # train your classifier with these params, return its accuracy
 ...

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)