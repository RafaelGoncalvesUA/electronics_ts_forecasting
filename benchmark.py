import multiprocessing
from joblib import Parallel, delayed
import os
from itertools import product
from time import perf_counter

# remove files generated by previous runs
if os.path.exists("models"):
    os.system("rm -rf models")

os.system("mkdir models")

if os.path.exists("out.csv"):
    os.remove("out.csv")

if os.path.exists("ignore.csv"):
    os.remove("ignore.csv")


header = "var,input,model,lookback,forecast,mse,timestamp\n"
with open("out.csv", "w") as f:
    f.write(header)

NUM_BUILDINGS = 6


def run_job(data_path, results_path, model, lb, fc, red, is_global):
    print(f"Running {model} with lookback {lb}, forecast {fc}, and reduction {red}")
    
    red_method = f"-r {red}" if red else ""
    is_global_flag = "-g " if is_global else ""
    
    command = f"python3 run.py -d data/preprocessed/{data_path[0]} -t data/preprocessed/{data_path[1]} {is_global_flag}-f {data_path[2]} "
    command += f"-o {results_path} -m {model} -l {lb} -w {fc} {red_method}"
    
    print(command)
    os.system(command)

total_start = perf_counter()


for mode in [("global", 0), ("global", -1), ("global", 1), ("local", 0)]:
    print(f"Training {mode[0].capitalize()} model with mode {mode[1]}")

    start = perf_counter()

    if mode == ("global", 0):
        datasets = [("", "b1_test.csv", 0)] # train global model from scratch (only once)
        out_file = "ignore.csv" # do not test
        is_global = True

    elif mode == ("local", 0):
        datasets = [(f"b{b}_local.csv", f"b{b}_test.csv", 0) for b in range(1, NUM_BUILDINGS+1)] # train local model from scratch
        out_file = "out.csv"
        is_global = False
    
    elif mode[1] == -1:
        datasets = [("", f"b{b}_test.csv", -1) for b in range(1, NUM_BUILDINGS+1)] # test global model for all buildings
        out_file = "out.csv"
        is_global = True

    elif mode[1] == 1:
        datasets = [(f"b{b}_local.csv", f"b{b}_test.csv", 1) for b in range(1, NUM_BUILDINGS+1)] # fine tune global model for all buildings
        out_file = "out.csv"
        is_global = False # because we are fine tuning the global model for each building

    models = ["ann"]
    lookback = [48]
    forecast = [1]
    reduction = ["vae"]

    # get number of available logical CPUs
    num_cpus = multiprocessing.cpu_count()
    
    Parallel(n_jobs=num_cpus)(
        delayed(run_job)(data, out_file, model, lb, fc, red, is_global)
        for data, model, lb, fc, red in product(datasets, models, lookback, forecast, reduction)
    )

    end = perf_counter()
    print(f"Elapsed time: {end - start:.2f} seconds")

total_end = perf_counter()
print(f"Total elapsed time: {total_end - total_start:.2f} seconds")