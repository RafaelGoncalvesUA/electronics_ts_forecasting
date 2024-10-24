import os
from itertools import product
from time import perf_counter
from dotenv import load_dotenv
import requests

# remove files generated by previous runs
if os.path.exists("models"):
    os.system("rm -rf models")

os.system("mkdir models")

if os.path.exists("out.csv"):
    os.remove("out.csv")

if os.path.exists("ignore.csv"):
    os.remove("ignore.csv")


header = "var,input,model,args,lookback,forecast,mse,train_time,inf_time\n"
with open("out.csv", "w") as f:
    f.write(header)

NUM_BUILDINGS = 6

load_dotenv()


def send_pushover_message(title, message):
    print("Sending pushover message:", message)
    url = "https://api.pushover.net/1/messages.json"

    data = {
        "token": os.getenv("PUSHOVER_TOKEN"),
        "user": os.getenv("PUSHOVER_USER"),
        "device": os.getenv("PUSHOVER_DEVICE"),
        "title": title,
        "message": message
    }

    response = requests.post(url, data=data)
    print(response.text)


def custom_error_handler(e):
    print(f"Error: {e}")
    exit(1)


def run_job(data_path, results_path, model, lb, fc, red, is_global):
    print(f"Running {model} with lookback {lb}, forecast {fc}, and reduction {red}")
    
    red_method = f"-r {red}" if red else ""
    model_args = f"-a {model[1]}" if len(model) > 1 else ""
    is_global_flag = "-g " if is_global else ""
    
    command = f"python3 run.py -d data/preprocessed/{data_path[0]} -t data/preprocessed/{data_path[1]} {is_global_flag}-f {data_path[2]} "
    command += f"-o {results_path} -m {model[0]} -l {lb} -w {fc} {model_args} {red_method}"
    
    print(command)
    return os.system(command)


total_start = perf_counter()

MIN_COUNTER = -1 # start from experiment 0

for mode in [("global", 0), ("global", -1), ("global", 1), ("local", 0)]:
    msg = f"Training {mode[0].capitalize()} model with mode {mode[1]}"
    print(msg)
    send_pushover_message("New benchmark", msg)

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

    models =  []
    
    dropout1 = [0.35] # TODO: add more
    dropout2 = [0.3] # TODO: add more
    ann_args = list(product(dropout1, dropout2))
    models += [("ann", f"{conf[0]},{conf[1]}") for conf in ann_args]

    lstm_units = [64]
    dropout = [0.1] # TODO: add 0.3 later
    recurrent_dropout = [0.1] # TODO: add 0.3 later
    lstm_args = list(product(lstm_units, dropout, recurrent_dropout))
    models += [("lstm", f"{conf[0]},{conf[1]},{conf[2]}") for conf in lstm_args]

    cnn_units = [64]
    models += [("cnn", conf) for conf in cnn_units]

    lookback = [24]
    forecast = [3]
    reduction = [None, "pca", "vae"] # add vae later

    jobs = list(product(datasets, models, lookback, forecast, reduction))
    
    ctr = 0
    for data_path, model, lb, fc, red in jobs:
        if ctr % 20 == 0:
            with open("logs.txt", "a") as f:
                f.write(f"{mode[0]}_{mode[1]}: {ctr} / {len(jobs)} jobs\n")

            # read all lines from the csv file
            with open("out.csv", "r") as f:
                body = f.readlines()[-1]
                send_pushover_message(f"Job {ctr}/{len(jobs)} [{mode[0]}_{mode[1]}]", body)

        if ctr < MIN_COUNTER:
            ctr += 1
            continue

        if run_job(data_path, out_file, model, lb, fc, red, is_global):
            print(f"Error running job {ctr}")
            exit(1)

        ctr += 1

    # write to logs.txt
    with open("logs.txt", "a") as f:
        f.write(f"{mode[0]}_{mode[1]}: {ctr} jobs completed\n")

    end = perf_counter()
    print(f"Elapsed time: {end - start:.2f} seconds")

total_end = perf_counter()
print(f"Total elapsed time: {total_end - total_start:.2f} seconds")
send_pushover_message("DONE!", "Benchmark completed.")
