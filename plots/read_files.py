import csv

def read_log_file(path):
    with open(path, 'r') as fl:
        epi_steps=[]
        returns=[]
        while True:
            line = fl.readline()
            if not line:
                break
            dc = eval(line)
            rw = dc['return']
            stp = dc['eval_step']
            epi_steps.append(int(stp))
            returns.append(float(rw))
    return epi_steps, returns

def read_csv_file(path):
    total_steps = []
    mean_returns = []
    
    with open(path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            total_steps.append(int(row['total_steps']))
            mean_returns.append(float(row['mean_return']))
    
    return total_steps, mean_returns

def get_total_run_time(path, type):
    if type == 'log':
        with open(path, 'r') as fl: 
            while True:
                line = fl.readline()
                if not line:
                    break
                dc = eval(line)
        total_time = float(dc['elapsed_time'])
    elif type == 'csv':
        with open(path, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                line = row
        total_time = float(line['elapsed_time'])
    return total_time
