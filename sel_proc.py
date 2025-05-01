import ana
import polars as pl
import importlib
import time
import argparse
import uproot
import pandas as pd
import glob
import multiprocessing
import sys
import os
import json
import monitor
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import inspect


importlib.reload(ana)
importlib.reload(monitor)

'''
/home/belle2/kuanying/modules/sel_to_parquet.py

usage : 

(e.g.)
python3 ~/modules/sel_to_parquet.py -i /group/belle/users/kuanying/b2bii/background/uds/root/s0 -o /group/belle/users/kuanying/b2bii/background/uds/parquet/s0


'''


dmID_pipipi0 = (pl.col('eta_dmID') == 1)
dmID_gamgam = (pl.col('eta_dmID') == 0)




def get_df(path, treename=None):
    
    if treename == None:
        print(uproot.open(path).keys())
    
    else:
        with uproot.open(path) as file:
            tree=file[treename]
            arrays=tree.arrays(library="pd")
            df=arrays.drop(columns = ['__eventType__']) if '__eventType__' in arrays.columns.to_list() else arrays
            df=pl.from_pandas(df)

        return df

def process_root_files(input_dir, output_dir, tree_name, filter_condition):
    """
    Processes all ROOT files in a directory, applies a filter, and writes the results to Parquet files.

    Parameters:
        input_dir (str): Directory containing the ROOT files.
        output_dir (str): Directory to save the Parquet files.
        tree_name (str): Name of the TTree in the ROOT files.
        filter_condition (pl.Expr): Polars filter condition.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all ROOT files in the input directory
    root_files = [f for f in os.listdir(input_dir) if f.endswith(".root")]

    num_file = len(root_files)

    if not root_files:
        print("No ROOT files found in the directory.")
        return

    success_count = 1
    error_list = []
    for root_file in root_files:
        input_path = os.path.join(input_dir, root_file)
        output_file = root_file.replace(".root", ".parquet")
        output_path = os.path.join(output_dir, output_file)

        try:
            # Read the ROOT file
            with uproot.open(input_path) as file:
                tree = file[tree_name]

                # Convert the TTree to a Polars DataFrame
                df = pl.DataFrame(tree.arrays(library="np"))
                df = ana.iter_apply_feature(df)
                
                # Apply the filter condition
                filtered_df = df.filter(filter_condition)

                # Write the filtered DataFrame to a Parquet file
                filtered_df.write_parquet(output_path)
                mark_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f" {mark_time} || Job {success_count} / {num_file} : Processed {root_file} -> {output_file} || Entries : {filtered_df.shape[0]}")
                success_count += 1

        except Exception as e:
            print(f"Failed to process {root_file}: {e}")
            error_list.append(f"Failed to process {root_file}: {e} \n")

    return error_list


# def process_task(args):
#     return process_single_file(*args)


# def process_single_file(input_path, output_path, tree_name, filter_condition):
#     try:
#         # 讀取 ROOT 檔案
#         with uproot.open(input_path) as file:
#             tree = file[tree_name]

#             # 轉換為 Polars DataFrame
#             df = pl.DataFrame(tree.arrays(library="np"))
#             df = ana.iter_apply_feature(df)

#             # 套用篩選條件
#             # filtered_df = df.filter(filter_condition)
#             filtered_df = df

#             # 寫入 Parquet 檔案
#             filtered_df.write_parquet(output_path)
#             mark_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             print(f"{mark_time} || Processed {os.path.basename(input_path)} -> {os.path.basename(output_path)} || Entries : {filtered_df.shape[0]}")
#             return None  # 成功時回傳 None

#     except Exception as e:
#         return f"Failed to process {os.path.basename(input_path)}: {e}"

# def process_root_files(input_dir, output_dir, tree_name, filter_condition, num_workers=4):
#     """
#     Processes all ROOT files in a directory concurrently, applies a filter, and writes the results to Parquet files.

#     Parameters:
#         input_dir (str): Directory containing the ROOT files.
#         output_dir (str): Directory to save the Parquet files.
#         tree_name (str): Name of the TTree in the ROOT files.
#         filter_condition (pl.Expr): Polars filter condition.
#         num_workers (int): Number of parallel workers.
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     root_files = [f for f in os.listdir(input_dir) if f.endswith(".root")]
#     if not root_files:
#         print("No ROOT files found in the directory.")
#         return []

#     tasks = []
#     for root_file in root_files:
#         input_path = os.path.join(input_dir, root_file)
#         output_file = root_file.replace(".root", ".parquet")
#         output_path = os.path.join(output_dir, output_file)
#         tasks.append((input_path, output_path, tree_name, filter_condition))

#     # 使用 ProcessPoolExecutor 進行並行處理
#     error_list = []
#     with ProcessPoolExecutor(max_workers=num_workers) as executor:
#         results = executor.map(process_task, tasks)

#     # 收集錯誤資訊
#     for result in results:
#         if result is not None:
#             error_list.append(result)

#     return error_list



def make_selection_to_parquet_chunk(Input, output, treename):
    
    sel_list = [

        (dmID_gamgam & pl.col('rho_InvM').is_between(0.6, 0.85) | dmID_pipipi0),
        (dmID_pipipi0 & pl.col('rho_InvM').is_between(0.6, 0.85) | dmID_gamgam ),
        (dmID_pipipi0 & pl.col('eta_InvM').is_between(0.53, 0.56) | dmID_gamgam),
        # ((pl.col('rho_pip_pionID') > 0.1)),
        # (dmID_pipipi0 & (pl.col('eta_pip_pionID') > 0.1) | dmID_gamgam),
        # (dmID_pipipi0 & (pl.col('eta_pim_pionID') > 0.1) | dmID_gamgam),
        (dmID_gamgam & (pl.col('asym') < 0.89) | dmID_pipipi0),
        (pl.col('rho_cosHeli') > -0.8),
        (pl.col('chiProb_Rank') ==1),
        ((pl.col('eta_gamma0_E') > 0.13) & (pl.col('eta_gamma1_E') > 0.13)),

        # Mbc deltaE
        ((pl.col("Mbc") > 5.26) & (pl.col("deltaE").abs() < 0.4)),
    ]

    print(f'**** read {Input} ****')
    print(f'file size : {os.path.getsize(Input) / 1000000} M')
    
    
    start=time.time()
    tuple_name = Input.split('/')[-1].replace('.root','')
    
    df_list = []
    run = 0
    loop_time=time.time()
    for i, df_chunk in enumerate(uproot.iterate(f'{Input}:{treename}', step_size=1_000, library="pd")):
        df_chunk = df_chunk.drop(columns = ['__eventType__']) 
        df_chunk = pl.from_pandas(df_chunk)
        
        df_chunk = ana.bcs_rank(df_chunk)
        df_chunk = ana.energy_asym(df_chunk)
        df_chunk = ana.cos_hel(df_chunk)
        df_chunk = ana.Dofilter(df_chunk, selection_list=sel_list)

        df_list.append(df_chunk)
        print(f"JOb {i} done,  Accumulated time : {time.time()-loop_time:.2f} seconds.")
        run += 1
            
    df = pl.concat(df_list)
    df.write_parquet(f'{output}')
    print(f"Save to {output} #Evts = {len(df)}, Spent {time.time()-start:.2f} seconds \n Have a good day.")

def tuples_to_parquet(Input, output, treename):
    
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())  # 使用所有可用的 CPU 核心數
    tasks = []

    for tuple_path in glob.glob(Input):
        tasks.append((tuple_path, output, treename))

    print(f'Info{tasks}')

    # 使用 starmap 來分發任務到進程池
    pool.starmap(make_selection_to_parquet_chunk, tasks)
    pool.close()
    pool.join()


def file_checking(input_path, output_path, log_path):
    input_list = []
    output_list = []
    
    for file in os.listdir(input_path):
        if file.endswith('.root'):
            file_name = file.replace('.root', '')
            input_list.append(file_name)

    for file in os.listdir(output_path):
        if file.endswith('.parquet'):
            file_name = file.replace('.parquet', '')
            output_list.append(file_name)


    unique_list = list(set(input_list) ^ set(output_list))
    data = pl.DataFrame({
        'type' : ['input dir', 'output dir', 'unfinished'],
        '# files' : [len(input_list), len(output_list), len(unique_list)]
    }, strict=False)

    print(data)

    response = input("Empty log file? (y/n): ").strip().lower()
    if response == 'y':
        with open(log_path, 'w') as file:
            pass

    else:
        print("Invalid input. Exiting.")
        
    response = input("Do you want to submit to the grid? (y/n): ").strip().lower()

    if response == 'y':
        success_count = 0
        fail_count = 0
        
        for file_name in unique_list:
            input_par = f'{input_path}/{file_name}.root'
            output_par = f'{output_path}/{file_name}.parquet'
            try:
                return_code = os.system(f'bsub -q s python ~/modules/sel_proc.py -i {input_par} -o {output_par} -l {log_path} -e single_proc')
                success_count += 1
                
                if return_code != 0:  
                    print(f"Error processing file: {file_name}. Stopping execution.")
                    with open(log_path, 'a') as file:
                        file.write(f'Failed | {file_name} \n')
                        fail_count += 1

            except Exception as e:
                print(f"Unexpected error occurred: {e}")
                with open(log_path, 'a') as file:
                    file.write(f'Failed | {file_name}, turn out that {e} \n')
                    fail_count += 1

        with open(log_path, 'a') as file:
            file.write(f'----------------------------------- \nTotal | Success : {success_count} | fail : {fail_count} \n')

        print(f'python ~/modules/sel_proc.py -l {log_path} -e line_check', f'save to {output_path}', sep='\n')
           
    elif response == 'n':
        print("Exiting without execution.")
    else:
        print("Invalid input. Exiting.")

    return 

def single_proc(input_path, output_path, log_path, treename, filter_condition):

    try:
        # Read the ROOT file
        with uproot.open(input_path) as file:
            tree = file[treename]

            # Convert the TTree to a Polars DataFrame
            df = pl.DataFrame(tree.arrays(library="np"))
            df = ana.iter_apply_feature(df)
            
            # Apply the filter condition
            filtered_df = df.filter(filter_condition)

            # Write the filtered DataFrame to a Parquet file
            filtered_df.write_parquet(output_path)
        print(f'Successful | {input_path} -> {output_path} | Entries | {filtered_df.shape[0]}')
        with open(log_path, 'a') as file:
            file.write(f'Successful | {input_path} -> {output_path} | Entries | {filtered_df.shape[0]} \n')

    except Exception as e:
        print(f"Failed to process {root_file}: {e}")
    
def line_check(log_path):
    try:
        with open(log_path, 'r') as file:
            for line in file:
                if line.startswith('Total'):
                    print(line.strip())
    except FileNotFoundError:
        print(f"File '{log_path}' not found.")


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Process some skimmed udsts files')
    parser.add_argument('-i', '--Input', help='input file path')
    parser.add_argument('-o', '--output', help='output file path')
    parser.add_argument('-l', '--log', help='log file path')
    parser.add_argument('-e', '--execute_func', help='execute func.')


    args = parser.parse_args()
    args_dict = {
        'input_path'            : args.Input,
        'output_path'           : args.output,
        'log_path'              : args.log,
        'treename'              : 'rhoeta',
        'filter_condition'      : monitor.Belle_I['signal MC selection list']
    }

    func_name = args.execute_func


    if func_name in globals() and callable(globals()[func_name]):
        func_signature = inspect.signature(globals()[func_name])
        func_params = func_signature.parameters.keys()
        
        filtered_args = {key: value for key, value in args_dict.items() if key in func_params}
        globals()[func_name](**filtered_args)
    


'''
usage : 
python ~/modules/sel_proc.py -i /group/belle/users/kuanying/b2bii/background/charm/root/s0 -o /group/belle/users/kuanying/b2bii/background/charm/parquet/s0 -l /group/belle/users/kuanying/b2bii/background/monitor.txt -e file_checking

python ~/modules/sel_proc.py -l /group/belle/users/kuanying/b2bii/background/monitor.txt -e line_check

'''


