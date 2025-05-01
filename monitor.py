import polars as pl
import pandas as pd
import ana
import os
import importlib
import threading
import multiprocessing
import json
import argparse


importlib.reload(ana)

# os.environ["OMP_NUM_THREADS"] = "1"
# threading.stack_size(128 * 1024)
# multiprocessing.set_start_method('spawn', force=True)

Belle_I = {
        # cut-flow-path
        "cut flow path" : '/home/belle2/kuanying/rhoeta/b2bii/cut-flow.json',
        
        # data path
        "signal path" : '/group/belle/users/kuanying/b2bii/signal/merge_1.parquet',
        "mixed path" : '/group/belle/users/kuanying/b2bii/background/mixed/parquet/s0_MVA.parquet',
        "charged path" : '/group/belle/users/kuanying/b2bii/background/charged/parquet/s0_MVA.parquet',
        "charm path" : '/group/belle/users/kuanying/b2bii/background/charm/parquet/s0_MVA.parquet',
        "uds path" : '/group/belle/users/kuanying/b2bii/background/uds/parquet/s0_MVA.parquet',
        "chargedrare path" : '/group/belle/users/kuanying/b2bii/background/rareB/pq/charged/0.parquet',
        "mixedrare path" : '/group/belle/users/kuanying/b2bii/background/rareB/pq/mixed/0.parquet',
    

    
        # data generated
        "signal_generated"      : 460000,
        "mixed_generated"       : 377388636,
        "charged_generated"     : 377426351,
        "charm_generated"       : 900305275,
        "uds_generated"         : 1447127323,
        "mixedrare generated"   : 156519276,
        "chargedrare generated" : 126517884,

    
        "signal MC selection table" : {
            "rho mass <0.6, 0.9>" : (pl.col('rho_InvM_nocstr').is_between(0.6, 0.9)),
            "eta_ppp mass <0.53, 0.56>" : ((pl.col('eta_dmID') == 1) & (pl.col('eta_InvM_nocstr').is_between(0.53, 0.56)) | (pl.col('eta_dmID') == 0)),
            "eta_gg mass <0.5, 0.57>" : ((pl.col('eta_dmID') == 0) & (pl.col('eta_InvM_nocstr').is_between(0.5, 0.57)) | (pl.col('eta_dmID') == 1)),
            "Asym < 0.9" : ((pl.col('eta_dmID') == 0) & (pl.col('asym') < 0.9) | (pl.col('eta_dmID') == 1)),
            "rho+ cos_heli > -0.8" : (pl.col('rho_cosHeli') > -0.8),
            "best rank" : (pl.col('chiProb_Rank') ==1),
            'Mbc deltaE' : ((pl.col('Mbc').is_between(5.26, 5.29)) & (pl.col('deltaE').is_between(-0.4, 0.4))),
        },
    
        "signal MC selection list" : (
            (pl.col('rho_InvM_nocstr').is_between(0.6, 0.9)),
            ((pl.col('eta_dmID') == 0) & (pl.col('eta_InvM_nocstr').is_between(0.5, 0.57)) | (pl.col('eta_dmID') == 1)),
            ((pl.col('eta_dmID') == 1) & (pl.col('eta_InvM_nocstr').is_between(0.53, 0.56)) | (pl.col('eta_dmID') == 0)),
            ((pl.col('eta_dmID') == 0) & (pl.col('asym') < 0.9) | (pl.col('eta_dmID') == 1)),
            (pl.col('rho_cosHeli') > -0.8),
            (pl.col('chiProb_Rank') ==1),
            ((pl.col('Mbc').is_between(5.26, 5.29)) & (pl.col('deltaE').is_between(-0.4, 0.4)))
        ),

        "background MC selection table" : {
            "pi0 veto" : (~(pl.col('pi0veto1').is_between(0.125, 0.145)) & ~(pl.col('pi0veto2').is_between(0.125, 0.145))),
        },

        "background MC selection list" : (
            (~(pl.col('pi0veto1').is_between(0.125, 0.145)) & ~(pl.col('pi0veto2').is_between(0.125, 0.145))),
        )

        
    }



def get_cut_flow_table(df_list=None, table=None, isjson=False):

    if table is None:
        table = {}

    if df_list is None:
        df_sig, df_mixed, df_charged, df_charm, df_uds = get_df()
        df_mixedrare, df_chargedrare = get_df(israre=True)
        df_list = [
            ('signal', df_sig),
            ('mixed', df_mixed),
            ('charged', df_charged),
            ('charm', df_charm),
            ('uds', df_uds),
            ('mixedrare', df_mixedrare),
            ('chargedrare', df_chargedrare),
        ]

    if not isjson:
        
        data = {'Filter': ['original'] + list(table.keys()) + ['remained (%)']}
        for name, _ in df_list:
            data[f'# of {name}'] = []

        
        for name, df in df_list:
            sel_list = []
            data[f'# of {name}'].append(int(df.shape[0]))  

            for _, sel_condition in table.items():
                sel_list.append(sel_condition)
                filtered_df = ana.Dofilter(df, sel_list)
                data[f'# of {name}'].append(int(filtered_df.shape[0]))

        
        for name, _ in df_list:
            col_name = f'# of {name}'
            original_count = data[col_name][0]
            final_count = data[col_name][-1]
            data[col_name].append(round(((final_count * 100) / original_count), 2) if original_count > 0 else 0)

        
        df = pl.DataFrame(data, strict=False)
        df.write_json(Belle_I['cut flow path'])

    else:
        
        with open(Belle_I['cut flow path']) as file:
            data = json.load(file)
            df = pl.DataFrame(data)

    return df



def get_df(info=True, israre=False):
    if israre == False:
        df_sig = pl.read_parquet(Belle_I['signal path'])
        df_mixed = pl.read_parquet(Belle_I['mixed path'])
        df_charged = pl.read_parquet(Belle_I['charged path'])
        df_charm = pl.read_parquet(Belle_I['charm path'])
        df_uds = pl.read_parquet(Belle_I['uds path'])
        
        df_sig = df_sig.filter(Belle_I['signal MC selection list'])
        df_sig = df_sig.filter(pl.col('isSignal')==1)
        df_sig = df_sig.with_columns(
            
            # gamma gamma
            pl.when(pl.col("eta_dmID") == 0)
            .then(pl.col("__weight__") / (230000) * 771E06*9.9E-6*0.394)
            # pi pi pi0 
            .when(pl.col("eta_dmID") == 1)
            .then(pl.col("__weight__") / (230000) * 771E06*9.9E-6*0.2274)
            .alias("__weight__")
        )
    
        data = pl.DataFrame({
            '_' : ['df_sig', 'df_mixed', 'df_charged', 'df_charm', 'df_uds'],
            'entries' : [df_sig.shape[0], df_mixed.shape[0], df_charged.shape[0], df_charm.shape[0], df_uds.shape[0]],
        })
        print(data) if info else None

        
        return df_sig, df_mixed, df_charged, df_charm, df_uds

    elif israre == True:
        df_mixedrare = pl.read_parquet(Belle_I['mixedrare path'])
        df_chargedrare = pl.read_parquet(Belle_I['chargedrare path'])

        data = pl.DataFrame({
            '_' : ['df_mixedrare', 'df_chargedrare'],
            'entries' : [df_mixedrare.shape[0], df_chargedrare.shape[0]],
        })
        print(data) if info else None

        df_chargedrare = df_chargedrare.with_columns(
            __weight__ = (pl.col('__weight__') / 50)
        )
        
        df_mixedrare = df_mixedrare.with_columns(
            __weight__ = (pl.col('__weight__') / 50)
        )

        return df_mixedrare, df_chargedrare
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="increase output verbosity",
                        action="store_true")
    args = parser.parse_args()

    df_sig, df_mixed, df_charged, df_charm, df_uds = get_df(info=True)
    df = get_cut_flow_table(df_sig, df_mixed, df_charged, df_charm, df_uds, isjson = args.json)
    print(df)
        
    
        


            




