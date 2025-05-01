import polars as pl


# initial the sample
def return_data_sample_info(df_sig, df_con):
    
    info_data = pl.DataFrame({
        'signal sample' : [df_sig.shape[0]],
        'contiunnm sample' : [df_con.shape[0]]
    })
    
    print(info_data)

    # df_con = df_con.drop('__eventType__') if '__eventType__' in df_con.columns else None
    min_sample = min(df_sig.shape[0], df_con.shape[0])
    
    df_sample = pl.concat([
        df_sig.sample(n=min_sample, shuffle=True, seed=1),
        df_con.sample(n=min_sample, shuffle=True, seed=1),
    ])

    return df_sample


# Start training
def train(df_sample, features  = [], is_save_pkl = False, pkl_path = None, is_save_fig = None, test_size=0.1):

    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import polars as pl
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    import joblib
    import warnings
    from sklearn.metrics import log_loss, roc_curve, roc_auc_score, accuracy_score, auc
    from sklearn.model_selection import GridSearchCV
    import xgboost as xgb

    X = df_sample[features]
    y = df_sample['isSignal']

    # seperate data
    train_X, val_X, train_y, val_y = train_test_split(X , y, test_size=test_size, random_state=1)

    if pkl_path :
        import joblib
        xgb_best_model = joblib.load(pkl_path) 

    else:
        
        # train
        xgb_model = xgb.XGBClassifier(eval_metric='mlogloss', n_jobs=2)
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.1, 0.2],
            'gamma': [0, 10],
            'max_depth': [5, 6],
        }
        
        grid_search = GridSearchCV(estimator=xgb_model, 
                                   param_grid=param_grid, 
                                   cv=5, 
                                   scoring='roc_auc', 
                                   n_jobs=2, 
                                   verbose=1)
        
        grid_search.fit(train_X, train_y)
        
        xgb_best_params = grid_search.best_params_
        print("Best parameters found: ", xgb_best_params)
        
        xgb_best_model = grid_search.best_estimator_

    # d-inf
    XGBoost_scores_train = xgb_best_model.predict_proba(train_X)[:, 1]
    XGBoost_roc_auc_train = roc_auc_score(train_y, XGBoost_scores_train)
    print("Best model ROC AUC on training data:", XGBoost_roc_auc_train)
    
    XGBoost_scores_val = xgb_best_model.predict_proba(val_X)[:, 1]
    XGBoost_roc_auc_val = roc_auc_score(val_y, XGBoost_scores_val)
    print("Best model ROC AUC on testing data:", XGBoost_roc_auc_val)
    
    XGBoost_fpr_val, XGBoost_tpr_val, _ = roc_curve(val_y, XGBoost_scores_val)
    XGBoost_fpr_train, XGBoost_tpr_train, _ = roc_curve(train_y, XGBoost_scores_train)
    
    joblib.dump(xgb_best_model, is_save_pkl) if is_save_pkl else None


    # plot figure 
    plt.figure(figsize=(17, 5))
    plt.subplot(131)
    plt.plot(XGBoost_fpr_train, XGBoost_tpr_train, '--', lw=2, label='train XGBoost ROC curve (AUC = {:.3f})'.format(XGBoost_roc_auc_train))
    plt.plot(XGBoost_fpr_val, XGBoost_tpr_val, lw=2, label='validation XGBoost ROC curve (AUC = {:.3f})'.format(XGBoost_roc_auc_val))
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    plt.subplot(132)
    data = {
            'feature': features,
            'importance': xgb_best_model.feature_importances_,
        }
    
    df = pd.DataFrame(data)
    df = df.sort_values(by='importance')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout() 
    bars = plt.bar(df['feature'], df['importance'])
    plt.bar_label(bars, labels=[f'{v:.3f}' for v in df['importance']], padding=3)
    plt.xlabel('variables')
    plt.ylabel('Value')
    plt.tight_layout()
    
    plt.subplot(133)
    df_train = train_X.with_columns(train_y.alias('isSignal'))
    df_val = val_X.with_columns(val_y.alias('isSignal'))
    
    train_xgboost = xgb_best_model.predict_proba(train_X[features])[:,1]
    val_xgboost = xgb_best_model.predict_proba(val_X[features])[:,1]
    
    df_train = df_train.with_columns(pl.Series('XGBoost', train_xgboost))
    df_val = df_val.with_columns(pl.Series('XGBoost', val_xgboost))
    
    hist_kwargs = dict(bins=30, range=(0, 1), density=True, )
    train_kwargs = dict(alpha=0.6, histtype="stepfilled",)
    
    # Train
    plt.hist(df_train.filter(pl.col('isSignal')==1)['XGBoost'], label="Train signal", color='tab:blue', **hist_kwargs, **train_kwargs)
    plt.hist(df_train.filter(pl.col('isSignal')!=1)['XGBoost'], label="Train continuum", color='tab:orange', **hist_kwargs, **train_kwargs)
    
    # Test
    plt.hist(df_val.filter(pl.col('isSignal')==1)['XGBoost'], label="Test signal", color='tab:blue', histtype="step", lw=2, **hist_kwargs)
    plt.hist(df_val.filter(pl.col('isSignal')!=1)['XGBoost'], label="Test continuum", color='tab:orange', histtype="step", lw=2, **hist_kwargs)
    
    # plt.yscale("log")
    plt.xlabel(r"$C_{cs}$", loc='right', fontsize=15)
    plt.ylabel('Normalized', loc='top',fontsize=15)
    plt.legend(loc=9,fontsize=12,frameon=False)
    
    
    plt.savefig(is_save_fig) if is_save_fig else None

def apply_tpr(pkl_path, df_sample, features):

    import joblib
    model = joblib.load(pkl_path) 
    xgboost_tpr = model.predict_proba(df_sample[features])[:,1]
    df_sample = df_sample.with_columns(pl.Series("xgboost_tpr", xgboost_tpr))

    print('xgboost_tpr the features added corrected.')
    return df_sample


def tpr_plot(df_list, save_path=None, x_label='xgboost_tpr', title = 'Density applied'):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 5))
    for df in df_list:
        plt.hist(df[1]['xgboost_tpr'], histtype='step', bins=40, label=f'{df[0]}', range=(0,1), density=True)
        legend = plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1.1), framealpha=0.8, frameon=True, facecolor='white')
        frame = legend.get_frame()
        frame.set_edgecolor('black')
        plt.subplots_adjust(right=0.8, top=0.8, bottom=0.2)
        plt.tight_layout()
        
    plt.xlabel(x_label, loc='right',fontsize=13)
    plt.title(title)
    plt.savefig(save_path) if save_path else None

    return None
        
    
def get_cut_flow_table(df_list=None, table=None):

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


    data = {'Filter': ['original'] + list(table.keys()) + ['remained (%)']}
    
    for name, _ in df_list:
        data[f'# {name}'] = []

    
    for name, df in df_list:
        sel_list = []
        data[f'# {name}'].append(df.shape[0])  

        for _, sel_condition in table.items():
            sel_list.append(sel_condition)
            filtered_df = Dofilter(df, sel_list)
            data[f'# {name}'].append(filtered_df.shape[0])

    
    for name, _ in df_list:
        col_name = f'# {name}'
        original_count = data[col_name][0]
        final_count = data[col_name][-1]
        data[col_name].append(round(((final_count * 100) / original_count), 2) if original_count > 0 else 0)

    
    df = pl.DataFrame(data, strict=False)
    print(df)

    return df

def Dofilter(df, selection_list=[]):
    if selection_list is not [] :
        for sel in selection_list:
            df=df.filter(sel)
    return df 

def df_to_pdf(df, pdf_path):
    from matplotlib.backends.backend_pdf import PdfPages
    try :
        pandas_df = df.to_pandas()
    except :
        pandas_df = df
    
    html_table = pandas_df.to_html(index=False)
    pdf_path = pdf_path
    with PdfPages(pdf_path) as pdf:

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, len(pandas_df) * 0.5))  
        ax.axis("tight")
        ax.axis("off")
        table = ax.table(cellText=pandas_df.values, colLabels=pandas_df.columns, cellLoc='center', loc='center')
     
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(pandas_df.columns))))
    
        pdf.savefig(fig)
        plt.close(fig)
    
    print(f"PDF save to {pdf_path}")



if __name__ == '__main__':

    df_sig = pl.read_parquet('/group/belle/users/kuanying/b2bii/ana/signal.parquet').filter(pl.col('isSignal')==1)
    df_mixed = pl.read_parquet('/group/belle/users/kuanying/b2bii/ana/mixed.parquet')
    df_charged = pl.read_parquet('/group/belle/users/kuanying/b2bii/ana/charged.parquet')
    df_charm = pl.read_parquet('/group/belle/users/kuanying/b2bii/ana/charm.parquet')
    df_uds = pl.read_parquet('/group/belle/users/kuanying/b2bii/ana/uds.parquet')

    df_sig = df_sig.drop('__eventType__')
    df_mixed = df_mixed.drop('__eventType__')
    df_charged = df_charged.drop('__eventType__')
    df_charm = df_charm.drop('__eventType__')
    df_uds = df_uds.drop('__eventType__')

    

    selection_list = [
        # pi0 veto
        ((~(pl.col('pi0veto1').is_between(0.125, 0.145)) & ~(pl.col('pi0veto2').is_between(0.125, 0.145))))
    ]
    
    df_con = pl.concat([df_charm, df_uds])
    features = [
            'cosB', 'cosB_pz',
            'R2', 'thrustBm', 'thrustOm',
            'cosTBTO',
            'cosTBz',
            'et', 'mm2',
            'k0hso00', 'k0hso10', 'k0hso20',  'k0hso02', 'k0hso12', 'k0hso22',
            'k0hso04', 'k0hso14', 'k0hso24', 'k0hoo0', 'k0hoo1', 'k0hoo2',
            'k0hoo3', 'k0hoo4',
            # 'abs(FBDT_qrCombined)',
            'abs(qrGNN)',
        ]
    df_sample = return_data_sample_info(Dofilter(df_sig, selection_list), Dofilter(df_con, selection_list))

    is_save_fig = 'qrGNN.pdf'
    pkl_path = None
    is_save_pkl = 'xgboost.pkl'
    train(df_sample = df_sample.filter(pl.col('isSignal').is_not_null()), 
          features = features, 
          is_save_fig = is_save_fig, 
          is_save_pkl = is_save_pkl, 
          pkl_path = pkl_path)

    pkl_path = 'xgboost.pkl'
    df_sig = apply_tpr(pkl_path, df_sig, features)
    df_mixed = apply_tpr(pkl_path, df_mixed, features)
    df_charged = apply_tpr(pkl_path, df_charged, features)
    df_charm = apply_tpr(pkl_path, df_charm, features)
    df_uds = apply_tpr(pkl_path, df_uds, features)
    
    df_list = [('signal entries', df_sig), 
               ('mixed entries', df_mixed), 
               ('charged entries', df_charged), 
               ('charm entries', df_charm), 
               ('uds entries', df_uds), 
              ] 
    table = {
        'pi0veto' : (~(pl.col('pi0veto1').is_between(0.125, 0.145)) & ~(pl.col('pi0veto2').is_between(0.125, 0.145))),
        'MVA (> 0.8)' : (pl.col('xgboost_tpr') > 0.8),
    }
    tpr_plot(df_list, save_path='tpr_test.pdf')
    df_info = get_cut_flow_table(df_list=df_list, table=table) 
    df_to_pdf(df_info, pdf_path='annual/bg-cut-flow-table.pdf')

    



