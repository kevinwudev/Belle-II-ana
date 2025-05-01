import matplotlib.pyplot as plt
import uproot
import pandas as pd
import numpy as np
import sys
import polars as pl
from tabulate import tabulate
import vector
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
import joblib
import warnings
# from PyFastBDT import FastBDT
from sklearn.metrics import log_loss
from sklearn.metrics import roc_curve, roc_auc_score
# from sklearn.metrics import roc_curve, roc_auc_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn import metrics
import joblib
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
import monitor
import importlib as lb

lb.reload(sys.modules['monitor'])



'''
Applying some features, eg. bcs-rank, energy_asym, kst_veto

'''

feature_list : list =[]

def registry_deco(func):
    feature_list.append(func)
    return func


@registry_deco 
def bcs_rank(df : pl.DataFrame, isSel=None) -> pl.DataFrame: 
    '''Setting best selection rank'''
    '''Require : 'bu_chiProb' '''
    
    df = df.with_columns((df['__event__'] != df['__event__'].shift(1)).cum_sum().alias('group'))
    df = df.with_columns(pl.col('bu_chiProb').rank(descending=True).over('group').alias('chiProb_Rank'))
    if isSel == True:
        df = df.filter(pl.col('chiProb_Rank') == 1)
    df = df.drop('group')
    return df

@registry_deco
def energy_asym(df : pl.DataFrame) -> pl.DataFrame: 
    '''Setting energy asymmetric of two photons from eta'''
    '''Require : 'eta_gamma0_E', 'eta_gamma1_E'  '''
    
    formula=abs(pl.col("eta_gamma0_E")-pl.col("eta_gamma1_E"))/(pl.col("eta_gamma0_E")+pl.col("eta_gamma1_E"))
    df=df.with_columns(formula.alias("asym"))
    return df


def kst_veto(df : pl.DataFrame) -> pl.DataFrame: 
    '''Setting K_s veto mass windows with fake tracks by creating two duaghters tracks of pi and K'''
    '''Require : 'rho_pip_px', 'rho_pip_py', 'rho_pip_pz'  '''
    
    df=df.to_pandas()
    rho0_pip00 = vector.arr({"px":df['rho_pip_px'], "py": df['rho_pip_py'], "pz":df['rho_pip_pz'], "mass" : np.full(len(df), 0.13957)}) #pi
    rho0_pip01 = vector.arr({"px":df['rho_pip_px'], "py": df['rho_pip_py'], "pz":df['rho_pip_pz'], "mass" : np.full(len(df), 0.493677)}) #K
    rho0_pim10 = vector.arr({"px":df['rho_pim_px'], "py": df['rho_pim_py'], "pz":df['rho_pim_pz'], "mass" : np.full(len(df), 0.13957)}) #pi
    rho0_pim11 = vector.arr({"px":df['rho_pim_px'], "py": df['rho_pim_py'], "pz":df['rho_pim_pz'], "mass" : np.full(len(df), 0.49368)}) #K
    
    rho_in_fact_kst_0=rho0_pip00+rho0_pim11
    rho_in_fact_kst_1=rho0_pip01+rho0_pim10    
    df['kst0_veto_0'] = rho_in_fact_kst_0.M
    df['kst0_veto_1'] = rho_in_fact_kst_1.M
    df=pl.from_pandas(df)
    return df

@registry_deco
def cos_hel(df : pl.DataFrame, particle='rho', particle_mother='bu', particle_daugther='rho_pip') -> pl.DataFrame: 
    '''Setting the cosine helicity angle in rho's rest frame with mother particle B+ and daughter particles pi'''
    '''Require  : 'rho_px', 'rho_py', 'rho_pz', 'bu_px', 'bu_py', 'bu_pz',  'rho_pip_px', 'rho_pip_py', 'rho_pip_pz' '''
    
    df=df.to_pandas()
    par_vec = vector.arr({"px":df[f'{particle}_px'], 
                          "py": df[f'{particle}_py'], 
                          "pz":df[f'{particle}_pz'], 
                          "E" : df[f'{particle}_E']})
    par_mother_vec = vector.arr({"px":df[f'{particle_mother}_px'], 
                                 "py": df[f'{particle_mother}_py'], 
                                 "pz":df[f'{particle_mother}_pz'], 
                                 "E" : df[f'{particle_mother}_E']})
    par_daugther_vec = vector.arr({"px":df[f'{particle_daugther}_px'], 
                                   "py": df[f'{particle_daugther}_py'], 
                                   "pz":df[f'{particle_daugther}_pz'], 
                                   "E" : df[f'{particle_daugther}_E']})
    par_m_vec_in_par_vec = par_mother_vec.boost(-par_vec.to_beta3())
    par_dau_vec_in_par_vec = par_daugther_vec.boost(-par_vec.to_beta3())
    cos_theta = par_dau_vec_in_par_vec.to_Vector3D().dot(par_m_vec_in_par_vec.to_Vector3D())/(par_dau_vec_in_par_vec.mag*par_m_vec_in_par_vec.mag)
    df[f'{particle}_cosHeli'] = cos_theta
    df=pl.from_pandas(df)
    return df 
    
@registry_deco
def cosB(df):
    df = df.to_pandas()
    df['cosB'] = (df['bu_CMS_px']*df['beamPx']+df['bu_CMS_py']*df['beamPy']+df['bu_CMS_pz']*df['beamPz'])/(df['bu_CMS_p']*np.sqrt(df['beamPx']*df['beamPx']+df['beamPy']*df['beamPy']+df['beamPz']*df['beamPz']))
    df['cosB_pz'] = (df['bu_CMS_pz'])/(df['bu_CMS_p'])
    df = pl.from_pandas(df)
    return df
    

def get_feature_list():
    return feature_list

def iter_apply_feature(df : pl.DataFrame, func_list : list=feature_list) -> pl.DataFrame:
    for func in func_list:
        try:
            df = func(df)  
        except Exception as e:
            print(f"Error in feature {func.__name__}: {e}, pass through it")
    return df



def Result(df):
    Mode1_df=df.filter((pl.col("jpsi_dmID") == 0) & (pl.col("etap_dmID") == 0))
    Mode2_df=df.filter((pl.col("jpsi_dmID") == 0) & (pl.col("etap_dmID") == 1))
    Mode3_df=df.filter((pl.col("jpsi_dmID") == 1) & (pl.col("etap_dmID") == 0))
    Mode4_df=df.filter((pl.col("jpsi_dmID") == 1) & (pl.col("etap_dmID") == 1))

    number=100000*0.5

    # data
    data = {
        '_': ["jpsi:ee etap:etapipi", "jpsi:ee etap:rhogamma", "jpsi:mumu etap:etapipi", "jpsi:mumu etap:rhogamma","All" ],
        'Generated Event': [number*0.6 , number*0.6, number*0.4, number*0.4, 100000],
        'Signal' : [len(Mode1_df.filter((pl.col("isSignal") == 1))["Mbc"]),
                    len(Mode2_df.filter((pl.col("isSignal") == 1))["Mbc"]),
                    len(Mode3_df.filter((pl.col("isSignal") == 1))["Mbc"]),
                    len(Mode4_df.filter((pl.col("isSignal") == 1))["Mbc"]),
                    len(df.filter((pl.col("isSignal") == 1))["Mbc"])],
        'Scf': [len(Mode1_df.filter((pl.col("isSignal") != 1))["Mbc"]),
                    len(Mode2_df.filter((pl.col("isSignal") != 1))["Mbc"]),
                    len(Mode3_df.filter((pl.col("isSignal") != 1))["Mbc"]),
                    len(Mode4_df.filter((pl.col("isSignal") != 1))["Mbc"]),
                    len(df.filter((pl.col("isSignal") != 1))["Mbc"])],
        'Total' : [len(Mode1_df["Mbc"]),
                    len(Mode2_df["Mbc"]),
                    len(Mode3_df["Mbc"]),
                    len(Mode4_df["Mbc"]),
                    len(df["Mbc"])],
        'Pur' : [],
        'Eff' : [],

    }

    for sig, tot, gevt in zip(data["Signal"], data["Total"], data["Generated Event"]):
        data['Pur'].append(f'{sig*100/tot:.2f}%')
        data['Eff'].append(f'{sig*100/gevt:.2f}%')

    df_info = pd.DataFrame(data)
    return print(tabulate(df_info, headers='keys', tablefmt='pretty'))


def get_df(path, sel=None, treename=None):
    
    if treename == None:
        print(uproot.open(path).keys())
    
    else:
        with uproot.open(path) as file:
            tree=file[treename]
            if sel :
                arrays=tree.arrays(sel, library="pd")
            else : 
                arrays=tree.arrays(library="pd")
            df=arrays.drop(columns = ['__eventType__']) if '__eventType__' in arrays.columns.to_list() else arrays
            df=pl.from_pandas(df)

        return df

def sel_line(start, dest):
    plt.axvline(start, color='red', alpha=1, label=None, ls="--") if start else None
    plt.axvline(dest, color='red', alpha=1, label=None, ls="--") if dest else None
    

def label_setting(x_label, title=None, log=False, y_label='Candidates'):
    plt.xlabel(x_label, loc='right',fontsize=13) if x_label != None else None
    plt.ylabel(y_label, loc='top',fontsize=13)
    plt.title(title,fontsize=18, loc='left')
    plt.yscale("log") if log else None
    
    legend = plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1.1), framealpha=0.8, frameon=True, facecolor='white')
    frame = legend.get_frame()
    frame.set_edgecolor('black')
    plt.subplots_adjust(right=0.8, top=0.8, bottom=0.2)
    plt.tight_layout()


def Dofilter(df, selection_list=[]):
    if selection_list is not [] :
        for sel in selection_list:
            df=df.filter(sel)
    return df 

def check_channel(data, group=[]):
    
    grouped = data.group_by(group) \
                    .agg(pl.len()) \
                    .with_columns(pl.col('len').rank(descending=True).alias('Rank')) \
                    .sort('Rank') \
                    .to_pandas()
    pd.set_option('display.max_rows', 100)
    
    return grouped.head(20)

def sig_plot(df, var, isSignal, selection_list=[], bins=40, ranges=None, x_label=None, title=None, density=False, fake_label='fake', signal_label='signal'):
    x_label = x_label if x_label is not None else var
    df = Dofilter(df, selection_list)
    plt.hist(df.filter(pl.col(isSignal) == 1)[var], histtype='step', bins=bins, range=ranges, density=density,
             label=f'{signal_label} : {df.filter(pl.col(isSignal) == 1).shape[0]}')
    plt.hist(df.filter(pl.col(isSignal) != 1)[var], histtype='step', bins=bins, range=ranges, density=density,
             label=f'{fake_label} : {df.filter(pl.col(isSignal) != 1).shape[0]}')
    label_setting(x_label, title=title)


def get_signalMC_info_table(df, selection_list=[], generated_evt=100000):

    dmID_pipipi0 = (pl.col('eta_dmID') == 1)
    dmID_gamgam = (pl.col('eta_dmID') == 0)
    title_list=[r'$B^+ \rightarrow \rho^+_{\pi^+ \pi^0} \eta_{\gamma \gamma}$' , r'$B^+ \rightarrow \rho^+_{\pi^+ \pi^0} \eta_{\pi \pi \pi^0}$']

    df=Dofilter(df, selection_list)
    df_pp = df.filter(dmID_pipipi0)
    df_gg = df.filter(dmID_gamgam)
    
    total_Expected_events = 771E06

    rhoeta = 9.9E-6
    pipipi0 = 0.2274
    gamgam = 0.394
    
    df = pd.DataFrame({
        'String' : [title_list[1], title_list[0], 'Total'],
        "generated" : [generated_evt*0.5, generated_evt*0.5, generated_evt],
        "selected" : [df_pp['isSignal'].drop_nulls().shape[0], 
                                  df_gg['isSignal'].drop_nulls().shape[0], 
                                  df['isSignal'].drop_nulls().shape[0]],
        "# signal" : [df_pp.filter(pl.col('isSignal')==1).shape[0], 
                              df_gg.filter(pl.col('isSignal')==1).shape[0], 
                              df.filter(pl.col('isSignal')==1).shape[0]],
        "ratio" : [rhoeta*pipipi0, rhoeta*gamgam, rhoeta*(gamgam+pipipi0)],
    })

    df['purity'] = (df["# signal"] / df["selected"]).round(4)
    df[r'$\epsilon$'] = (df["# signal"] / df["generated"]).round(4)
    df['Expected'] = (total_Expected_events * df[r'$\epsilon$'] * df["ratio"]).round(2)

    df.style.set_table_styles(
    [{'selector': 'th.row_heading', 'props': [('font-size', '40px')]}]
    )
    df = df.drop(columns='ratio')

    return df



class selectionManager():
    def __init__(self, df_sig, df_fake, var):
        self.df_sig=df_sig
        self.df_fake=df_fake
        self.var=var


    def __normal_FOM(self, num_sig_cut, num_fake_cut):
        result=num_sig_cut/((num_fake_cut + num_sig_cut)**0.5) if (num_fake_cut + num_sig_cut) != 0 else 0

        return result

    def __punzi_FOM(self, num_sig_cut, num_fake_cut, num_gen_evt):
        efficiency=num_sig_cut / num_gen_evt
        result=efficiency / (1.5 + (num_fake_cut + num_sig_cut)**0.5)

        return result

    
    def fom_selected(self, ranges=[], interval=0.2, save=None, operator=">", x_label=None, method='normal', num_gen_evt=0, is_ignore_weight=False):
        '''
        df_fominfo = ana.selectionManager(df_sig, df_fake, var).fom_selected(ranges=[], interval=0.2, save=None, 
        operator=">", x_label=None, method='normal', num_gen_evt=0)
        '''
        df_sig=self.df_sig
        df_fake=self.df_fake
        var=self.var
        
        result_list=[]
        point_list=[]
        point=ranges[0]
        
        num_sig = df_sig.shape[0]
        num_fake = df_fake.shape[0]
        
        while point < ranges[1]:
            if operator == ">":
                num_sig_cut = df_sig.filter(pl.col(var) > point).shape[0]
                num_fake_cut = df_fake.filter(pl.col(var) > point).shape[0]
    
            if operator == "<":
                num_sig_cut = df_sig.filter(pl.col(var) < point).shape[0]
                num_fake_cut = df_fake.filter(pl.col(var) < point).shape[0]

            if is_ignore_weight == False :
                num_sig_cut*=df_sig['__weight__'][0]
                num_fake_cut*=df_fake['__weight__'][0]

            if method == 'normal':
                result=self.__normal_FOM(num_sig_cut, num_fake_cut) 
            if method == 'punzi' :
                result=self.__punzi_FOM(num_sig_cut, num_fake_cut, num_gen_evt) 
            
            result_list.append(result)
            point_list.append(point)
            point+=interval
        
        result_df = pd.DataFrame({
                    "point" : point_list,
                    "result" : result_list,
                })
        
        max_point = result_df[result_df['result'] == result_df['result'].max()]['point'].values[0] 
        max_result = result_df['result'].max() 

        print(f'Max selection point : {max_point} and the max result is {max_result}')

        '''
        plt.plot(result_df['point'], result_df['result'], label='FOM curve')
        ana.label_setting(x_label='$\eta_{\gamma \gamma} \ Energy \ asymmetric$', y_label='FOM result', title=title_list[0])
        '''

        return result_df


def find_awkward(df):
    return {col: dtype for col, dtype in df.dtypes.items() if dtype not in [np.int64, np.float64, np.datetime64, np.int32]}

def get_multiplicity(df, var, isSignal, selection_list=[]):

    df=Dofilter(df, selection_list)
    df_sig=df.filter(pl.col(isSignal)==1)
    df_fake=df.filter(pl.col(isSignal)!=1)
    
    df_sig_cut=df_sig.filter(pl.col(var)==1)
    df_fake_cut=df_fake.filter(pl.col(var)==1)
    
    num_sig=df_sig.shape[0]
    num_fake=df_fake.shape[0]
    num_sig_cut=df_sig_cut.shape[0]
    num_fake_cut=df_fake_cut.shape[0]
    
    result=(num_sig+num_fake) / (num_sig_cut+num_fake_cut)
    return result

def multi_label_plot(df_list, var, density=False, common_selection_list=[], ranges=None, bins=40, rescale=True, log=False, is_n_evt=False, x_label=None, title=None, y_label='Candidates'):

    '''
    jpsiX = ((pl.col('d0_d0')==443) | (pl.col('d0_d1')==443) | (pl.col('d1_d0')==443) | (pl.col('d1_d1')==443))
    jpsietap=(pl.col('d0_d0').abs()==443) & (pl.col('d0_d1').abs()==331) | (pl.col('d1_d0').abs()==443) & (pl.col('d1_d1').abs()==331)
    
    sel=[]
    df_list={
        r"$J/\psi \eta'$" : [[], df_signal_1],
        " qq" : [[], df_qq_1],
        " BB" : [[], df_BB_1],
    }
    multi_label_plot(df_list, var, density=False, common_selection_list=[], ranges=None, bins=40, rescale=True, log=False, is_n_evt=False)
    '''
    x_label = x_label if x_label != None else var
    for df_name, info in df_list.items():
        selection_list=info[0]
        selection_list+=common_selection_list
        df=info[1]
        df=Dofilter(df, selection_list)
        weight=df['__weight__'] if rescale else None
        n_evt = df['__weight__'].sum() if (rescale) and list(df['__weight__']) != [] else df.shape[0]
        n_evt = f' : {n_evt:.1f}' if is_n_evt == True else None
        label=df_name + n_evt if is_n_evt == True else df_name
        plt.hist(df[var], histtype='step', bins=bins, range=ranges, density=density, weights=weight, label=label, log=log)
        label_setting(x_label=x_label, title=title, y_label=y_label)


def get_remain_event_info(df, selection_list=[]):
    origin_df_event=df.shape[0]
    df_filtered=Dofilter(df, selection_list)
    filtered_df_event=df_filtered.shape[0]

    retained_ratio=filtered_df_event/origin_df_event
    rejected_ratio=1-retained_ratio
    
    return f'Rejected {rejected_ratio*100:.2f}%'


def qq_weight(df):
    df=df.with_columns(
        (pl.col('__weight__') *362 / 700)
        .alias('__weight__')
    )
    return df

def high_rela_bar(model, features):
    data = {
        'feature': features,
        'importance': model.feature_importances_,
    }
    
    df = pd.DataFrame(data)
    df = df.sort_values(by='importance')
    
    # plt.figure(figsize=(14,8))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout() 
    plt.bar(df['feature'], df['importance'])
    plt.xlabel('variables')
    plt.ylabel('Value')
    plt.tight_layout()


def apply_tpr(model, df_sample, features):
    xgboost_tpr = model.predict_proba(df_sample[features])[:,1]
    df_sample = df_sample.with_columns(pl.Series("xgboost_tpr", xgboost_tpr))

    print('xgboost_tpr the features added corrected.')
    return df_sample

def df_to_pdf(df, pdf_path):
    from matplotlib.backends.backend_pdf import PdfPages
    try :
        pandas_df = df.to_pandas()
    except :
        pandas_df = df
    
    html_table = pandas_df.to_html(index=False)
    pdf_path = pdf_path
    with PdfPages(pdf_path) as pdf:
        
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

    pass

