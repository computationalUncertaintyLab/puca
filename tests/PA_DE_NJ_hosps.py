
#mcandrew
import numpy as np
import pandas as pd
from puca import puca

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import seaborn as sns

import scienceplots

import jax 

if __name__ == "__main__":

    hosp_data = pd.read_csv("./data/target-hospital-admissions.csv")
    ili_data  = pd.read_csv("./data/ili_data_all_states_2021_present__formatted.csv")

    def add_season_info(d):
        from epiweeks import Week
        from datetime import datetime
        
        time_data = {"date":[],"MMWRYR":[],"MMWRWK":[],"enddate":[],"season":[]}
        for time in d[ ["date"] ].drop_duplicates().values:
            w = Week.fromdate(datetime.strptime(time[0], "%Y-%m-%d"))
            time_data["date"].append(time[0])
            time_data["MMWRYR"].append(w.year)
            time_data["MMWRWK"].append(w.week)
            time_data["enddate"].append(w.enddate().strftime("%Y-%m-%d"))

            if w.week>=40 and w.week<=53:
                time_data["season"].append("{:d}/{:d}".format(w.year,w.year+1) )
            elif w.week>=1 and w.week<=20:
                time_data["season"].append("{:d}/{:d}".format(w.year-1,w.year) )
            else:
                time_data["season"].append("offseason")
        time_data = pd.DataFrame(time_data)

        d = d.merge(time_data, on = ["date"])
        return d

    hosp_data       = add_season_info(hosp_data)
    
    STATES = ["42","34","10"] #<--these are FIPS

    all_hosps = []
    for STATE in STATES:
        state_hosp_data = hosp_data.loc[ (hosp_data.location==STATE) & (hosp_data.season!="offseason") ]
        hosp_data__wide = pd.pivot_table( index = ["MMWRWK"], columns = ["season"], values = ["value"], data = state_hosp_data )
        hosp_data__wide = hosp_data__wide.loc[ list(np.arange(40,52+1)) + list(np.arange(1,20+1))]

        all_hosps.append(hosp_data__wide.to_numpy()[:,2:])


    X                 = None                                                               #<--external covariates
    y                 = all_hosps
    target_indicators = [ _.shape[1]-1 for _ in y ]                                        #<--last column in each dataset


    puca_model = puca(y = y , target_indicators = target_indicators, X = None)
    puca_model.fit()
    forecast   = puca_model.forecast()
    
    
    _25,_10,_250,_500,_750,_900,_975 = np.percentile( np.clip(forecast,0,None) , [2.5,10,25,50,75,90,97.5], axis=0)

    plt.style.use("science")
    fig, axs = plt.subplots(1,3)

    times = np.arange(len(_25))
    all_tobs  = puca_model.tobs 

    for n,(ax,tobs,STATE,hosps) in enumerate(zip(axs, all_tobs,STATES,all_hosps)):
        ax.fill_between(times[tobs:], _25[tobs:,n], _975[tobs:,n], color="blue",alpha=0.10)
        ax.fill_between(times[tobs:], _10[tobs:,n], _900[tobs:,n], color="blue",alpha=0.10)
        ax.fill_between(times[tobs:], _250[tobs:,n], _750[tobs:,n], color="blue",alpha=0.10)
        ax.plot(times[tobs:],_500[tobs:,n],color="purple")

        ax.plot(hosps[:,-1],color="black")
        ax.plot(hosps,color="black",alpha=0.05)
        ax.set_ylabel(f"{STATE} inc hosps")       

    fig.set_size_inches( (8.5-2), (11-2)/3 )
    fig.set_tight_layout(True)
    
    plt.show()


