
#mcandrew
import numpy as np
import pandas as pd
from puca import puca

import matplotlib.pyplot as plt
import seaborn as sns

import scienceplots

if __name__ == "__main__":

    hosp_data = pd.read_csv("./data/target-hospital-admissions.csv")

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


    STATE = "42" #<--these are FIPS
    
    hosp_data = add_season_info(hosp_data)
    hosp_data = hosp_data.loc[ (hosp_data.location==STATE) & (hosp_data.season!="offseason") ]

    puca_model = puca(hosp_data)
    puca_model.fit(use_anchor=True)

    forecasts = puca_model.forecast()

    natural_scale_forecasts = forecasts["forecast_natural"]
    organized_dataset = puca_model.d_wide
    

    this_season_data = organized_dataset.iloc[:,-1].values
    last_obs = np.min(np.argwhere(np.isnan(this_season_data)))


    fig,axs = plt.subplots(2,2)
    
    ax = axs[0,0]

    weeks = np.arange(len(this_season_data))

    #--observed data
    ax.plot( weeks, this_season_data, lw = 2, color="black" )
    ax.scatter( weeks, this_season_data, s=8,  color="black" )

    #--past seasonal data
    ax.plot( organized_dataset.to_numpy()[:,:-1], color = "black", alpha=0.1, lw=0.5 )
    
    #--puca forecast
    _25,_250,_500,_750,_900,_975 = np.percentile( natural_scale_forecasts, [2.5, 25,50,75,90,97.5],axis=0 )

    ax.plot( weeks[last_obs:], _500[last_obs:] )
    ax.fill_between( weeks[last_obs:], _25[last_obs:], _975[last_obs:], color = "blue", alpha=0.25 )
    ax.fill_between( weeks[last_obs:], _250[last_obs:], _750[last_obs:], color = "blue", alpha=0.25 )

    ax = axs[1,0]

    fundemental_shapes = puca_model.post_samples["L"].mean(0)
    ax.plot(fundemental_shapes)

    ax = axs[0,1]

    sns.boxplot(puca_model.post_samples["w"], fliersize=0)
    ax.set_xlabel("Seasons")
    ax.set_ylabel("Similarity (measured as weight)G")
    

    plt.show()
    


    

