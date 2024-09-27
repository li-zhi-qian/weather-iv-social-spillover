import pickle
import conversion as conv
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
from statsmodels.sandbox.regression.gmm import IV2SLS
import statsmodels.stats.sandwich_covariance as sw
import conversion
import random
from sklearn.metrics import r2_score

# Load preprocessed data
with open("mod_pkl/for_analyses.pkl", "rb") as file:
    data = pickle.load(file)


def do_ivreg(X, Y, Z):
    """run 2SLS regression, X: endogenous var., Y: dependent var., Z: instrumental var."""
    # Add constant to X and Z
    X = sm.add_constant(X)
    Z = sm.add_constant(Z)
    return IV2SLS(Y, X, Z).fit()


# Define relevant columns/variables
outcome = "tickets"  # default value: all tickets
clus = "date"
control1 = [
    x
    for x in data.keys()
    if x.startswith("ww")
    or x.startswith("yy")
    or x.startswith("h")
    or x.startswith("dow_")
]


def get_res(selected_ownweather):
    """* get residual tickets controling for own weather for weeks 2+"""
    for i in range(2, 7):
        # filter for rows for wk i
        filtered_data = data[(data["wk" + str(i)] == 1) & (data[outcome] > 0)]
        # define y and X, add constant to X
        y = filtered_data[outcome]
        X = filtered_data[control1 + selected_ownweather]  #
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        # extract residuals
        filtered_data[outcome + "_r"] = model.resid
        # assign max residuals in the group to each individual in the group
        filtered_data[outcome + "_wk" + str(i) + "d_r"] = filtered_data.groupby(
            ["opening_sat_date", "dow"]
        )[outcome + "_r"].transform("max")
        # get the residuals back into the original dataframe
        data.loc[filtered_data.index, outcome + "_r"] = filtered_data[outcome + "_r"]
        data.loc[filtered_data.index, outcome + "_wk" + str(i) + "d_r"] = filtered_data[
            outcome + "_wk" + str(i) + "d_r"
        ]
    # print(filtered_data[[x for x in filtered_data.keys() if x.startswith('tickets')]].head(10))
    data.replace({np.NaN: 0.0}, inplace=True)
    # (PLEASE OMIT THIS)Calculate total resifuals for rows with the same opening saturday
    data[outcome + "_wkn1d_r"] = sum(
        [data[(outcome + "_wk" + str(i) + "d_r")] for i in range(2, 7)]
    )
    data[outcome + "_wkn1d_r"] = data.groupby("opening_sat_date")[
        outcome + "_wkn1d_r"
    ].transform("sum")


def ivreg_s1(selected):
    """First-stage"""
    filtered_data = data[(data["wk1"] == 1) & (data[outcome] > 0)]
    y = filtered_data[outcome + "_wk1d_r"]
    X = filtered_data[selected]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit(
        cov_type="cluster", cov_kwds={"groups": filtered_data["date"]}
    )
    hypothesis_string = ", ".join([f"{var} = 0" for var in selected])

    hypothesis_test = model.f_test(hypothesis_string)
    print(hypothesis_test)
    f_stat = hypothesis_test.fvalue
    print(f"F-statistic: {f_stat}")
    return model.summary()


def ivreg_s2(selected):
    coefficients = []
    std_errors = []
    models = []  # save the entire model to a list
    model_summaries = ""
    """Second Stage"""
    for i in range(2, 7):
        filtered_data = data[(data[f"wk{i}"] == 1) & (data[outcome] > 0)]
        iv_model = do_ivreg(
            X=filtered_data[outcome + "_wk1d_r"],
            Y=filtered_data[f"{outcome}_wk{i}d_r"],
            Z=filtered_data[selected],
        )
        coefficients.append(f"{iv_model.params.iloc[1]:.3f}")

        # Define the cluster variable
        clusters = filtered_data[clus]
        # Calculate clustered covariance matrix
        cluster_cov = sw.cov_cluster(iv_model, clusters)
        # Update the model's covariance with the clustered covariance
        iv_model.cov_params_default = cluster_cov
        # Get clustered standard errors
        clustered_std_errors = iv_model.bse
        std_errors.append(f"{clustered_std_errors.iloc[1]:.4f}")  # std_errors

        # Save full model summary to a text file
        # with open(f"model_summary_wk{i}.txt", "w") as file:
        #     file.write(model_summary.as_text())

        models.append(iv_model)
        model_summaries += "\n\n"
        model_summaries += str(iv_model.summary())

    # Save the coefficients and standard deviations to a CSV file
    results_df = pd.DataFrame(
        {
            "Week": list(range(2, 7)),
            "Coefficient": coefficients,
            "Std_Error(clustered)": std_errors,
        }
    )
    print(results_df)
    return results_df, models, model_summaries


def ols():
    ols_coef = []
    ols_sd = []
    ols_r2 = []
    models_ols = []
    for i in range(2, 7):
        filtered_data = data[data[f"wk{i}"] == 1]
        X = filtered_data[outcome + "_wk1d_r"]
        y = filtered_data[f"{outcome}_wk{i}d_r"]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        ols_coef.append(f"{model.params.iloc[1]:.3f}")

        # Define the cluster variable
        clusters = filtered_data[clus]
        # Calculate clustered covariance matrix
        cluster_cov = sw.cov_cluster(model, clusters)
        # Update the model's covariance with the clustered covariance
        model.cov_params_default = cluster_cov
        # Get clustered standard errors
        clustered_std_errors = model.bse
        ols_sd.append(f"{clustered_std_errors.iloc[1]:.4f}")  # std_errors

        ols_r2.append(r2_score(y, model.predict(X)))
        models_ols.append(model)

    results_df = pd.DataFrame(
        {
            "Week": list(range(2, 7)) + [-1],
            "Coefficient": ols_coef,
            "sd(clustered)": ols_sd,
            "R2": ols_r2,
        }
    )
    return results_df, models_ols


def save_as(obj, name: str):
    """for saving results"""
    if isinstance(obj, list):
        with open("results/" + name + ".pkl", "wb") as file:
            pickle.dump(obj, file)
    elif isinstance(obj, pd.DataFrame):
        obj.to_csv("results/" + name + ".csv", index=True)
    else:
        with open("results/" + name + ".txt", "w") as file:
            file.write(str(obj))


if __name__ == "__main__":
    instruments = conversion.txt_to_lst(
        "open_res_own_mat5_10_6 open_res_own_mat5_15_6 open_res_own_mat5_20_6 open_res_own_mat5_25_6 open_res_own_mat5_30_6 open_res_own_mat5_35_6 open_res_own_mat5_40_6 open_res_own_mat5_45_6 open_res_own_mat5_50_6 open_res_own_mat5_55_6 open_res_own_mat5_60_6 open_res_own_mat5_65_6 open_res_own_mat5_70_6 open_res_own_mat5_75_6 open_res_own_mat5_80_6 open_res_own_mat5_85_6 open_res_own_mat5_90_6 open_res_own_mat5_95_6 open_res_own_mat5_10_0 open_res_own_mat5_15_0 open_res_own_mat5_20_0 open_res_own_mat5_25_0 open_res_own_mat5_30_0 open_res_own_mat5_35_0 open_res_own_mat5_40_0 open_res_own_mat5_45_0 open_res_own_mat5_50_0 open_res_own_mat5_55_0 open_res_own_mat5_60_0 open_res_own_mat5_65_0 open_res_own_mat5_70_0 open_res_own_mat5_75_0 open_res_own_mat5_80_0 open_res_own_mat5_85_0 open_res_own_mat5_90_0 open_res_own_mat5_95_0 open_res_own_rain_6 open_res_own_rain_0 open_res_own_snow_6 open_res_own_snow_0 open_res_own_prec_0_6 open_res_own_prec_1_6 open_res_own_prec_2_6 open_res_own_prec_3_6 open_res_own_prec_4_6 open_res_own_prec_5_6 open_res_own_prec_0_0 open_res_own_prec_1_0 open_res_own_prec_2_0 open_res_own_prec_3_0 open_res_own_prec_4_0 open_res_own_prec_5_0 "
    )
    instrument_dict = {
        "tickets_ratedgpg": "open_res_own_prec_0_6",
        "tickets_adult": "open_res_own_mat5_75_0",
        "tickets": "open_res_own_mat5_75_0",
        "tickets_hs": "open_res_own_mat_la_cens_6",
    }
    comments_dict = {
        "tickets_ratedgpg": "gpg_prec_iv",
        "tickets_adult": "adult",
        "tickets": "baseline",
        "tickets_hs": "hand_select",
    }
    outcome = "tickets"  # TODO: set dependent variable
    hand_select = "_hs"  # TODO: "_hs" or "" hand selected or not
    selected_instruments = [
        instrument_dict[outcome + hand_select]
    ]  # set instruments["open_res_own_mat_la_cens_6"] #open_res_own_prec_0_6 "open_res_own_mat5_75_0"
    comments = comments_dict[
        outcome + hand_select
    ]  # additional stuffs in output file names open_res_own_mat5_75_0

    s1_model = ivreg_s1(
        selected=selected_instruments
    )  # do 2sls stage 1 with the selected instrument(s)
    save_as(s1_model, "s1_" + comments)  # save stage 1 results

    selected_ownweather = conv.txt_to_lst(
        "own_mat10_10 own_mat10_20 own_mat10_30 own_mat10_40 own_mat10_50 own_mat10_60 own_mat10_70 own_mat10_80 own_mat10_90 own_snow own_rain own_prec_0 own_prec_1 own_prec_2 own_prec_3 own_prec_4 own_prec_5"
    )  # get weather control variables
    get_res(selected_ownweather)  # residualize film ticket measure

    results, model_list, summary = ivreg_s2(
        selected=selected_instruments
    )  # do stage 2 and retrieve coefficients and models
    save_as(results, "s2_" + comments)
    save_as(model_list, "s2_" + comments)
    save_as(summary, "s2_" + comments)
