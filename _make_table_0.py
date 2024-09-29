import statsmodels.api as sm
import pickle
from statsmodels.sandbox.regression.gmm import IV2SLS



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

instrument_dict = {1:["open_res_own_mat5_75_0"],
                   2:["open_res_own_mat5_75_0","open_res_own_mat5_55_0"],
                   3:["open_res_own_mat5_75_0","open_res_own_mat5_55_0","open_res_own_mat5_75_6"]}
ivreg_s1(instrument_dict[1])