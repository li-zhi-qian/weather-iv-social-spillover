import scipy.stats as stats
import pandas as pd
import pickle
import numpy as np
import statsmodels.api as sm
from scipy import stats


def hausman_test(b1, b2, V1, V2):
    """Perform Hausman test
    b1, b2: Coefficient estimates from model 1 and model 2
    V1, V2: Covariance matrices of the coefficients from model 1 and model 2
    """
    diff = b2-b1
    V_diff = V1 - V2
    # Test statistic: H = (b2 - b1)' * (V1 - V2)^(-1) * (b2 - b1)
    stat = np.dot(np.dot(diff.T, np.linalg.inv(V_diff)), diff)
    df = b1.shape[0]  # degrees of freedom (number of coefficients)
    p_value = 1 - stats.chi2.cdf(stat, df)
    return stat, p_value


def test_and_print(model1, model2):
    # Coefficients
    b1 = model1.params.values#.reshape(-1, 1)
    b2 = model2.params.values#.reshape(-1, 1)
    print(f"model1: coefficients: {b1}; {b1.shape}")
    print(f"model2: coefficients: {b2}; {b1.shape}")

    # Variance-covariance matrices
    V1 = model1.cov_params_default
    V2 = model2.cov_params_default
    print(f'model1: cov: {V1}; {V1.shape}')
    print(f'model2: cov: {V2}; {V2.shape}')

    # Perform Hausman test
    stat, p_value = hausman_test(b1, b2, V1, V2)

    print(f"Hausman Test Statistic: {stat}")
    print(f"P-value: {p_value}")

    if p_value < 0.05:
        print("Reject null hypothesis: The coefficients are significantly different.")
    else:
        print(
            "Fail to reject null hypothesis: No significant difference between coefficients."
        )
    print("-------------")


def test_for_n(models1, models2, n):
    """perform hausman test for each week (wk2-wk6)"""
    for i in range(n):
        print(f"Week {i+2}: ")
        model1 = models1[i]
        model2 = models2[i]
        test_and_print(model1, model2)


print("==============OLS and IV==============")

with open("results/s2_baseline.pkl", "rb") as file:
    iv_model = pickle.load(file)

with open("results/ols.pkl", "rb") as file:
    ols_model = pickle.load(file)

test_for_n(ols_model, iv_model, 5)
print("\n\n\n")

print("================Control for weather or not==============")

with open("results/s2_baseline.pkl", "rb") as file:
    models2 = pickle.load(file)

with open("results/s2_control_weather.pkl", "rb") as file:
    models1 = pickle.load(file)

test_for_n(models1, models2, 5)
print("\n\n\n")

# print("==============OLS and Handselected-IV==============")

# with open("results/s2_hand_select.pkl", "rb") as file:
#     iv_model = pickle.load(file)

# with open("results/ols.pkl", "rb") as file:
#     ols_model = pickle.load(file)

# test_for_n(ols_model, iv_model, 5)

