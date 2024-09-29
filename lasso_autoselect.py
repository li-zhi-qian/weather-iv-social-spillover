import numpy as np
import sys
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plt
import conversion
from sklearn.linear_model import lasso_path
from itertools import cycle

instruments = conversion.txt_to_lst(
    "open_res_own_mat5_10_6 open_res_own_mat5_15_6 open_res_own_mat5_20_6 open_res_own_mat5_25_6 open_res_own_mat5_30_6 open_res_own_mat5_35_6 open_res_own_mat5_40_6 open_res_own_mat5_45_6 open_res_own_mat5_50_6 open_res_own_mat5_55_6 open_res_own_mat5_60_6 open_res_own_mat5_65_6 open_res_own_mat5_70_6 open_res_own_mat5_75_6 open_res_own_mat5_80_6 open_res_own_mat5_85_6 open_res_own_mat5_90_6 open_res_own_mat5_95_6 open_res_own_mat5_10_0 open_res_own_mat5_15_0 open_res_own_mat5_20_0 open_res_own_mat5_25_0 open_res_own_mat5_30_0 open_res_own_mat5_35_0 open_res_own_mat5_40_0 open_res_own_mat5_45_0 open_res_own_mat5_50_0 open_res_own_mat5_55_0 open_res_own_mat5_60_0 open_res_own_mat5_65_0 open_res_own_mat5_70_0 open_res_own_mat5_75_0 open_res_own_mat5_80_0 open_res_own_mat5_85_0 open_res_own_mat5_90_0 open_res_own_mat5_95_0 open_res_own_rain_6 open_res_own_rain_0 open_res_own_snow_6 open_res_own_snow_0 open_res_own_prec_0_6 open_res_own_prec_1_6 open_res_own_prec_2_6 open_res_own_prec_3_6 open_res_own_prec_4_6 open_res_own_prec_5_6 open_res_own_prec_0_0 open_res_own_prec_1_0 open_res_own_prec_2_0 open_res_own_prec_3_0 open_res_own_prec_4_0 open_res_own_prec_5_0 "
)


def select_lambda(x_scaled, y, num_iv=1, visualization=True):
    alphas_lasso, coefs_lasso, _ = lasso_path(
        x_scaled, y, alphas=np.arange(0.055, 0.2, step=0.001).tolist()
    )

    alpha_0 = -1
    alpha_1 = -1
    indices = []
    for coef, alpha in zip(coefs_lasso.T, alphas_lasso):
        non_zero_count = np.sum(coef != 0)
        if non_zero_count >= num_iv + 1:
            break
        if non_zero_count == num_iv:
            alpha_1 = alpha
            indices = np.where(coef != 0)[0]
            print(coef)
        if non_zero_count == num_iv - 1:
            alpha_0 = alpha

    if visualization:
        # Display lasso path
        plt.figure(1)
        colors = cycle(["b", "r", "g", "c", "k", "m", "y", "orange", "purple", "pink"])
        i = 0
        instrument_count = 0
        plt.text(
            0.57,
            0.95,
            "Selected instrument(s):",
            color="black",
            fontsize=10,
            va="top",
            transform=plt.gca().transAxes,
        )

        for coef_l, c, i in zip(coefs_lasso, colors, list(range(len(coefs_lasso)))):
            l1 = plt.semilogx(alphas_lasso, coef_l, c=c)
            if i in indices:
                instrument_count += 1
                plt.text(
                    0.59,
                    0.95 - instrument_count * 0.05,
                    f"{instruments[i]}",
                    color=c,
                    fontsize=10,
                    va="top",
                    transform=plt.gca().transAxes,
                )
        plt.plot(
            [(alpha_0), (alpha_0)],
            [np.min(coefs_lasso), 0.01],
            color="black",
            linestyle="--",
        )
        plt.xlabel("Lambda")
        plt.ylabel("Variable Coefficients")
        plt.title("LASSO Path")
        plt.axis("tight")

        plt.savefig("lasso_path.jpg", dpi=300, bbox_inches="tight")
        # plt.show()

    if alpha_1 == -1 and alpha_0 == -1:
        return 0, []
    else:
        return (alpha_0 + alpha_1) / 2, [instruments[i] for i in indices]


what_and_how_many = ["tickets", 3]
outcome = what_and_how_many[0]
# load data
with open("mod_pkl/for_analyses.pkl", "rb") as file:
    data = pickle.load(file)
data = data[(data["wk1"] == 1) & (data[outcome] > 0)]

x = data[instruments]
# print(x.describe())

y = data[outcome + "_wk1d_r"] - data[outcome + "_wk1d_r"].mean()  # demean y

# standardize x values
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, test_size=0.2, random_state=42
)

alpha, selected_instruments = select_lambda(
    x_scaled, y, num_iv=what_and_how_many[1]
)  # select lambda/model

print(f"Selected instrument(s):{selected_instruments}")
