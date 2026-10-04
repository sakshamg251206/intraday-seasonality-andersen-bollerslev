# Reference values from Andersen & Bollerslev (1997)

`ab1997_values.csv` is transcribed by hand from the published tables of
Andersen, T.G. & Bollerslev, T. (1997), *Journal of Empirical Finance* 4, 115-158.

| column | source |
|---|---|
| sd, kurtosis, rho1, rho1_abs, VR_abs | Table 1 (a: DM-$, b: S&P 500) |
| persist_raw | Table 2, alpha+beta of MA(1)-GARCH(1,1) on raw returns |
| persist_filtered | Table 4, returns divided by the FFF periodic component |
| persist_standardized | Table 5, returns divided by daily vol x periodic component (blank = model not estimated / ARCH(1) only in the paper) |

`market = fx` is the DM-$ (1992-93, N=288); `equity` is S&P 500 futures (1986-89, N=80).
Values are used only for side-by-side comparison; they never enter an estimation.
