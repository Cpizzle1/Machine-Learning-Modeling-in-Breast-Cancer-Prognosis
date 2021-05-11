
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd    


def calculate_vif_(X, thresh=5.0):
    """[drops columns of dataframe if VIF is above threshold]
    Args:
        X ([dataframe]): [pandas dataframe]
        thresh (float, optional): [Threshold VIF value to drop columns unti reached]. Defaults to 5.0.
    Returns:
        [dataframe]: [dataframe with columns of high multicolinarity dropped]
    """
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]
        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True
    print('Remaining variables:')
    print(X.columns[variables])
    return X.iloc[:, variables]

if __name__ == '__main__':

    mrna_df =pd.read_csv('/Users/cp/Documents/dsi/capstone2/capstone2/data/capstone2.mrn_df2.csv')
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(mrna_df.values, i) for i in range(mrna_df.shape[1])]
    vif["features"] = mrna_df.columns
    VIF_df = calculate_vif_(mrna_df)