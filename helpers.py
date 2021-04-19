import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import importlib
matplotlib.use("Agg")
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV
from sklearn.decomposition import PCA
from src.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor


# importlib.reload(src.pipeline)
# from src.pipeline import Pipeline



def plot_corr_matrix(df):
    f = plt.figure(figsize=(19, 15))
    plt.matshow(df.corr(), fignum=f.number)
    plt.xticks(range(df.shape[1]), df.columns, fontsize=16, rotation=90)
    plt.yticks(range(df.shape[1]), df.columns, fontsize=16)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.xlabel('Correlation Matrix', fontsize=22)
    # plt.tight_layout()

def get_redundant_pairs(df):
     '''Get diagonal and lower triangular pairs of correlation matrix'''
     pairs_to_drop = set()
     cols = df.columns
     for i in range(0, df.shape[1]):
         for j in range(0, i+1):
             pairs_to_drop.add((cols[i], cols[j]))
     return pairs_to_drop

def get_top_abs_correlations(df, n=5):
     au_corr = df.corr().abs().unstack()
     labels_to_drop = get_redundant_pairs(df)
     au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
     return au_corr[0:n]

def scree_plot(ax, pca, n_components_to_plot=8, title=None): # Credit: Galvanize Data Science
    """Make a scree plot showing the variance explained (i.e. varaince of the projections) for the principal components in a fit sklearn PCA object.
    
    Parameters
    ----------
    ax: matplotlib.axis object
      The axis to make the scree plot on.
      
    pca: sklearn.decomposition.PCA object.
      A fit PCA object.
      
    n_components_to_plot: int
      The number of principal components to display in the skree plot.
      
    title: str
      A title for the skree plot.
    """
    num_components = pca.n_components_
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
    ax.plot(ind, vals, color='blue')
    ax.scatter(ind, vals, color='blue', s=50)

    for i in range(num_components):
        ax.annotate(r"{:2.2f}".format(vals[i]), 
                   (ind[i]+0.2, vals[i]+0.005), 
                   va="bottom", 
                   ha="center", 
                   fontsize=12)

    ax.set_xticklabels(ind, fontsize=12)
    ax.set_ylim(0, max(vals) + 0.05)
    ax.set_xlim(0 - 0.45, n_components_to_plot + 0.45)
    ax.set_xlabel("Principal Component", fontsize=16)
    ax.set_ylabel("Variance Explained (%)", fontsize=16)
    if title is not None:
        ax.set_title(title, fontsize=20)


def plot_num_estimators_mse(num_estimator_list, train_errors_rf, test_errors_rf):
    plt.figure(figsize=(15,10))
    plt.plot(num_estimator_list, train_errors_rf, label='Training MSE')
    plt.plot(num_estimator_list, test_errors_rf, label='Test MSE')
    plt.xlabel('Number of Estimators')
    plt.ylabel('MSE')
    plt.xscale('linear')
    plt.title('Random Forest MSE vs. Num Estimators')
    plt.legend()
    
def gridsearch():
    parameters = {'n_estimators': (2, 5, 10, 20, 30), 
                'max_depth': (None, 5, 7), 
                'max_features': ('auto', 'sqrt', 'log2')}
    rf = RandomForestRegressor(verbose=True, n_jobs=-1)
    grid = GridSearchCV(rf, parameters, verbose=1, n_jobs=-1)

    grid.fit(X_train,y_train)
    gridscore_test = grid.score(X_test, y_test)
    grisdcore_train = grid.score(X_train, y_train)
    return gridscore_train, gridscore_test, grid.best_params_

def pdplots():

    first_pdp = ['generation fossil gas', 
            'generation fossil hard coal', 'total load actual'] 
             
    second_pdp = ['generation other renewable',
             'generation solar']

    third_pdp = ['generation wind onshore', 'generation nuclear', 
             'Madrid_wind_speed']

    fourth_pdp = ['generation hydro pumped storage consumption']

    plot_partial_dependence(rf, X_train, first_pdp, n_jobs=-1)
    fig.suptitle("Partial Dependence of Energy Price on Various Generation Types")
    fig.subplots_adjust(hspace=2.0, wspace=2.0)
    plt.tight_layout()
    plt.savefig('images/pd1.png')
    plt.close()

    plot_partial_dependence(rf, X_train, second_pdp, n_jobs=-1)
    fig.suptitle("Partial Dependence of Energy Price on Various Generation Types")
    fig.subplots_adjust(hspace=2.0, wspace=2.0)
    plt.tight_layout()
    plt.savefig('images/pd2.png')
    plt.close()

    plot_partial_dependence(rf, X_train, third_pdp, n_jobs=-1)
    fig.suptitle("Partial Dependence of Energy Price on Various Generation Types")
    fig.subplots_adjust(hspace=2.0, wspace=2.0)
    plt.tight_layout()
    plt.savefig('images/pd3.png')
    plt.close()

    plot_partial_dependence(rf, X_train, fourth_pdp, n_jobs=-1)
    fig.suptitle("Partial Dependence of Energy Price on Various Generation Types")
    fig.subplots_adjust(hspace=2.0, wspace=2.0)
    plt.tight_layout()
    plt.savefig('images/pd4.png')
    plt.close()

def compare_default_models():
    models = [RandomForestRegressor(n_jobs=-1), Lasso(), Ridge(), LinearRegression(n_jobs=-1)]

    for model in models:
        pipe = SKPipe([('scaler', StandardScaler()), (f'{model}', model)], verbose=True)
        pipe.fit(X_train, y_train)
        
        train_score = pipe.score(X_train, y_train)
        test_score = pipe.score(X_test, y_test)
        
        print(f"{model} \n\ntest score = {test_score}\n train score = {train_score}\n\n")

def pca_with_scree(full_df):
    '''Takes a Pipeline object
    '''
    print("\nLet's try PCA")
    pca = PCA()
    X_pca = pca.fit_transform(full_df.X_std)

    fig, ax = plt.subplots(figsize=(10, 6))
    scree_plot(ax, pca, title="Scree Plot for Energy Principal Components")
    plt.savefig('images/pca_part2.png')
    plt.close()

def feat_imp_plots(full_df, rf):
    feature_names = full_df.X.columns
    feat_imp = pd.DataFrame({'feature_name':feature_names, 'feat_imp': rf.feature_importances_})
    feat_imp.sort_values('feat_imp',ascending=False,inplace=True)
    fig, ax = plt.subplots(1, figsize=(8,10))
    ax.barh(feat_imp['feature_name'].head(9), feat_imp['feat_imp'].head(9))
    ax.invert_yaxis()
    ax.set_title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.savefig('images/feature_imp_auto.png')
    plt.close()

def plot_oob_error():
    fig, ax = plt.subplots()
    oob_diff = []
    oob = []
    for i in list(range(2,20)):
        rf = RandomForestRegressor(max_depth=i, max_features='auto', n_estimators=30, oob_score=True, n_jobs=-1)

        rf.fit(X_train, y_train)

        print(f"R2 Train = {rf.score(X_train, y_train)}")
        print(f"R2 Test = {rf.score(X_test, y_test)}")
        print(f"R2 Holdout = {rf.score(X_holdout, y_holdout)}")
        print(f"OOB score = {rf.oob_score_}")
        oob_diff.append(rf.score(X_train, y_train) - rf.oob_score_)
        oob.append(rf.oob_score_)

    ax.plot(oob_diff, color='red')
    ax.plot(oob, color='blue')
    ax.set_title("Reducing OOB Error by limiting max_depth")
    plt.savefig('images/oob.png')

if __name__ == '__main__':
    # energy_df = pd.read_csv('data/energy_dataset.csv',index_col=0, parse_dates=[0])
    # weather_df = pd.read_csv('data/weather_features.csv',index_col=0, parse_dates=[0])
    print("Loading Data")
    # read in files from s3 bucket
    energy = Pipeline('s3://ajzcap2/energy_dataset.csv')
    weather = Pipeline('s3://ajzcap2/weather_features.csv')

    #make index a datetime object
    energy.my_reset_index()
    weather.my_reset_index()

    # Drop columns
    weather_drop_cols = ['weather_icon', 'weather_main', 'weather_id', 'temp_min', 'temp_max']
    energy_drop_cols = ['generation fossil coal-derived gas','generation fossil oil shale', 
                        'generation fossil peat', 'generation geothermal',
                        'generation marine', 'generation hydro pumped storage aggregated',
                         'forecast wind offshore eday ahead', 'generation wind offshore', 
                         'price day ahead', 'total load forecast', 'forecast wind onshore day ahead', 'forecast solar day ahead']

    for i in weather_drop_cols:
        weather.df.drop(i, axis=1, inplace=True)
    for i in energy_drop_cols:
        energy.df.drop(i, axis=1, inplace=True)

    # propagate last valid observation forward to next valid to fill NaNs
    for i in energy.df.columns:
        energy.df[i].fillna(method='pad', inplace=True)

    # for i in energy.df.columns:
    #     print(f"{i}: missing {energy.df[i].isna().sum()}")

    # Demonstrate over-featurization of weather.df
    # for i in weather.df.weather_description.unique():
    #     print(f"{i} = {weather.df.weather_id[weather.df.weather_description == i].unique()}, {weather.df.weather_main[weather.df.weather_description == i].unique()}")




    # plot_corr_matrix(energy.df)
    # plt.savefig('images/energy_corr.png')
    # # plt.show()
    # plt.close()
    # plot_corr_matrix(weather.df)
    # plt.savefig('images/weather_corr.png')
    # # plt.show()
    # plt.close()


    # Clean Catagoricals
    weather.clean_categoricals(['weather_description'])



    #Featurizing Cities
    city_df_list = weather.featurize_cities(['Valencia', 'Madrid', "Bilbao", ' Barcelona', 'Seville'])

    valencia = Pipeline.from_df(city_df_list[0])
    madrid = Pipeline.from_df(city_df_list[1])
    bilbao = Pipeline.from_df(city_df_list[2])
    barcelona = Pipeline.from_df(city_df_list[3])
    sevilla = Pipeline.from_df(city_df_list[4])

    # There has GOT to be a better way to do this
    vm = valencia.merge_dfs(madrid.df)
    bb = bilbao.merge_dfs(barcelona.df)
    sbb = sevilla.merge_dfs(bb.df)
    all_cities_df = vm.merge_dfs(sbb.df)

    # clean residual col names that came from the merge
    for i in ["Valencia_city_name", " Barcelona_city_name", "Bilbao_city_name", 
            "Seville_city_name", "Madrid_city_name", 'Seville_snow_3h', ' Barcelona_snow_3h']:
        all_cities_df.df.drop(i, axis=1, inplace=True)

    
    # Transformations
    print('\nPerforming transformations')
  

    # Merge energy with the featurized cities DF to make the complete DataFrame
    full_df = energy.merge_dfs(all_cities_df.df)

    plot_corr_matrix(full_df.df)
    plt.savefig('images/full_corr.png')
    plt.close()

    get_top_abs_correlations(full_df.df, 10)


    print('\nCreating train, test, and holdout sets')
    full_df.getXy('price actual')
    full_df.create_holdout()



    print('\nWriting train, test, and holdouts to filesystem')

    train_test_split_holdout_list = [full_df.X_train, full_df.X_test, 
                                    full_df.X_holdout, full_df.X_std, full_df.y_train, 
                                    full_df.y_test, full_df.y_holdout]

    ttsh_filenames = ['X_train', 'X_test', 'X_holdout', 'X_std','y_train', 
                    'y_test', 'y_holdout']


    # Dont need to to this everytime I run the script for EDA
    # for (i, fname) in zip(train_test_split_holdout_list, ttsh_filenames):
    #         i.to_csv(f"data/{fname}.csv")
   
    # for (i, fname) in zip(train_test_split_holdout_list, ttsh_filenames):
    #     if type(i) == 'numpy.ndarray':
    #         i.to_csv(f's3://ajzcap2/{fname}.csv')
   

    
    
    plot_corr_matrix(energy.df)
    plt.savefig('images/clean_energy_corr.png')
    # plt.show()
    plt.close()
    plot_corr_matrix(weather.df)
    plt.savefig('images/clean_weather_corr.png')
    # plt.show()
    plt.close()

    print("\nLasso time, yee-haw")
   
    

# LassoCV

    X = full_df.X_std
    y = full_df.y_train

    reg = LassoCV(random_state=0, verbose=True, n_jobs=-1, max_iter = 10000)
    reg.fit(X,y)
    regscore = reg.score(X, y)
    print(regscore)
    y_preds = reg.predict(full_df.Xscaler.transform(full_df.X_test))
    best_alpha = reg.alpha_
    coefs = reg.coef_
    print(f"Best alpha = {best_alpha}")
    

# VIF
    # print("\nChecking VIF")
    # vif = pd.DataFrame()
    # vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    # vif["features"] = full_df.X.columns

    # print(vif.sort_values('VIF Factor', ascending=False).head(20).round(1))




# PCA
    print("\nLet's try PCA")
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(full_df.X_std)

    fig, ax = plt.subplots(figsize=(10, 6))
    scree_plot(ax, pca, title="Scree Plot for Energy Principal Components")
    plt.savefig('images/pca_full.png')
    plt.close()

   


    
    print("\nRandom Forest")

    num_estimator_list = [10,20]
    train_errors_rf = []
    test_errors_rf = []
    for i, num_est in enumerate(num_estimator_list):
        print(i)
        rf = RandomForestRegressor(n_estimators = num_est, n_jobs=-1, verbose=True)
        rf.fit(full_df.X_std, full_df.y_train)
        y_pred_test =  rf.predict(full_df.Xscaler.fit_transform(full_df.X_test))
        y_pred_train =  rf.predict(full_df.Xscaler.fit_transform(full_df.X_train))
    
        train_errors_rf.append(mean_squared_error(y_pred_train, full_df.y_train)) 
        test_errors_rf.append(mean_squared_error(y_pred_test, full_df.y_test))

        train_score = rf.score(full_df.X_std, full_df.y_train)
        test_score = rf.score(full_df.Xscaler.transform(full_df.X_test), full_df.y_test)
        holdout_score = rf.score(full_df.Xscaler.transform(full_df.X_holdout), full_df.y_holdout)

        print(f"\nTrain R2: {train_score}")
        print(f"\nTest R2: {test_score}")
        print(f"\nHoldout R2: {holdout_score}")

    # plot_num_estimators_mse(num_estimator_list, train_errors_rf, test_errors_rf)
    # plt.savefig('images/rf_num_estimator_plot_short_list.png')
    # plt.close()

    



    print('\nall done.')