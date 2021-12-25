import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.stats as stats
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
import seaborn as sns



class survival_analyze():
     def __init__(self, data):
        self.data = data


    """[Initialize an instance of survival_analyze class that will be used to plot and analyze various 
    data and plot results 
    args:
        data: cleaned dataframe of  METABRIC clinical data
    ]
    """
   
    def plot_survival(self, column, value):
        """[plot KDE survival plots of cleaned METABRIC clinical data]

        Args:
            column ([string]): [column in METABRIC data corresponding to a patient attribute, such as her2 receptor
            status]
            value ([string or integer]): [value of column that is a point of comparision. ie column:her2_recepter value:'negative']
        Plots values in column vs != values in column
        """
        # df = data[data['type_of_breast_surgery']=='BREAST CONSERVING'] 
        treatment_df = self.data[self.data[column]==value]
        not_treatment_df = self.data[self.data[column]!=value]
        treatment_months = treatment_df.overall_survival_months
        not_treatment_months = not_treatment_df.overall_survival_months
        x = treatment_months
        y = not_treatment_months
        sns.kdeplot(x, color="green", shade=True, label = value)
        sns.kdeplot(y, color="blue", shade=True, label = f'Not {value}')
        plt.legend()
        plt.show()

    def plot_kaplan_meier(self, column, value):
             """[plot Kaplan meier survival plots of cleaned METABRIC clinical data]

        Args:
            column ([string]): [column in METABRIC data corresponding to a patient attribute, such as her2 receptor
            status]
            value ([string or integer]): [value of column that is a point of comparision. ie column:her2_recepter value:'negative']
        Plots values in column vs != values in column
        """
        kmf = KaplanMeierFitter()
        treatment_df = self.data[self.data[column]==value]
        not_treatment_df = self.data[self.data[column]!=value]
        treatment_months = treatment_df.overall_survival_months
        not_treatment_months = not_treatment_df.overall_survival_months

        kmf.fit(treatment_months, event_observed=treatment_df['death_from_cancer'], label = value)
        ax = kmf.plot()

        kmf2 = KaplanMeierFitter()
        kmf2.fit(not_treatment_months, event_observed=not_treatment_df['death_from_cancer'], label = f'not {value}')
        ax = kmf2.plot(ax=ax)
        add_at_risk_counts(kmf, kmf2, ax=ax)
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel('Timeline (Months)')
        ax.set_title(f'Kaplan Meier plot in months of {column} variable')
        # plt.figure(dpi=350)
        plt.tight_layout()
        plt.show()
    def diagnosis_age(self, column, value):
        """[gets average age of patients rounded to 2 decimal places with column attribute and value]

        Args:
            column ([string]): [column of desired patient attribute ie her2 receptor status]
            value ([string or integer]): [value of column of average age desired]

        Returns:
            [float]: [average age of patients with column at the value attribute 
            * returned value is not rounded printed value is rounded to 2 decimal places.]
        """
        treatment_df = self.data[self.data[column]==value]
        not_treatment_df = self.data[self.data[column]!=value]
        treatment_months = treatment_df.overall_survival_months
        not_treatment_months = not_treatment_df.overall_survival_months
        # print(f'{treatment_df['age_at_diagnosis'].mean()}')
        avg_age = treatment_df['age_at_diagnosis'].mean()
        print(f'{round(avg_age, 2)} years old')

        return treatment_df['age_at_diagnosis'].mean()



if __name__ == "__main__":
    mrna_df =pd.read_csv('/Users/cp/Desktop/Capstone2b/capstone2.mrn_df2.csv')
    df = pd.read_csv('/Users/cp/data/METABRIC_RNA_Mutation.csv')
    death_from_dict = {
    'Living':0
    ,'Died of Other Causes':0
    ,'Died of Disease':1
    }
    df.replace(death_from_dict, inplace =True)
    mrna_df['death_from_cancer'] = df.death_from_cancer
    df.death_from_cancer.fillna(0, inplace = True)
    mrna_df.death_from_cancer.fillna(0, inplace = True)

    # breast_conserving_df['death_from_cancer'].fillna(0, inplace = True)

    test = survival_analyze(df)
    # test.plot_survival('type_of_breast_surgery', 'BREAST CONSERVING')
    # test.plot_survival('chemotherapy', 1)
    # test.plot_survival('hormone_therapy', 1)
    # test.plot_survival('her2_status', 'Negative')

    # test.plot_kaplan_meier('type_of_breast_surgery', 'BREAST CONSERVING')
    # test.plot_kaplan_meier('chemotherapy', 1)
    # test.plot_kaplan_meier('her2_status', 'Negative')
    test.plot_kaplan_meier('type_of_breast_surgery', 'MASTECTOMY')
    # test.diagnosis_age('type_of_breast_surgery', 'BREAST CONSERVING')






        