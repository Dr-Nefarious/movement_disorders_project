# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt



def assignStage(stage_value):
    if pd.isna(stage_value):
        return np.nan
    elif stage_value <= 2:
        return "early"
    elif stage_value == 3:
        return "middle"
    else:
        return "late"

def main():
    
    # Research question: Do early vs middle vs late patients progress differently?
    # Do people in early, middle, and late Parkinson’s get worse at different speeds over time?
    # This means: 
    #    time = MONTH
    #    group = STAGE
    #    We care about = whether the time slope is different across stages
    
    # Read in master dataframe 
    master_df = pd.read_csv(r'kelseysinclair/Downloads/master_df.csv', sep = ',')
    
    # Assign each patient to a stage based on baseline stage only
    # This is because if NHY changes over time and you assign stage at every row, then a patient could be “early” at one visit and “middle” later
    
    # Step 1: Get baseline/first measurement for each person
    baseline = master_df.sort_values("MONTH").groupby("PATNO").first().reset_index()

    # Step 2: Assign stage BASED ON BASELINE ONLY
    # Create column stating which patient belongs to which stage
    # Turn the numeric disease stage into a categorical group label for use in statistical test
    baseline["Stage"] = baseline["NHY"].apply(assignStage)
    baseline["Stage"] = pd.Categorical(baseline["Stage"], categories=["early", "middle", "late"], ordered = True)
    
    # Step 3: Keep only PATNO and baseline stage
    baseline_stage = baseline[["PATNO", "Stage"]]

    # Step 4: merge back into full dataset
    merged_df = master_df.merge(baseline_stage, on="PATNO", how="left")
    
    # Run Linear Mixed-Effects Model (LMM) on data:
        # A regression model that accounts for repeated measurements from the same people
        # Each person is measured multiple times over time
        # So measurements woud not be independent
        
    # MONTH = time passing
    # BRADY (or other score) = how bad symptoms are
    # Stage = which group someone belongs to
    
    # Test whether slopes differ by stage.
    # Use early as the reference group
    brady_df = merged_df.dropna(subset=["BRADY", "MONTH", "PATNO", "Stage"])

    model = smf.mixedlm(
        "BRADY ~ MONTH * C(Stage, Treatment(reference='early'))",
        data = brady_df,
        groups = brady_df["PATNO"],  # rows from the same PATNO belong to the same patient
        re_formula = "~MONTH"
    )
    
    result = model.fit(method = "lbfgs", maxiter=200)
    print(result.summary())
    
    # Results: For bradykinesia   
    # Early-stage patients do get worse over time
    # Rate = 0.076 units per month
    
    # Computing the slopes: Early: 0.076, Middle: 0.076 - 0.019 = 0.057, Late: 0.076 - 0.041 = 0.035
    # The differences are not statistically significant
    # The rate of progression of bradykinesia did not differ significantly across early, middle, and late stages of Parkinson’s disease.
    
    
    # Middle and late patients start with higher symptom severity
    # But they don’t worsen faster
    
    # Plot output of results
    sns.set(style="whitegrid")
    
    plt.figure()
    
    sns.lineplot(
        data=merged_df,
        x="MONTH",
        y="BRADY",
        hue="Stage",
        estimator="mean",   # average at each time point
        ci=95               # confidence interval
    )

    plt.title("Bradykinesia Progression Over Time by Stage")
    plt.xlabel("Months")
    plt.ylabel("BRADY Score")
    
    plt.show()
    
    # Test the other symptoms
    symptom_list = ["BRADY", "NP3TOT", "RIGIDITY", "PIGD", "TREMOR"]
    
    results_list = []
    
    for symptom in symptom_list:
        df = merged_df.dropna(subset=[symptom, "MONTH", "PATNO", "Stage"])
    
        formula = f"{symptom} ~ MONTH * C(Stage, Treatment(reference='early'))"
    
        model = smf.mixedlm(
            formula,
            data=df,
            groups=df["PATNO"],
            re_formula="~MONTH"
        )
    
        result = model.fit(method="lbfgs", maxiter=200)
    
        fe = result.fe_params
    
        early = fe["MONTH"]
        mid = early + fe["MONTH:C(Stage, Treatment(reference='early'))[T.middle]"]
        late = early + fe["MONTH:C(Stage, Treatment(reference='early'))[T.late]"]
    
        p_mid = result.pvalues["MONTH:C(Stage, Treatment(reference='early'))[T.middle]"]
        p_late = result.pvalues["MONTH:C(Stage, Treatment(reference='early'))[T.late]"]
    
        results_list.append({
            "Symptom": symptom,
            "Early_slope": early,
            "Middle_slope": mid,
            "Late_slope": late,
            "p_middle_vs_early": p_mid,
            "p_late_vs_early": p_late
        })
    
    results_df = pd.DataFrame(results_list)
    print(results_df)
    
    # Results:
        # Most symptoms -> same progression rate across stages
        # Except for tremor: p (middle vs early) = 0.0306 (significant)
        # The lack of significant differences across most subscores suggests that while symptom 
        # severity varies by stage, the rate of progression may remain relatively consistent across 
        # disease stages.
    
    # Generate plots to double check data
    sns.set(style="whitegrid")
    
    for symptom in symptom_list:
        plt.figure()
    
        sns.lineplot(
            data=merged_df,
            x="MONTH",
            y=symptom,
            hue="Stage",
            estimator="mean",
            ci=95
        )
    
        plt.title(f"{symptom} Progression Over Time by Stage")
        plt.xlabel("Months")
        plt.ylabel(f"{symptom} Score")
    
        plt.legend(title="Stage")
        plt.show()
        
if __name__ == "__main__":
    main()