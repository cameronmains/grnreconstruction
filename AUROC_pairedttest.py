import numpy as np
import pandas as pd
from scipy import stats
#model_aurocs = np.random.random(size=[100,1])
model_aurocs = np.random.normal(loc=0.8, scale=0.2, size=[100,1])
model_aurocs = np.clip(model_aurocs, 0, 1)

#model_aurocs = np.load(r"C:\Users\crmai\OneDrive\Documents\GT Spring 2025\BMED 6517\target_auroc.npy")
excel_data = pd.read_excel(r"C:\Users\crmai\OneDrive\Documents\GT Spring 2025\BMED 6517\Project1_updated_021125\baseline_use_correlation.xlsx", 
                            sheet_name=2,  # 0-based index, so 2 is the 3rd sheet
                            #skiprows=1,
                            usecols="B",  # Column range
                            nrows=100)      # Number of rows
baseline_aurocs = excel_data.to_numpy()

print(model_aurocs.shape)
#print(baseline_aurocs.shape)
print(model_aurocs)
#print(baseline_aurocs)

def paired_ttest(array1, array2):
    
    # Perform the paired t-test
    t_stat, p_value = stats.ttest_rel(array1, array2)

    # Print results
    print(f"t-statistic: {t_stat}")
    print(f"p-value: {p_value}")

    # Interpret results
    alpha = 0.05
    if float(p_value) < alpha:
        print("Significant difference found")
    else:
        print("No significant difference")


if __name__ == "__main__":
    # Call with None to use random data for demonstration
    paired_ttest(model_aurocs, baseline_aurocs)
