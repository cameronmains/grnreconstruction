import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
#model_aurocs = np.random.random(size=[100,1])
#model_aurocs = np.random.normal(loc=0.8, scale=0.2, size=[100,1])
#model_aurocs = np.clip(model_aurocs, 0, 1)

#model_aurocs = np.load(r"C:\Users\crmai\OneDrive\Documents\GT Spring 2025\BMED 6517\target_auroc.npy")
excel_data = pd.read_excel(r"C:\Users\crmai\OneDrive\Documents\GT Spring 2025\BMED 6517\Project1_updated_021125\baseline_use_correlation.xlsx", 
                            sheet_name=0,  # 0-based index, so 2 is the 3rd sheet
                            #skiprows=1,
                            usecols="B",  # Column range
                            nrows=100)      # Number of rows
baseline_aurocs = excel_data.to_numpy()

model_aurocs = np.loadtxt(r"C:\Users\crmai\OneDrive\Documents\GT Spring 2025\BMED 6517\Project1_updated_021125\auc_scores_pearson_5.csv", delimiter=',', ndmin=2)

#print(model_aurocs.shape)
#print(baseline_aurocs.shape)
#print(model_aurocs)
#print(baseline_aurocs)

def paired_ttest(array1, array2):
    
    # Perform the paired t-test
    t_stat, p_value = stats.ttest_rel(array1, array2)
    std_dev_baseline = np.std(array1)
    std_dev_model = np.std(array2)

    # Print results
    print(f"t-statistic: {t_stat}")
    print(f"p-value: {p_value}")
    print(f"standard deviation b: {std_dev_baseline}")
    print(f"standard deviation m: {std_dev_model}")

    # Interpret results
    alpha = 0.05
    if float(p_value) < alpha:
        print("Significant difference found")
    else:
        print("No significant difference")


def box_whisker(array1, array2):

    array1_flat = array1.flatten()
    array2_flat = array2.flatten()

    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(array1, array2)
    print(f"t-statistic: {t_stat}")
    print(f"p-value: {p_value}")

    # Create a box and whisker plot
    fig, ax = plt.subplots(figsize=(3, 6))

    # Create the boxplot
    #box_data = [array1, array2]
    box_plot = ax.boxplot([array1_flat, array2_flat], patch_artist=True, labels=['Baseline', 'Model'])

    # Customize boxplot colors
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)

    # Add title and labels
    plt.title('Random Forest 100', fontsize=15)
    plt.ylabel('AUROC', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add t-test results as annotation
    p_text = f"p-value: {p_value}"
    t_text = f"t-statistic: {t_stat}"
    test_result = "Significant difference" if p_value < 0.05 else "No significant difference"
    plt.figtext(0.6, -0.05, f"{t_text}\n{p_text}\n{test_result}", 
                ha='center', fontsize=10, bbox={"facecolor":"orange", "alpha":0.2})

    # Show plot
    plt.tight_layout(pad=3)
    plt.savefig('paired_ttest_boxplot.png')  # Save the figure
    plt.show()

if __name__ == "__main__":
    # Call with None to use random data for demonstration
    paired_ttest(baseline_aurocs, model_aurocs)
    box_whisker(baseline_aurocs, model_aurocs)
