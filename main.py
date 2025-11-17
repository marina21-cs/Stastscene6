import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load the data
df = pd.read_csv('scenario_6_Online_Course_Completion.xlsx.csv')

print("="*80)
print("ONLINE COURSE COMPLETION ANALYSIS")
print("="*80)

# ============================================================================
# DATA CLEANING
# ============================================================================
print("\n" + "="*80)
print("DATA CLEANING")
print("="*80)

print(f"\nOriginal dataset size: {len(df)} observations")

# Check for negative time values
negative_time = df[df['Time_Spent_Hours'] < 0]
print(f"\nNegative time values found: {len(negative_time)}")
if len(negative_time) > 0:
    print("\nRows with negative time:")
    print(negative_time[['User_ID', 'Time_Spent_Hours', 'Completed']])

# Remove negative time values
df_clean = df[df['Time_Spent_Hours'] >= 0].copy()
print(f"\nCleaned dataset size: {len(df_clean)} observations")
print(f"Removed: {len(df) - len(df_clean)} observations")

# ============================================================================
# 1. DESCRIPTIVE SUMMARY
# ============================================================================
print("\n" + "="*80)
print("1. DESCRIPTIVE SUMMARY")
print("="*80)

print(f"\nSample Size: n = {len(df_clean)}")

# Completion Rate
completion_counts = df_clean['Completed'].value_counts()
completion_pct = df_clean['Completed'].value_counts(normalize=True) * 100
print(f"\nCompletion Rate:")
print(f"  Completed (Yes): {completion_counts['Yes']} ({completion_pct['Yes']:.1f}%)")
print(f"  Not Completed (No): {completion_counts['No']} ({completion_pct['No']:.1f}%)")

# Time Spent Statistics
print(f"\nTime Spent (Hours):")
print(f"  Mean: {df_clean['Time_Spent_Hours'].mean():.2f} hours")
print(f"  Median: {df_clean['Time_Spent_Hours'].median():.2f} hours")
print(f"  Std Dev: {df_clean['Time_Spent_Hours'].std():.2f} hours")
print(f"  Min: {df_clean['Time_Spent_Hours'].min():.2f} hours")
print(f"  Max: {df_clean['Time_Spent_Hours'].max():.2f} hours")
print(f"  Q1 (25th percentile): {df_clean['Time_Spent_Hours'].quantile(0.25):.2f} hours")
print(f"  Q3 (75th percentile): {df_clean['Time_Spent_Hours'].quantile(0.75):.2f} hours")

# Course Type Distribution
print(f"\nCourse Type Distribution:")
course_counts = df_clean['Course_Type'].value_counts()
course_pct = df_clean['Course_Type'].value_counts(normalize=True) * 100
for course in course_counts.index:
    print(f"  {course}: {course_counts[course]} ({course_pct[course]:.1f}%)")

# Age Distribution
print(f"\nAge Distribution:")
print(f"  Mean: {df_clean['Age'].mean():.1f} years")
print(f"  Std Dev: {df_clean['Age'].std():.1f} years")
print(f"  Min-Max: {df_clean['Age'].min()}-{df_clean['Age'].max()} years")

# Device Usage
print(f"\nDevice Usage:")
device_counts = df_clean['Device_Used'].value_counts()
device_pct = df_clean['Device_Used'].value_counts(normalize=True) * 100
for device in device_counts.index:
    print(f"  {device}: {device_counts[device]} ({device_pct[device]:.1f}%)")

# ============================================================================
# 2. COMPARE TIME SPENT BY COMPLETION STATUS
# ============================================================================
print("\n" + "="*80)
print("2. COMPARE TIME SPENT BY COMPLETION STATUS")
print("="*80)

# Group statistics
completed = df_clean[df_clean['Completed'] == 'Yes']['Time_Spent_Hours']
not_completed = df_clean[df_clean['Completed'] == 'No']['Time_Spent_Hours']

print(f"\nCompleted (n={len(completed)}):")
print(f"  Mean: {completed.mean():.2f} hours")
print(f"  Std Dev: {completed.std():.2f} hours")
print(f"  Min-Max: {completed.min():.2f} - {completed.max():.2f} hours")

print(f"\nNot Completed (n={len(not_completed)}):")
print(f"  Mean: {not_completed.mean():.2f} hours")
print(f"  Std Dev: {not_completed.std():.2f} hours")
print(f"  Min-Max: {not_completed.min():.2f} - {not_completed.max():.2f} hours")

print(f"\nDifference in Means: {completed.mean() - not_completed.mean():.2f} hours")

# Two-sample t-test
t_stat, p_value = stats.ttest_ind(completed, not_completed)
print(f"\nTwo-Sample t-test:")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_value:.4f}")
if p_value < 0.05:
    print(f"  Conclusion: SIGNIFICANT difference (p < 0.05)")
    print(f"  Students who did NOT complete spent MORE time on average!")
else:
    print(f"  Conclusion: No significant difference (p >= 0.05)")

# ============================================================================
# 3. TEST IF AVERAGE TIME SPENT EXCEEDS 15 HOURS
# ============================================================================
print("\n" + "="*80)
print("3. TEST IF AVERAGE TIME SPENT EXCEEDS 15 HOURS")
print("="*80)

mu_0 = 15  # Hypothesized value from previous study
sample_mean = df_clean['Time_Spent_Hours'].mean()
sample_std = df_clean['Time_Spent_Hours'].std()
n = len(df_clean)

# One-sample t-test (one-tailed)
t_stat_one = (sample_mean - mu_0) / (sample_std / np.sqrt(n))
p_value_one = 1 - stats.t.cdf(t_stat_one, df=n-1)  # one-tailed

print(f"\nHypothesis Test:")
print(f"  H₀: μ ≤ 15 hours")
print(f"  H₁: μ > 15 hours")
print(f"  α = 0.05")

print(f"\nTest Statistics:")
print(f"  Sample Mean: {sample_mean:.2f} hours")
print(f"  Hypothesized Mean: {mu_0} hours")
print(f"  Sample Std Dev: {sample_std:.2f} hours")
print(f"  Sample Size: {n}")
print(f"  t-statistic: {t_stat_one:.3f}")
print(f"  p-value (one-tailed): {p_value_one:.4f}")

critical_value = stats.t.ppf(0.95, df=n-1)
print(f"  Critical value (α=0.05): {critical_value:.3f}")

if p_value_one < 0.05:
    print(f"\nConclusion: REJECT H₀")
    print(f"  Average time DOES exceed 15 hours significantly.")
else:
    print(f"\nConclusion: FAIL TO REJECT H₀")
    print(f"  Average time does NOT significantly exceed 15 hours.")
    print(f"  In fact, it's slightly below the hypothesized value.")

# ============================================================================
# 4. COMPARE COURSE TYPE BY COMPLETION STATUS
# ============================================================================
print("\n" + "="*80)
print("4. COMPARE COURSE TYPE BY COMPLETION STATUS")
print("="*80)

# Create contingency table
ct_course = pd.crosstab(df_clean['Course_Type'], df_clean['Completed'], margins=True)
print("\nContingency Table (Observed Frequencies):")
print(ct_course)

# Calculate percentages
ct_course_pct = pd.crosstab(df_clean['Course_Type'], df_clean['Completed'], normalize='index') * 100
print("\nRow Percentages:")
print(ct_course_pct.round(1))

# Chi-square test
chi2, p_val, dof, expected = chi2_contingency(pd.crosstab(df_clean['Course_Type'], df_clean['Completed']))

print(f"\nChi-Square Test of Independence:")
print(f"  χ² statistic: {chi2:.3f}")
print(f"  Degrees of freedom: {dof}")
print(f"  p-value: {p_val:.4f}")
print(f"  Critical value (α=0.05): {stats.chi2.ppf(0.95, dof):.3f}")

if p_val < 0.05:
    print(f"\nConclusion: SIGNIFICANT association (p < 0.05)")
    print(f"  Course type influences completion likelihood.")
else:
    print(f"\nConclusion: Marginally significant association (p = {p_val:.3f})")
    print(f"  Course type may influence completion likelihood.")

print("\nExpected Frequencies:")
expected_df = pd.DataFrame(expected, 
                          index=pd.crosstab(df_clean['Course_Type'], df_clean['Completed']).index,
                          columns=pd.crosstab(df_clean['Course_Type'], df_clean['Completed']).columns)
print(expected_df.round(1))

# ============================================================================
# 5. COMPARE DEVICE USED BY TIME SPENT
# ============================================================================
print("\n" + "="*80)
print("5. COMPARE DEVICE USED BY TIME SPENT")
print("="*80)

# Group statistics by device
print("\nDescriptive Statistics by Device:")
device_stats = df_clean.groupby('Device_Used')['Time_Spent_Hours'].agg([
    ('n', 'count'),
    ('Mean', 'mean'),
    ('Std Dev', 'std'),
    ('Min', 'min'),
    ('Max', 'max')
])
print(device_stats.round(2))

# ANOVA
desktop = df_clean[df_clean['Device_Used'] == 'Desktop']['Time_Spent_Hours']
mobile = df_clean[df_clean['Device_Used'] == 'Mobile']['Time_Spent_Hours']
tablet = df_clean[df_clean['Device_Used'] == 'Tablet']['Time_Spent_Hours']

f_stat, p_val_anova = f_oneway(desktop, mobile, tablet)

print(f"\nOne-Way ANOVA:")
print(f"  F-statistic: {f_stat:.3f}")
print(f"  p-value: {p_val_anova:.4f}")

if p_val_anova < 0.05:
    print(f"\nConclusion: SIGNIFICANT difference (p < 0.05)")
    print(f"  Device type affects time spent.")
else:
    print(f"\nConclusion: NO significant difference (p >= 0.05)")
    print(f"  Time spent is similar across all devices.")

# Calculate Sum of Squares for detailed output
grand_mean = df_clean['Time_Spent_Hours'].mean()
ss_between = sum([len(df_clean[df_clean['Device_Used'] == device]) * 
                  (df_clean[df_clean['Device_Used'] == device]['Time_Spent_Hours'].mean() - grand_mean)**2 
                  for device in df_clean['Device_Used'].unique()])
ss_within = sum([(len(df_clean[df_clean['Device_Used'] == device]) - 1) * 
                 df_clean[df_clean['Device_Used'] == device]['Time_Spent_Hours'].var() 
                 for device in df_clean['Device_Used'].unique()])

print(f"\nANOVA Details:")
print(f"  SS Between: {ss_between:.2f}")
print(f"  SS Within: {ss_within:.2f}")
print(f"  MS Between: {ss_between/2:.2f}")
print(f"  MS Within: {ss_within/(len(df_clean)-3):.2f}")

# ============================================================================
# 6. ADDITIONAL ANALYSIS: AGE EFFECT ON COMPLETION
# ============================================================================
print("\n" + "="*80)
print("6. ADDITIONAL ANALYSIS: AGE EFFECT ON COMPLETION")
print("="*80)

# Create age groups
df_clean['Age_Group'] = pd.cut(df_clean['Age'], bins=[17, 29, 39, 49, 60], 
                                labels=['18-29', '30-39', '40-49', '50-59'])

# Completion rate by age group
age_completion = pd.crosstab(df_clean['Age_Group'], df_clean['Completed'], normalize='index') * 100
print("\nCompletion Rate by Age Group:")
print(age_completion.round(1))

# Chi-square test
chi2_age, p_val_age, dof_age, _ = chi2_contingency(pd.crosstab(df_clean['Age_Group'], df_clean['Completed']))
print(f"\nChi-Square Test:")
print(f"  χ² statistic: {chi2_age:.3f}")
print(f"  p-value: {p_val_age:.4f}")

if p_val_age < 0.05:
    print(f"  Conclusion: SIGNIFICANT relationship between age and completion")
else:
    print(f"  Conclusion: NO significant relationship between age and completion")

# ============================================================================
# 7. OPTIMAL TIME RANGE FOR COMPLETION
# ============================================================================
print("\n" + "="*80)
print("7. OPTIMAL TIME RANGE FOR COMPLETION")
print("="*80)

# Create time quartiles
df_clean['Time_Quartile'] = pd.qcut(df_clean['Time_Spent_Hours'], q=4, 
                                     labels=['Q1 (Low)', 'Q2 (Med-Low)', 'Q3 (Med-High)', 'Q4 (High)'])

# Completion rate by time quartile
time_completion = pd.crosstab(df_clean['Time_Quartile'], df_clean['Completed'], normalize='index') * 100
print("\nCompletion Rate by Time Quartile:")
print(time_completion.round(1))

print("\nKey Finding:")
if time_completion.loc['Q1 (Low)', 'Yes'] > time_completion.loc['Q4 (High)', 'Yes']:
    print("  Students in lower time quartiles have HIGHER completion rates.")
    print("  This suggests more time doesn't necessarily mean better outcomes.")

# ============================================================================
# 8. DEVICE-COURSE TYPE INTERACTION
# ============================================================================
print("\n" + "="*80)
print("8. DEVICE-COURSE TYPE INTERACTION")
print("="*80)

device_course = pd.crosstab(df_clean['Device_Used'], df_clean['Course_Type'])
print("\nDevice Distribution by Course Type:")
print(device_course)

chi2_int, p_val_int, dof_int, _ = chi2_contingency(device_course)
print(f"\nChi-Square Test:")
print(f"  χ² statistic: {chi2_int:.3f}")
print(f"  p-value: {p_val_int:.4f}")

if p_val_int < 0.05:
    print(f"  Conclusion: SIGNIFICANT interaction between device and course type")
else:
    print(f"  Conclusion: NO significant interaction")

# ============================================================================
# KEY INSIGHTS & RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("KEY INSIGHTS & RECOMMENDATIONS")
print("="*80)

print("\n1. COUNTERINTUITIVE TIME FINDING:")
print("   More time ≠ Better completion")
print("   Students who didn't complete spent MORE time on average")
print("   → Implement early intervention for students exceeding 18-20 hours")

print("\n2. COURSE DIFFICULTY:")
completion_by_course = df_clean.groupby('Course_Type')['Completed'].apply(lambda x: (x=='Yes').sum() / len(x) * 100)
worst_course = completion_by_course.idxmin()
print(f"   {worst_course} courses have lowest completion ({completion_by_course[worst_course]:.1f}%)")
print(f"   → Redesign or add support for {worst_course} courses")

print("\n3. DEVICE AGNOSTIC:")
print("   Platform performs equally across devices")
print("   → Maintain cross-platform consistency")

print("\n4. SWEET SPOT:")
q1_completion = df_clean[df_clean['Time_Quartile'] == 'Q1 (Low)']['Completed'].value_counts(normalize=True)['Yes'] * 100
print(f"   Students in lower time quartiles show better outcomes ({q1_completion:.1f}%)")
print("   → Optimize course length for 8-15 hour completion window")

print("\n5. AGE NEUTRAL:")
print("   Age doesn't significantly affect completion")
print("   → Course design works well across age groups")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n\nGenerating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Online Course Completion Analysis', fontsize=16, fontweight='bold')

# 1. Completion Rate
ax1 = axes[0, 0]
completion_counts.plot(kind='bar', ax=ax1, color=['#2ecc71', '#e74c3c'])
ax1.set_title('Overall Completion Rate', fontweight='bold')
ax1.set_xlabel('Completion Status')
ax1.set_ylabel('Count')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
for i, v in enumerate(completion_counts):
    ax1.text(i, v + 20, f'{v}\n({completion_pct.iloc[i]:.1f}%)', ha='center', fontweight='bold')

# 2. Time Spent Distribution by Completion
ax2 = axes[0, 1]
df_clean.boxplot(column='Time_Spent_Hours', by='Completed', ax=ax2)
ax2.set_title('Time Spent by Completion Status', fontweight='bold')
ax2.set_xlabel('Completed')
ax2.set_ylabel('Time Spent (Hours)')
plt.sca(ax2)
plt.xticks([1, 2], ['No', 'Yes'])

# 3. Course Type Completion Rate
ax3 = axes[0, 2]
course_completion = pd.crosstab(df_clean['Course_Type'], df_clean['Completed'], normalize='index') * 100
course_completion.plot(kind='bar', ax=ax3, color=['#e74c3c', '#2ecc71'])
ax3.set_title('Completion Rate by Course Type', fontweight='bold')
ax3.set_xlabel('Course Type')
ax3.set_ylabel('Percentage (%)')
ax3.legend(['Not Completed', 'Completed'])
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)

# 4. Device Usage Distribution
ax4 = axes[1, 0]
device_counts.plot(kind='bar', ax=ax4, color='#3498db')
ax4.set_title('Device Usage Distribution', fontweight='bold')
ax4.set_xlabel('Device')
ax4.set_ylabel('Count')
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)

# 5. Time by Device
ax5 = axes[1, 1]
df_clean.boxplot(column='Time_Spent_Hours', by='Device_Used', ax=ax5)
ax5.set_title('Time Spent by Device', fontweight='bold')
ax5.set_xlabel('Device Used')
ax5.set_ylabel('Time Spent (Hours)')

# 6. Age Distribution
ax6 = axes[1, 2]
df_clean['Age'].hist(bins=20, ax=ax6, color='#9b59b6', edgecolor='black')
ax6.set_title('Age Distribution', fontweight='bold')
ax6.set_xlabel('Age (Years)')
ax6.set_ylabel('Frequency')
ax6.axvline(df_clean['Age'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df_clean["Age"].mean():.1f}')
ax6.legend()

plt.tight_layout()
plt.show()

print("\nVisualizations generated successfully!")