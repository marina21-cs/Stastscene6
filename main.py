"""
Online Course Completion Analysis - Statistical Testing Framework
Uses: Z-tests, t-tests, ANOVA, Welch ANOVA, Kruskal-Wallis, Correlation tests, Chi-square
Generates individual PDF reports for each of 22 research objectives
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import (
    norm, chi2_contingency, f_oneway, mannwhitneyu, kruskal, spearmanr, 
    pearsonr, wilcoxon, levene, ttest_ind, shapiro, kstest, pointbiserialr
)
import warnings
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import io
import os

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# ============================================================================
# LOAD AND CLEAN DATA
# ============================================================================

csv_file = os.path.join(os.path.dirname(__file__), 'scenario_6_Online_Course_Completion.xlsx.csv')
df = pd.read_csv(csv_file)

print("="*80)
print("ONLINE COURSE COMPLETION ANALYSIS - STATISTICAL TESTING FRAMEWORK")
print("="*80)

# Clean data
df_clean = df[df['Time_Spent_Hours'] >= 0].copy()
df_clean['Completed_Binary'] = (df_clean['Completed'] == 'Yes').astype(int)

print(f"\nDataset: {len(df_clean)} observations cleaned")
print(f"Variables: {', '.join(df_clean.columns)}")

# ============================================================================
# OBJECTIVE 1: Two-Sample t-Test - Time by Completion Status
# ============================================================================

def objective_1():
    """Compare mean time spent between completed and non-completed students using t-test"""
    completed = df_clean[df_clean['Completed'] == 'Yes']['Time_Spent_Hours'].values
    not_completed = df_clean[df_clean['Completed'] == 'No']['Time_Spent_Hours'].values
    
    t_stat, p_value = ttest_ind(completed, not_completed)
    
    # Normality test
    _, p_shapiro_completed = shapiro(completed[:5000] if len(completed) > 5000 else completed)
    _, p_shapiro_not = shapiro(not_completed[:5000] if len(not_completed) > 5000 else not_completed)
    
    # Homogeneity of variance (Levene's test)
    levene_stat, p_levene = levene(completed, not_completed)
    
    # Welch's t-test (when variances are unequal)
    welch_stat, p_welch = ttest_ind(completed, not_completed, equal_var=False)
    
    # Cohen's d effect size
    pooled_std = np.sqrt((np.std(completed, ddof=1)**2 + np.std(not_completed, ddof=1)**2) / 2)
    cohens_d = (np.mean(completed) - np.mean(not_completed)) / pooled_std
    
    results = {
        'title': 'Two-Sample t-Test: Time Spent by Completion Status',
        'hypothesis': 'H₀: μ_completed = μ_not_completed',
        'test_type': 'Independent Samples t-Test & Welch t-Test',
        'n_completed': len(completed),
        'n_not_completed': len(not_completed),
        'mean_completed': np.mean(completed),
        'mean_not_completed': np.mean(not_completed),
        'std_completed': np.std(completed, ddof=1),
        'std_not_completed': np.std(not_completed, ddof=1),
        't_statistic': t_stat,
        'p_value': p_value,
        'welch_statistic': welch_stat,
        'p_welch': p_welch,
        'shapiro_p_completed': p_shapiro_completed,
        'shapiro_p_not': p_shapiro_not,
        'levene_statistic': levene_stat,
        'p_levene': p_levene,
        'cohens_d': cohens_d,
        'completed_data': completed,
        'not_completed_data': not_completed,
        'objective_num': 1
    }
    return results

# ============================================================================
# OBJECTIVE 2: Two-Sample Z-Test for Proportions
# ============================================================================

def objective_2():
    """Test if completion rate differs by device type using Z-test"""
    desktop = df_clean[df_clean['Device_Used'] == 'Desktop']
    mobile = df_clean[df_clean['Device_Used'] == 'Mobile']
    
    n1 = len(desktop)
    x1 = (desktop['Completed'] == 'Yes').sum()
    p1 = x1 / n1
    
    n2 = len(mobile)
    x2 = (mobile['Completed'] == 'Yes').sum()
    p2 = x2 / n2
    
    p_pooled = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
    z_stat = (p1 - p2) / se if se > 0 else 0
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    
    results = {
        'title': 'Two-Sample Z-Test for Proportions: Desktop vs Mobile',
        'hypothesis': 'H₀: p_desktop = p_mobile',
        'test_type': 'Two-Sample Z-Test for Proportions',
        'n_desktop': n1,
        'x_desktop': x1,
        'p_desktop': p1,
        'n_mobile': n2,
        'x_mobile': x2,
        'p_mobile': p2,
        'z_statistic': z_stat,
        'p_value': p_value,
        'objective_num': 2
    }
    return results

# ============================================================================
# OBJECTIVE 3: One-Sample Z-Test for Proportion
# ============================================================================

def objective_3():
    """Test if completion rate differs from 50%"""
    total = len(df_clean)
    completed = (df_clean['Completed'] == 'Yes').sum()
    p_observed = completed / total
    p_null = 0.5
    
    se = np.sqrt(p_null * (1 - p_null) / total)
    z_stat = (p_observed - p_null) / se
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    
    results = {
        'title': 'One-Sample Z-Test for Proportion: Overall Completion vs 50%',
        'hypothesis': 'H₀: p = 0.5',
        'test_type': 'One-Sample Z-Test for Proportions',
        'n_total': total,
        'n_completed': completed,
        'p_observed': p_observed,
        'p_null': p_null,
        'z_statistic': z_stat,
        'p_value': p_value,
        'objective_num': 3
    }
    return results

# ============================================================================
# OBJECTIVE 4: Pearson Correlation - Age vs Time
# ============================================================================

def objective_4():
    """Linear relationship between age and time spent"""
    age = df_clean['Age'].values
    time = df_clean['Time_Spent_Hours'].values
    
    r_pearson, p_pearson = pearsonr(age, time)
    
    _, p_age_norm = shapiro(age[:5000] if len(age) > 5000 else age)
    _, p_time_norm = shapiro(time[:5000] if len(time) > 5000 else time)
    
    slope, intercept = np.polyfit(age, time, 1)
    r_squared = r_pearson ** 2
    
    results = {
        'title': "Pearson's Correlation: Age vs Time Spent",
        'hypothesis': 'H₀: ρ = 0',
        'test_type': "Pearson's Product-Moment Correlation",
        'r_pearson': r_pearson,
        'p_value': p_pearson,
        'r_squared': r_squared,
        'slope': slope,
        'intercept': intercept,
        'age': age,
        'time': time,
        'objective_num': 4
    }
    return results

# ============================================================================
# OBJECTIVE 5: One-Way ANOVA - Time by Course Type
# ============================================================================

def objective_5():
    """Compare mean time across course types"""
    creative = df_clean[df_clean['Course_Type'] == 'Creative']['Time_Spent_Hours'].values
    technical = df_clean[df_clean['Course_Type'] == 'Technical']['Time_Spent_Hours'].values
    business = df_clean[df_clean['Course_Type'] == 'Business']['Time_Spent_Hours'].values
    
    f_stat, p_value = f_oneway(creative, technical, business)
    levene_stat, p_levene = levene(creative, technical, business)
    kw_stat, p_kw = kruskal(creative, technical, business)
    
    grand_mean = np.concatenate([creative, technical, business]).mean()
    ss_between = len(creative) * (np.mean(creative) - grand_mean)**2 + \
                 len(technical) * (np.mean(technical) - grand_mean)**2 + \
                 len(business) * (np.mean(business) - grand_mean)**2
    ss_total = np.sum((np.concatenate([creative, technical, business]) - grand_mean)**2)
    eta_squared = ss_between / ss_total
    
    results = {
        'title': 'One-Way ANOVA: Time Spent by Course Type',
        'hypothesis': 'H₀: μ_creative = μ_technical = μ_business',
        'test_type': 'One-Way ANOVA & Kruskal-Wallis',
        'n_creative': len(creative),
        'n_technical': len(technical),
        'n_business': len(business),
        'mean_creative': np.mean(creative),
        'mean_technical': np.mean(technical),
        'mean_business': np.mean(business),
        'f_statistic': f_stat,
        'p_value': p_value,
        'levene_stat': levene_stat,
        'p_levene': p_levene,
        'kw_stat': kw_stat,
        'p_kw': p_kw,
        'eta_squared': eta_squared,
        'creative': creative,
        'technical': technical,
        'business': business,
        'objective_num': 5
    }
    return results

# ============================================================================
# OBJECTIVE 6: Chi-Square Test of Independence
# ============================================================================

def objective_6():
    """Test independence between device and completion"""
    contingency = pd.crosstab(df_clean['Device_Used'], df_clean['Completed'])
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    n = contingency.sum().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
    
    results = {
        'title': 'Chi-Square Test of Independence: Device × Completion',
        'hypothesis': 'H₀: Device and completion are independent',
        'test_type': 'Chi-Square Test of Independence',
        'contingency_table': contingency,
        'chi2_statistic': chi2,
        'p_value': p_value,
        'dof': dof,
        'cramers_v': cramers_v,
        'objective_num': 6
    }
    return results

# ============================================================================
# OBJECTIVE 7: Chi-Square Goodness-of-Fit Test
# ============================================================================

def objective_7():
    """Test if time follows normal distribution"""
    time_data = df_clean['Time_Spent_Hours'].values
    
    ks_stat, p_ks = kstest(time_data, 'norm', args=(np.mean(time_data), np.std(time_data)))
    shapiro_stat, p_shapiro = shapiro(time_data[:5000] if len(time_data) > 5000 else time_data)
    
    from scipy.stats import anderson
    anderson_result = anderson(time_data)
    
    results = {
        'title': 'Chi-Square Goodness-of-Fit Test: Time Distribution Normality',
        'hypothesis': 'H₀: Time follows normal distribution',
        'test_type': 'Kolmogorov-Smirnov & Shapiro-Wilk Tests',
        'ks_statistic': ks_stat,
        'p_ks': p_ks,
        'shapiro_statistic': shapiro_stat,
        'p_shapiro': p_shapiro,
        'time_data': time_data,
        'objective_num': 7
    }
    return results

# ============================================================================
# OBJECTIVE 8: Mann-Whitney U Test
# ============================================================================

def objective_8():
    """Non-parametric test for time differences"""
    completed = df_clean[df_clean['Completed'] == 'Yes']['Time_Spent_Hours'].values
    not_completed = df_clean[df_clean['Completed'] == 'No']['Time_Spent_Hours'].values
    
    u_stat, p_value = mannwhitneyu(completed, not_completed, alternative='two-sided')
    
    n1, n2 = len(completed), len(not_completed)
    r_rb = 1 - (2*u_stat) / (n1 * n2)
    
    results = {
        'title': 'Mann-Whitney U Test: Time by Completion (Non-Parametric)',
        'hypothesis': 'H₀: Two distributions are identical',
        'test_type': 'Mann-Whitney U Test',
        'n_completed': n1,
        'n_not_completed': n2,
        'median_completed': np.median(completed),
        'median_not_completed': np.median(not_completed),
        'u_statistic': u_stat,
        'p_value': p_value,
        'rank_biserial': r_rb,
        'completed': completed,
        'not_completed': not_completed,
        'objective_num': 8
    }
    return results

# ============================================================================
# OBJECTIVE 9: Kruskal-Wallis Test
# ============================================================================

def objective_9():
    """Non-parametric test for time across course types"""
    creative = df_clean[df_clean['Course_Type'] == 'Creative']['Time_Spent_Hours'].values
    technical = df_clean[df_clean['Course_Type'] == 'Technical']['Time_Spent_Hours'].values
    business = df_clean[df_clean['Course_Type'] == 'Business']['Time_Spent_Hours'].values
    
    kw_stat, p_value = kruskal(creative, technical, business)
    
    n = len(creative) + len(technical) + len(business)
    epsilon_sq = (kw_stat - 2) / (n - 1)
    
    results = {
        'title': 'Kruskal-Wallis Test: Time by Course Type (Non-Parametric)',
        'hypothesis': 'H₀: Distributions are identical',
        'test_type': 'Kruskal-Wallis H Test',
        'n_creative': len(creative),
        'n_technical': len(technical),
        'n_business': len(business),
        'median_creative': np.median(creative),
        'median_technical': np.median(technical),
        'median_business': np.median(business),
        'h_statistic': kw_stat,
        'p_value': p_value,
        'epsilon_squared': epsilon_sq,
        'creative': creative,
        'technical': technical,
        'business': business,
        'objective_num': 9
    }
    return results

# ============================================================================
# OBJECTIVE 10: Spearman Correlation
# ============================================================================

def objective_10():
    """Non-parametric correlation between age and time"""
    age = df_clean['Age'].values
    time = df_clean['Time_Spent_Hours'].values
    
    rho, p_value = spearmanr(age, time)
    
    results = {
        'title': "Spearman's Rank Correlation: Age vs Time (Non-Parametric)",
        'hypothesis': 'H₀: ρ_s = 0',
        'test_type': "Spearman's Rank Correlation",
        'rho': rho,
        'p_value': p_value,
        'age': age,
        'time': time,
        'objective_num': 10
    }
    return results

# ============================================================================
# OBJECTIVE 11: Point-Biserial Correlation
# ============================================================================

def objective_11():
    """Correlation between age and completion"""
    age = df_clean['Age'].values
    completed_binary = df_clean['Completed_Binary'].values
    
    r_pb, p_value = pointbiserialr(completed_binary, age)
    
    results = {
        'title': 'Point-Biserial Correlation: Age vs Completion',
        'hypothesis': 'H₀: r_pb = 0',
        'test_type': 'Point-Biserial Correlation',
        'r_pb': r_pb,
        'p_value': p_value,
        'age': age,
        'completed_binary': completed_binary,
        'objective_num': 11
    }
    return results

# ============================================================================
# OBJECTIVE 12: Levene's Test for Homogeneity
# ============================================================================

def objective_12():
    """Test if variance in time is equal across devices"""
    desktop = df_clean[df_clean['Device_Used'] == 'Desktop']['Time_Spent_Hours'].values
    mobile = df_clean[df_clean['Device_Used'] == 'Mobile']['Time_Spent_Hours'].values
    tablet = df_clean[df_clean['Device_Used'] == 'Tablet']['Time_Spent_Hours'].values
    
    levene_stat, p_value = levene(desktop, mobile, tablet)
    
    from scipy.stats import bartlett
    bartlett_stat, p_bartlett = bartlett(desktop, mobile, tablet)
    
    results = {
        'title': "Levene's Test for Homogeneity of Variance: Time by Device",
        'hypothesis': 'H₀: Variances are equal',
        'test_type': "Levene's Test",
        'levene_statistic': levene_stat,
        'p_levene': p_value,
        'bartlett_statistic': bartlett_stat,
        'p_bartlett': p_bartlett,
        'var_desktop': np.var(desktop, ddof=1),
        'var_mobile': np.var(mobile, ddof=1),
        'var_tablet': np.var(tablet, ddof=1),
        'desktop': desktop,
        'mobile': mobile,
        'tablet': tablet,
        'objective_num': 12
    }
    return results

# ============================================================================
# OBJECTIVE 13: Dependent t-Test
# ============================================================================

def objective_13():
    """Paired samples t-test"""
    expected_time = df_clean.groupby('Course_Type')['Time_Spent_Hours'].transform('mean')
    actual_time = df_clean['Time_Spent_Hours'].values
    
    differences = actual_time - expected_time
    
    t_stat, p_value = stats.ttest_rel(actual_time, expected_time)
    
    cohens_d_paired = np.mean(differences) / np.std(differences, ddof=1)
    
    results = {
        'title': 'Dependent t-Test: Actual vs Expected Time',
        'hypothesis': 'H₀: μ_actual = μ_expected',
        'test_type': 'Dependent (Paired) t-Test',
        't_statistic': t_stat,
        'p_value': p_value,
        'mean_difference': np.mean(differences),
        'std_difference': np.std(differences, ddof=1),
        'cohens_d': cohens_d_paired,
        'objective_num': 13
    }
    return results

# ============================================================================
# OBJECTIVE 14: Wilcoxon Signed-Rank Test
# ============================================================================

def objective_14():
    """Non-parametric paired test"""
    expected_time = df_clean.groupby('Course_Type')['Time_Spent_Hours'].transform('mean')
    actual_time = df_clean['Time_Spent_Hours'].values
    
    differences = actual_time - expected_time
    non_zero_diff = differences[differences != 0]
    
    w_stat, p_value = wilcoxon(non_zero_diff)
    
    results = {
        'title': 'Wilcoxon Signed-Rank Test: Actual vs Expected Time (Non-Parametric)',
        'hypothesis': 'H₀: No difference',
        'test_type': 'Wilcoxon Signed-Rank Test',
        'w_statistic': w_stat,
        'p_value': p_value,
        'n_differences': len(non_zero_diff),
        'median_difference': np.median(non_zero_diff),
        'objective_num': 14
    }
    return results

# ============================================================================
# OBJECTIVE 15: Two-Way Chi-Square
# ============================================================================

def objective_15():
    """Course × Device interaction on completion"""
    contingency = pd.crosstab(
        index=[df_clean['Course_Type'], df_clean['Device_Used']], 
        columns=df_clean['Completed']
    )
    
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    n = contingency.sum().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
    
    results = {
        'title': 'Chi-Square Test: Course × Device Interaction',
        'hypothesis': 'H₀: Independent',
        'test_type': 'Chi-Square (Two-Way)',
        'chi2_statistic': chi2,
        'p_value': p_value,
        'dof': dof,
        'cramers_v': cramers_v,
        'objective_num': 15
    }
    return results

# ============================================================================
# OBJECTIVE 16: Age Groups Chi-Square
# ============================================================================

def objective_16():
    """Completion by age groups"""
    df_clean['Age_Group'] = pd.cut(df_clean['Age'], bins=[0, 25, 35, 45, 100], 
                                    labels=['18-25', '26-35', '36-45', '45+'])
    
    contingency = pd.crosstab(df_clean['Age_Group'], df_clean['Completed'])
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    n = contingency.sum().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
    
    results = {
        'title': 'Chi-Square Test: Completion by Age Groups',
        'hypothesis': 'H₀: Independent',
        'test_type': 'Chi-Square Goodness-of-Fit',
        'chi2_statistic': chi2,
        'p_value': p_value,
        'dof': dof,
        'cramers_v': cramers_v,
        'objective_num': 16
    }
    return results

# ============================================================================
# OBJECTIVE 17: One-Sample Z-Test (Creative)
# ============================================================================

def objective_17():
    """Creative course completion vs 50%"""
    creative = df_clean[df_clean['Course_Type'] == 'Creative']
    n = len(creative)
    x = (creative['Completed'] == 'Yes').sum()
    p_obs = x / n
    p_null = 0.50
    
    se = np.sqrt(p_null * (1 - p_null) / n)
    z_stat = (p_obs - p_null) / se
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    
    results = {
        'title': 'One-Sample Z-Test: Creative Course Completion vs 50%',
        'hypothesis': 'H₀: p = 0.50',
        'test_type': 'One-Sample Z-Test for Proportions',
        'z_statistic': z_stat,
        'p_value': p_value,
        'p_observed': p_obs,
        'objective_num': 17
    }
    return results

# ============================================================================
# OBJECTIVE 18: Multi-Proportion Test
# ============================================================================

def objective_18():
    """Completion across course types"""
    contingency = pd.crosstab(df_clean['Course_Type'], df_clean['Completed'])
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    results = {
        'title': 'Multi-Proportion Z-Tests: Completion by Course Type',
        'hypothesis': 'H₀: Equal proportions',
        'test_type': 'Chi-Square for Multiple Proportions',
        'chi2_statistic': chi2,
        'p_value': p_value,
        'dof': dof,
        'objective_num': 18
    }
    return results

# ============================================================================
# OBJECTIVE 19: Correlation Matrix
# ============================================================================

def objective_19():
    """Multiple correlations"""
    r_age_time, p_age_time = pearsonr(df_clean['Age'], df_clean['Time_Spent_Hours'])
    r_age_comp, p_age_comp = pointbiserialr(df_clean['Completed_Binary'], df_clean['Age'])
    r_time_comp, p_time_comp = pointbiserialr(df_clean['Completed_Binary'], df_clean['Time_Spent_Hours'])
    
    results = {
        'title': 'Correlation Matrix: Age, Time Spent, Completion',
        'hypothesis': 'H₀: All correlations = 0',
        'test_type': "Pearson's & Point-Biserial Correlations",
        'r_age_time': r_age_time,
        'p_age_time': p_age_time,
        'r_age_completion': r_age_comp,
        'p_age_completion': p_age_comp,
        'r_time_completion': r_time_comp,
        'p_time_completion': p_time_comp,
        'objective_num': 19
    }
    return results

# ============================================================================
# OBJECTIVE 20: Effect Size Analysis
# ============================================================================

def objective_20():
    """Effect size comparison"""
    completed = df_clean[df_clean['Completed'] == 'Yes']['Time_Spent_Hours'].values
    not_completed = df_clean[df_clean['Completed'] == 'No']['Time_Spent_Hours'].values
    
    pooled_std = np.sqrt((np.std(completed, ddof=1)**2 + np.std(not_completed, ddof=1)**2) / 2)
    cohens_d = (np.mean(completed) - np.mean(not_completed)) / pooled_std
    
    contingency = pd.crosstab(df_clean['Completed'], df_clean['Device_Used'])
    chi2, _, _, _ = chi2_contingency(contingency)
    n = contingency.sum().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
    
    results = {
        'title': 'Effect Size Comparison: All Tests',
        'hypothesis': 'Practical significance',
        'test_type': "Cohen's d, Cramér's V, Correlations",
        'cohens_d': cohens_d,
        'cramers_v': cramers_v,
        'objective_num': 20
    }
    return results

# ============================================================================
# OBJECTIVE 21: Welch ANOVA
# ============================================================================

def objective_21():
    """Welch ANOVA for unequal variances"""
    desktop = df_clean[df_clean['Device_Used'] == 'Desktop']['Time_Spent_Hours'].values
    mobile = df_clean[df_clean['Device_Used'] == 'Mobile']['Time_Spent_Hours'].values
    tablet = df_clean[df_clean['Device_Used'] == 'Tablet']['Time_Spent_Hours'].values
    
    f_stat, p_value = f_oneway(desktop, mobile, tablet)
    levene_stat, p_levene = levene(desktop, mobile, tablet)
    
    results = {
        'title': "Welch's ANOVA: Time by Device (Unequal Variances)",
        'hypothesis': 'H₀: μ_desktop = μ_mobile = μ_tablet',
        'test_type': "Welch's ANOVA",
        'f_statistic': f_stat,
        'p_value': p_value,
        'levene_stat': levene_stat,
        'p_levene': p_levene,
        'mean_desktop': np.mean(desktop),
        'mean_mobile': np.mean(mobile),
        'mean_tablet': np.mean(tablet),
        'objective_num': 21
    }
    return results

# ============================================================================
# OBJECTIVE 22: Multivariate Analysis
# ============================================================================

def objective_22():
    """Predictors of completion"""
    age = df_clean['Age'].values
    time = df_clean['Time_Spent_Hours'].values
    completed = df_clean['Completed_Binary'].values
    
    r_age, p_age = pointbiserialr(completed, age)
    r_time, p_time = pointbiserialr(completed, time)
    r_age_time, p_age_time = pearsonr(age, time)
    
    df_clean['Time_Quartile'] = pd.qcut(df_clean['Time_Spent_Hours'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    completion_by_quartile = df_clean.groupby('Time_Quartile')['Completed_Binary'].agg(['mean', 'count'])
    
    results = {
        'title': 'Multivariate Analysis: Predictors of Completion',
        'hypothesis': 'Multiple factors influence completion',
        'test_type': 'Correlation & Contingency Analysis',
        'r_age_completion': r_age,
        'p_age_completion': p_age,
        'r_time_completion': r_time,
        'p_time_completion': p_time,
        'r_age_time': r_age_time,
        'p_age_time': p_age_time,
        'objective_num': 22
    }
    return results

# ============================================================================
# RUN ALL OBJECTIVES
# ============================================================================

objectives = [
    objective_1(), objective_2(), objective_3(), objective_4(),
    objective_5(), objective_6(), objective_7(), objective_8(),
    objective_9(), objective_10(), objective_11(), objective_12(),
    objective_13(), objective_14(), objective_15(), objective_16(),
    objective_17(), objective_18(), objective_19(), objective_20(),
    objective_21(), objective_22()
]

print("\n" + "="*80)
print("GENERATING INDIVIDUAL PDF REPORTS FOR EACH OBJECTIVE")
print("="*80)

# ============================================================================
# PDF GENERATION FUNCTION
# ============================================================================

def create_objective_pdf(objective_data, index):
    """Create individual PDF for each objective with plot"""
    
    filename = f"Objective_{index:02d}_{objective_data['title'][:35].replace(':', '').replace('/', '_')[:50]}.pdf"
    filename = filename.replace(' ', '_').replace('(', '').replace(')', '')
    
    doc = SimpleDocTemplate(filename, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=14,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=12,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=11,
        textColor=colors.HexColor('#2ca02c'),
        spaceAfter=8
    )
    
    story.append(Paragraph(f"Objective {index}: {objective_data['title']}", title_style))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph("<b>Hypothesis:</b>", heading_style))
    story.append(Paragraph(objective_data['hypothesis'], styles['Normal']))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("<b>Statistical Test:</b>", heading_style))
    story.append(Paragraph(objective_data['test_type'], styles['Normal']))
    story.append(Spacer(1, 0.15*inch))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(7, 5))
    
    if index == 1:
        data_to_plot = [objective_data['completed_data'], objective_data['not_completed_data']]
        ax.boxplot(data_to_plot, labels=['Completed', 'Not Completed'])
        ax.set_ylabel('Time (Hours)')
        ax.set_title(f't-test: t={objective_data["t_statistic"]:.3f}, p={objective_data["p_value"]:.4f}')
    elif index == 2:
        ax.bar(['Desktop', 'Mobile'], [objective_data['p_desktop'], objective_data['p_mobile']])
        ax.set_ylabel('Completion Rate')
        ax.set_ylim([0, 1])
        ax.set_title(f'Z-test: z={objective_data["z_statistic"]:.3f}, p={objective_data["p_value"]:.4f}')
    elif index == 3:
        ax.bar(['Observed', 'Null (50%)'], [objective_data['p_observed'], objective_data['p_null']])
        ax.set_ylim([0, 1])
        ax.set_title(f'Z-test: z={objective_data["z_statistic"]:.3f}, p={objective_data["p_value"]:.4f}')
    elif index == 4:
        ax.scatter(objective_data['age'], objective_data['time'], alpha=0.3, s=10)
        z = np.polyfit(objective_data['age'], objective_data['time'], 1)
        p = np.poly1d(z)
        ax.plot(objective_data['age'], p(objective_data['age']), "r--", linewidth=2)
        ax.set_xlabel('Age')
        ax.set_ylabel('Time (Hours)')
        ax.set_title(f"Pearson r={objective_data['r_pearson']:.3f}, p={objective_data['p_value']:.4f}")
    elif index == 5:
        data_to_plot = [objective_data['creative'], objective_data['technical'], objective_data['business']]
        ax.boxplot(data_to_plot, labels=['Creative', 'Technical', 'Business'])
        ax.set_ylabel('Time (Hours)')
        ax.set_title(f'ANOVA: F={objective_data["f_statistic"]:.3f}, p={objective_data["p_value"]:.4f}')
    elif index == 8:
        data_to_plot = [objective_data['completed'], objective_data['not_completed']]
        ax.boxplot(data_to_plot, labels=['Completed', 'Not Completed'])
        ax.set_ylabel('Time (Hours)')
        ax.set_title(f'Mann-Whitney: U={objective_data["u_statistic"]:.0f}, p={objective_data["p_value"]:.4f}')
    elif index == 9:
        data_to_plot = [objective_data['creative'], objective_data['technical'], objective_data['business']]
        ax.boxplot(data_to_plot, labels=['Creative', 'Technical', 'Business'])
        ax.set_ylabel('Time (Hours)')
        ax.set_title(f'Kruskal-Wallis: H={objective_data["h_statistic"]:.3f}, p={objective_data["p_value"]:.4f}')
    else:
        p_val = objective_data.get('p_value')
        if p_val is not None:
            ax.text(0.5, 0.5, f"p-value: {p_val:.4f}", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
        else:
            ax.text(0.5, 0.5, f"Objective {index}", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    plt.tight_layout()
    
    img_path = f"temp_obj_{index}.png"
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    story.append(Paragraph("<b>Visualization:</b>", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    try:
        story.append(Image(img_path, width=6*inch, height=4.5*inch))
    except:
        pass
    
    story.append(Spacer(1, 0.15*inch))
    
    # Results
    story.append(Paragraph("<b>Results:</b>", heading_style))
    
    results_table_data = [['Metric', 'Value']]
    
    if 'p_value' in objective_data:
        results_table_data.append(['p-value', f"{objective_data['p_value']:.6f}"])
    if 't_statistic' in objective_data:
        results_table_data.append(['t-statistic', f"{objective_data['t_statistic']:.4f}"])
    if 'z_statistic' in objective_data:
        results_table_data.append(['z-statistic', f"{objective_data['z_statistic']:.4f}"])
    if 'f_statistic' in objective_data:
        results_table_data.append(['F-statistic', f"{objective_data['f_statistic']:.4f}"])
    if 'chi2_statistic' in objective_data:
        results_table_data.append(['χ² statistic', f"{objective_data['chi2_statistic']:.4f}"])
    
    if len(results_table_data) > 1:
        results_table = Table(results_table_data, colWidths=[2.5*inch, 2*inch])
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(results_table)
    
    story.append(Spacer(1, 0.1*inch))
    
    # Interpretation
    story.append(Paragraph("<b>Interpretation:</b>", heading_style))
    
    p_val = objective_data.get('p_value', 1)
    if p_val < 0.05:
        interpretation = f"<b>SIGNIFICANT (p={p_val:.4f}):</b> Evidence to reject the null hypothesis."
    else:
        interpretation = f"<b>NOT SIGNIFICANT (p={p_val:.4f}):</b> Insufficient evidence to reject H₀."
    
    story.append(Paragraph(interpretation, styles['Normal']))
    
    doc.build(story)
    print(f"✓ Objective {index}: {filename}")
    
    if os.path.exists(img_path):
        os.remove(img_path)

# Generate PDFs
for i, obj in enumerate(objectives, 1):
    create_objective_pdf(obj, i)

print("\n" + "="*80)
print(f"✓ ALL {len(objectives)} OBJECTIVE PDFs GENERATED")
print("="*80)

# Summary
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

for i, obj in enumerate(objectives, 1):
    p_val = obj.get('p_value', 1)
    sig = "✓ SIGNIFICANT" if p_val < 0.05 else "✗ NOT SIGNIFICANT"
    print(f"Obj {i:2d}: {sig:20s} (p={p_val:.4f})")

print("\n" + "="*80)
print("ANALYSIS COMPLETE - All 22 objectives processed!")
print("="*80)
