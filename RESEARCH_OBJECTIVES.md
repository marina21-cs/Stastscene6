# Research Objectives & Statistical Tests
## Online Course Completion Analysis

This document outlines the research objectives added to the analysis, including both parametric and non-parametric statistical tests.

---

## Objective 1: Compare Time Spent by Completion Status (Parametric & Non-Parametric)
**Research Question:** Do students who complete courses spend a different amount of time than those who don't complete?

**Methods:**
- **Parametric:** Two-sample t-test
- **Non-Parametric:** Mann-Whitney U Test
- **Effect Size:** Rank-biserial correlation

**Key Findings:**
- Completed students: Mean = 15.16 hrs, Median = 15.19 hrs
- Not completed students: Mean = 15.22 hrs, Median = 15.29 hrs
- Mann-Whitney U: U = 543006, p = 0.7307 (not significant)
- Effect size (r_rb) = 0.009 (negligible)

**Interpretation:** No significant difference in time distributions between completed and non-completed students.

---

## Objective 2: Relationship Between Age and Time Spent (Non-Parametric Correlation)
**Research Question:** Is there a correlation between student age and time spent on courses?

**Methods:**
- **Non-Parametric:** Spearman's Rank Correlation (ρ)
- **Assumptions:** No normality assumption required

**Key Findings:**
- Spearman's ρ = -0.0388, p = 0.0760
- Correlation strength: Very weak negative (but not significant)
- Students slightly decrease time as age increases (not significant)

**Interpretation:** Age does not significantly correlate with time spent on courses.

---

## Objective 3: Comparison of Time Spent Across Course Types (Non-Parametric)
**Research Question:** Do students spend different amounts of time on different course types?

**Methods:**
- **Parametric:** One-way ANOVA
- **Non-Parametric:** Kruskal-Wallis H Test

**Key Findings:**
- Creative courses: Mean = 15.32 hrs, Median = 15.59 hrs (n=666)
- Technical courses: Mean = 15.09 hrs, Median = 14.82 hrs (n=710)
- Business courses: Mean = 15.17 hrs, Median = 15.25 hrs (n=719)
- Kruskal-Wallis H = 1.571, p = 0.4560 (not significant)

**Interpretation:** Time distributions are similar across all course types.

---

## Objective 4: Age and Completion Status Relationship (Point-Biserial Correlation)
**Research Question:** Does age predict course completion likelihood?

**Methods:**
- **Point-Biserial Correlation:** Non-parametric measure for continuous-binary relationship
- **Chi-Square Test:** For age groups vs. completion status

**Key Findings:**
- Point-biserial r = -0.0018, p = 0.9345
- Chi-square by age group: χ² = 0.937, p = 0.8166
- All age groups show similar completion rates (~47-50%)

**Interpretation:** Age has no significant effect on course completion likelihood.

---

## Objective 5: Effect Size Analysis
**Research Question:** What is the practical significance of observed differences?

**Methods:**
- **Cohen's d:** For continuous variable differences (time spent)
- **Cramer's V:** For categorical associations

**Key Findings:**
- Cohen's d (completion status) = -0.0134 (negligible effect)
- Cramer's V (course type × completion) = 0.0200 (negligible association)
- Interpretation thresholds:
  - |d| < 0.2: Negligible
  - 0.2-0.5: Small
  - 0.5-0.8: Medium
  - > 0.8: Large

**Interpretation:** While some tests show statistical trends, the practical significance is minimal.

---

## Objective 6: Device Type and Completion Status Association
**Research Question:** Does the device used influence course completion?

**Methods:**
- **Chi-Square Test of Independence:** Categorical association test
- **Effect Size:** Cramer's V

**Key Findings:**
- Desktop completion: 47.4%
- Mobile completion: 48.6%
- Tablet completion: 48.1%
- χ² = 0.187, p = 0.9107 (not significant)
- Cramer's V = 0.0094 (negligible effect)

**Interpretation:** Device type does not significantly influence completion likelihood.

---

## Objective 7: Device Type and Time Spent (Parametric & Non-Parametric)
**Research Question:** Do students spend different amounts of time on different devices?

**Methods:**
- **Parametric:** One-way ANOVA
- **Non-Parametric:** Kruskal-Wallis H Test

**Key Findings:**
- Desktop: Mean = 15.17 hrs, Median = 15.17 hrs
- Mobile: Mean = 15.03 hrs, Median = 15.16 hrs
- Tablet: Mean = 15.35 hrs, Median = 15.31 hrs
- ANOVA: F = 0.697, p = 0.4984
- Kruskal-Wallis H = 0.549, p = 0.7599

**Interpretation:** No significant differences in time spent across devices.

---

## Objective 8: Completion Efficiency Analysis
**Research Question:** What is the optimal time range for course completion?

**Methods:**
- **Decile Analysis:** Stratified completion rates by time spent
- **Efficiency Metric:** Completion percentage per hour

**Key Findings:**
- Optimal time range: 15.2-16.4 hours (51.7% completion rate)
- Most efficient course type: Technical (3.26% completion per hour)
- Less efficient: Creative (3.05% completion per hour)

**Interpretation:** 
- Sweet spot appears to be 15-16 hours of study
- Extremes (very low or very high time) show lower completion
- Technical courses maximize completion efficiency

---

## Objective 9: Device Preference and Time Relationship (Non-Parametric Correlation)
**Research Question:** Does device preference correlate with time spent?

**Methods:**
- **Spearman's Rank Correlation:** Non-parametric ordinal correlation

**Key Findings:**
- Device encoding: Desktop(1) < Mobile(2) < Tablet(3)
- Spearman's ρ = 0.0118, p = 0.5907
- No significant correlation

**Interpretation:** Device type ranking does not correlate with time spent.

---

## Summary Table: Statistical Tests Used

| Objective | Parametric Test | Non-Parametric Test | Purpose |
|-----------|-----------------|-------------------|---------|
| Time × Completion | t-test | Mann-Whitney U | Compare groups |
| Age × Time | Pearson r | Spearman ρ | Correlation |
| Time × Course Type | ANOVA | Kruskal-Wallis H | Multiple groups |
| Age × Completion | - | Point-biserial r | Binary-continuous |
| Device × Time | ANOVA | Kruskal-Wallis H | Multiple groups |
| Device × Completion | - | Chi-square | Categorical assoc. |
| Effect sizes | Cohen's d | Cramer's V | Practical significance |

---

## Key Research Insights

1. **Time Paradox:** More time spent does NOT guarantee completion. Optimal range is 15-16 hours.

2. **Device Neutral:** Platform performs consistently across Desktop, Mobile, and Tablet.

3. **Age Independent:** Age is not a significant predictor of completion or study time.

4. **Course Type Effect:** Minor differences exist (Creative courses: 46.7% vs Technical: 49.2%), but not statistically significant.

5. **Efficiency Focus:** Technical courses show best efficiency (completion % per hour).

---

## Statistical Test Selection Rationale

### Why Non-Parametric Tests?
- **Robustness:** Don't assume normal distribution
- **Median Focus:** Better for skewed data
- **Ranks:** Based on data order, not values
- **Distribution-Free:** More conservative, fewer assumptions

### When Used:
1. When sample data may violate normality assumption
2. When dealing with ordinal or ranked data
3. For robustness checks against parametric results
4. When non-normal distributions are suspected
