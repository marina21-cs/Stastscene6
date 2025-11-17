# Online Course Completion - Full Hypothesis Testing Script (R)
# Save as: online_course_tests.R
# Run in R (RStudio). Assumes dataset is: /mnt/data/scenario_6_Online_Course_Completion.xlsx.csv
# Outputs plots to: /mnt/data/plots/

# -------------------------
# Libraries
# -------------------------
if(!require(tidyverse)) install.packages("tidyverse", repos="https://cloud.r-project.org"); library(tidyverse)
if(!require(car)) install.packages("car", repos="https://cloud.r-project.org"); library(car)       # Levene
if(!require(rstatix)) install.packages("rstatix", repos="https://cloud.r-project.org"); library(rstatix) # normals
if(!require(DescTools)) install.packages("DescTools", repos="https://cloud.r-project.org"); library(DescTools) # z-tests
if(!require(ggpubr)) install.packages("ggpubr", repos="https://cloud.r-project.org"); library(ggpubr)
if(!require(broom)) install.packages("broom", repos="https://cloud.r-project.org"); library(broom)

# -------------------------
# Settings & helpers
# -------------------------
data_path <- "/mnt/data/Online_Course_Completion.xlsx.csv"
plots_dir <- "/mnt/data/plots"
dir.create(plots_dir, showWarnings = FALSE, recursive = TRUE)

# small helper: two-sample z-test for means (large-sample approx)
z_test_mean <- function(x, mu = 0, sigma = NULL, alternative = c("two.sided","less","greater")) {
  alternative <- match.arg(alternative)
  if(is.null(sigma)) stop("z_test_mean requires 'sigma' (population sd) or use a t-test instead.")
  n <- length(x)
  xbar <- mean(x, na.rm=TRUE)
  se <- sigma / sqrt(n)
  z <- (xbar - mu) / se
  p <- if(alternative=="two.sided") 2*pnorm(-abs(z)) else if(alternative=="less") pnorm(z) else 1-pnorm(z)
  list(z=z, p.value=p, mean=xbar, n=n)
}

# two-sample z-test for proportions (counts)
z_test_prop <- function(x1, n1, x2, n2, alternative = c("two.sided","less","greater")) {
  alternative <- match.arg(alternative)
  p1 <- x1/n1; p2 <- x2/n2
  p_pool <- (x1 + x2) / (n1 + n2)
  se <- sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2))
  z <- (p1 - p2) / se
  p <- if(alternative=="two.sided") 2*pnorm(-abs(z)) else if(alternative=="less") pnorm(z) else 1-pnorm(z)
  list(z=z, p.value=p, p1=p1, p2=p2)
}

# -------------------------
# Load & initial cleaning
# -------------------------
df <- read.csv(data_path, stringsAsFactors = FALSE)

# column detection (change names here if needed)
time_col <- if("Time_Spent_Hours" %in% names(df)) "Time_Spent_Hours" else names(df)[grepl("time", names(df), ignore.case=TRUE)][1]
completion_col <- if("Completed" %in% names(df)) "Completed" else names(df)[grepl("complete|status", names(df), ignore.case=TRUE)][1]
course_col <- if("Course_Type" %in% names(df)) "Course_Type" else names(df)[grepl("course|type", names(df), ignore.case=TRUE)][1]
device_col <- if("Device_Used" %in% names(df)) "Device_Used" else names(df)[grepl("device|platform", names(df), ignore.case=TRUE)][1]
age_col <- if("Age" %in% names(df)) "Age" else names(df)[grepl("^age", names(df), ignore.case=TRUE)][1]

cat("Columns used:\n")
cat("time_col:", time_col, "\ncompletion_col:", completion_col, "\ncourse_col:", course_col, "\ndevice_col:", device_col, "\nage_col:", age_col, "\n\n")

# convert types
df[[time_col]] <- as.numeric(df[[time_col]])
df[[age_col]] <- as.numeric(df[[age_col]])
df[[completion_col]] <- as.character(df[[completion_col]])
df[[course_col]] <- as.character(df[[course_col]])
df[[device_col]] <- as.character(df[[device_col]])

# remove negative time values (per your instruction)
neg_count <- sum(df[[time_col]] < 0, na.rm = TRUE)
df <- df %>% mutate(!!time_col := ifelse((!!sym(time_col)) < 0, NA, !!sym(time_col)))
cat("Negative time values removed:", neg_count, "\n\n")

# quick descriptive summary
cat("Descriptive summary (time):\n")
print(df %>% summarise(n = sum(!is.na(!!sym(time_col))),
                       mean = mean(!!sym(time_col), na.rm=TRUE),
                       median = median(!!sym(time_col), na.rm=TRUE),
                       sd = sd(!!sym(time_col), na.rm=TRUE),
                       min = min(!!sym(time_col), na.rm=TRUE),
                       max = max(!!sym(time_col), na.rm=TRUE)))

# counts for completion
cat("\nCompletion counts:\n")
print(table(df[[completion_col]], useNA="ifany"))

# -------------------------
# PLOTS
# -------------------------
# histogram of time
p1 <- ggplot(df, aes_string(x = time_col)) +
  geom_histogram(bins=30) + labs(title="Histogram of Time Spent (hours)", x="Time Spent (hours)", y="Count")
ggsave(file.path(plots_dir,"hist_time_spent.png"), p1, width=7, height=4)

# boxplot: time by completion
p2 <- ggplot(df, aes_string(x = completion_col, y = time_col)) +
  geom_boxplot() + labs(title="Time Spent by Completion Status", x=completion_col, y="Time Spent (hours)")
ggsave(file.path(plots_dir,"box_time_by_completion.png"), p2, width=7, height=4)

# boxplot: time by device
if(!is.null(device_col)) {
  p3 <- ggplot(df, aes_string(x = device_col, y = time_col)) +
    geom_boxplot() + labs(title="Time Spent by Device", x=device_col, y="Time Spent (hours)") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  ggsave(file.path(plots_dir,"box_time_by_device.png"), p3, width=8, height=4)
}

# bar: course type by completion
if(!is.null(course_col)) {
  p4 <- df %>% filter(!is.na(!!sym(course_col))) %>%
    ggplot(aes_string(x = course_col, fill = completion_col)) +
    geom_bar(position="dodge") + labs(title="Course Type × Completion", x="Course Type", y="Count")
  ggsave(file.path(plots_dir,"bar_course_by_completion.png"), p4, width=7, height=4)
}

# scatter: age vs time with regression
if(!is.null(age_col)) {
  p5 <- ggplot(df, aes_string(x=age_col, y=time_col)) +
    geom_point(alpha=0.4) + geom_smooth(method="lm", se=TRUE) +
    labs(title="Age vs Time Spent", x="Age", y="Time Spent (hours)")
  ggsave(file.path(plots_dir,"scatter_age_time.png"), p5, width=6, height=4)
}

cat("Plots saved to:", plots_dir, "\n\n")

# -------------------------
# NORMALITY & HOMOGENEITY
# -------------------------
cat("Normality tests (Shapiro-Wilk) where applicable:\n")
time_non_na <- df %>% filter(!is.na(!!sym(time_col)))
if(nrow(time_non_na) >= 3 & nrow(time_non_na) <= 5000) {
  sh <- shapiro_test(time_non_na, !!sym(time_col))
  print(sh)
} else {
  cat("Shapiro overall: sample size not in 3..5000; Skipping or rely on CLT for large n\n")
}

# By group normality
if(!is.null(completion_col)) {
  cat("\nShapiro by completion group:\n")
  print(df %>% group_by(!!sym(completion_col)) %>% summarise(n = sum(!is.na(!!sym(time_col))), sh_p = ifelse(n>=3 & n<=5000, shapiro.test(!!sym(time_col))$p.value, NA)))
}

# Levene (homogeneity) between completion groups
if(!is.null(completion_col)) {
  lev <- leveneTest(as.formula(paste(time_col, "~", completion_col)), data=df)
  cat("\nLevene test for equality of variances (by completion):\n"); print(lev)
}

# -------------------------
# 1) Two Independent Samples Z-test (for proportions demonstration)
#    We'll also show two-sample z for means if a population sigma is provided.
# -------------------------
# Example: Two-sample z-test for proportions (Completed proportion by below/above-average age)
if(!is.null(age_col) & !is.null(completion_col)) {
  avg_age <- mean(df[[age_col]], na.rm=TRUE)
  df <- df %>% mutate(below_avg_age = ifelse(!is.na(!!sym(age_col)) & !!sym(age_col) < avg_age, "Below", "AboveOrEqual"))
  tab_age_comp <- table(df$below_avg_age, df[[completion_col]])
  cat("\nContingency (below/above avg age × completion):\n"); print(tab_age_comp)
  # detect which value indicates completion; try common labels
  completion_vals <- unique(df[[completion_col]])
  complete_label <- completion_vals[grep("yes|complete|1|true", tolower(as.character(completion_vals)))]
  if(length(complete_label)==0) complete_label <- completion_vals[1]
  x1 <- tab_age_comp["Below", complete_label]
  n1 <- sum(tab_age_comp["Below", ])
  x2 <- tab_age_comp["AboveOrEqual", complete_label]
  n2 <- sum(tab_age_comp["AboveOrEqual", ])
  cat("\nTwo-sample z-test for proportions (H1: below > above):\n")
  zprop <- z_test_prop(x1, n1, x2, n2, alternative = "greater")
  print(zprop)
}

# -------------------------
# 2) Independent t-Test (time by completion)
# -------------------------
cat("\nIndependent t-test (Welch) - Time by Completion:\n")
if(!is.null(completion_col)) {
  t_res <- t.test(as.formula(paste(time_col,"~",completion_col)), data=df, var.equal = FALSE)
  print(t_res)
  # Nonparametric alternative - Mann-Whitney
  mw <- wilcox.test(as.formula(paste(time_col,"~",completion_col)), data = df, exact = FALSE)
  cat("\nMann-Whitney U test (two-sample) as nonparametric alternative:\n"); print(mw)
}

# -------------------------
# Paired/Dependent t-test & Wilcoxon signed-rank
# -------------------------
# These require paired columns (e.g., Time_Before, Time_After). Check for common names
paired_candidates <- names(df)[grepl("pre|post|before|after|time1|time2", names(df), ignore.case=TRUE)]
if(length(paired_candidates) >= 2) {
  # pick first two candidates (this is heuristic)
  a <- paired_candidates[1]; b <- paired_candidates[2]
  cat("\nDetected potential paired columns:", a, "and", b, "\n")
  paired_df <- df %>% filter(!is.na(!!sym(a)) & !is.na(!!sym(b)))
  if(nrow(paired_df) > 2) {
    pt <- t.test(paired_df[[a]], paired_df[[b]], paired = TRUE)
    cat("Paired t-test result:\n"); print(pt)
    wpt <- wilcox.test(paired_df[[a]], paired_df[[b]], paired = TRUE, exact = FALSE)
    cat("Wilcoxon signed-rank test (paired):\n"); print(wpt)
  } else {
    cat("Not enough paired data for paired tests.\n")
  }
} else {
  cat("\nNo obvious paired columns found; skipping dependent t / Wilcoxon signed-rank.\n")
}

# -------------------------
# ANOVA, Welch ANOVA, Kruskal-Wallis (Device × Time)
# -------------------------
if(!is.null(device_col)) {
  df_dev <- df %>% filter(!is.na(!!sym(device_col)) & !is.na(!!sym(time_col)))
  if(nrow(df_dev) > 2) {
    cat("\nOne-way ANOVA (Time ~ Device):\n")
    aov_res <- aov(as.formula(paste(time_col,"~",device_col)), data = df_dev)
    print(summary(aov_res))
    cat("\nWelch ANOVA (using oneway.test):\n")
    print(oneway.test(as.formula(paste(time_col,"~",device_col)), data = df_dev, var.equal = FALSE))
    cat("\nKruskal-Wallis (nonparametric):\n")
    print(kruskal.test(as.formula(paste(time_col,"~",device_col)), data = df_dev))
  }
}

# -------------------------
# Part 4: Tests Concerning Proportions
# 1) One-sample z-test for proportion (e.g., is completion rate = p0?)
# 2) Two-sample z-test for proportion (we did above)
# -------------------------
# One-sample test: test if completion rate equals 0.5 for example
if(!is.null(completion_col)) {
  comp_tab <- table(df[[completion_col]])
  # determine completion label as before
  completion_vals <- names(comp_tab)
  complete_label <- completion_vals[grep("yes|complete|1|true", tolower(completion_vals))]
  if(length(complete_label)==0) complete_label <- completion_vals[1]
  x <- comp_tab[complete_label]
  n <- sum(comp_tab)
  p0 <- 0.5  # change benchmark as needed
  # z statistic:
  p_hat <- x/n
  se_p0 <- sqrt(p0*(1-p0)/n)
  z_stat <- (p_hat - p0) / se_p0
  p_val <- 2 * pnorm(-abs(z_stat))
  cat("\nOne-sample z-test for proportion (H0: p = ", p0, "):\n", sep="")
  cat("completed count:", x, "n:", n, "p_hat:", round(p_hat,4), "\n")
  cat("z:", round(z_stat,4), "p(two-sided):", round(p_val,4), "\n")
}

# -------------------------
# Correlation: Age vs Time
# -------------------------
if(!is.null(age_col)) {
  df_corr <- df %>% filter(!is.na(!!sym(age_col)) & !is.na(!!sym(time_col)))
  if(nrow(df_corr) > 2) {
    cat("\nPearson correlation (Age vs Time):\n")
    print(cor.test(df_corr[[age_col]], df_corr[[time_col]], method = "pearson"))
    cat("\nSpearman correlation (Age vs Time):\n")
    print(cor.test(df_corr[[age_col]], df_corr[[time_col]], method = "spearman"))
  }
}

# -------------------------
# Chi-square tests
# 1) Goodness-of-fit (example: are course types equally distributed?)
# 2) Test of independence (Course_Type × Completion)
# -------------------------
if(!is.null(course_col)) {
  tab_course <- table(df[[course_col]])
  cat("\nChi-square goodness-of-fit (H0: equal proportions across course types):\n")
  expected <- rep(sum(tab_course)/length(tab_course), length(tab_course))
  print(chisq.test(tab_course, p = rep(1/length(tab_course), length(tab_course))))
}

if(!is.null(course_col) & !is.null(completion_col)) {
  cat("\nChi-square test of independence: Course_Type × Completion:\n")
  ct <- table(df[[course_col]], df[[completion_col]])
  print(ct)
  print(chisq.test(ct))
}

# -------------------------
# Test if average time spent exceeds 15 hours (Your key objective #3)
# H0: mu = 15; H1: mu > 15  (one-sided)
# -------------------------
cat("\nOne-sample t-test: H0: mu = 15 vs H1: mu > 15\n")
time_vec <- df[[time_col]][!is.na(df[[time_col]])]
t1 <- t.test(time_vec, mu = 15, alternative = "greater")
print(t1)
cat("Interpretation: if p < 0.05, reject H0 and conclude mean > 15 hours.\n")

# If you want a one-sample z-test for mean and you have a population sigma estimate:
# sigma_est <- sd(time_vec) # (but usually unknown) - use with caution
# zres <- z_test_mean(time_vec, mu=15, sigma = sigma_est, alternative="greater")

# -------------------------
# Additional outputs: effect sizes & tidy results
# -------------------------
# Cohen's d for time by completion
if(!is.null(completion_col)) {
  df_c <- df %>% filter(!is.na(!!sym(time_col)) & !is.na(!!sym(completion_col)))
  if(length(unique(df_c[[completion_col]]))==2) {
    lev <- cohens_d(as.formula(paste(time_col,"~",completion_col)), data = df_c, paired = FALSE)
    cat("\nCohen's d (time by completion):\n"); print(lev)
  }
}

# Save cleaned dataset
write.csv(df, file = file.path("/mnt/data","cleaned_online_course.csv"), row.names = FALSE)
cat("\nCleaned dataset saved to /mnt/data/cleaned_online_course.csv\n")

# -------------------------
# End of script
# -------------------------
cat("\nAll tests completed. Review plots in", plots_dir, "and printed outputs above.\n")
