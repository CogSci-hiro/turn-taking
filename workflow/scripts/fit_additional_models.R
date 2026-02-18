# =============================================================================
# ERP–Alpha–Duration minimal model tests (LMM LRTs + partial correlation)
#
# Tests included (renamed):
#   TEST 1: alpha ~ ERP + latency + other + (1|subject)    [LRT vs no-ERP]
#   TEST 2: duration ~ ERP + alpha + latency + other + (1|subject) [LRT vs no-alpha]
#   TEST 3: partial corr(ERP, alpha | duration + latency + other)
#
# Output:
#   - CSV table with columns: test, outcome, term, beta, se, t, p, chi2, df, n
#
# Snakemake usage (typical):
#   Rscript script.R --input path/to/table.csv --output path/to/results.csv
# =============================================================================

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)  # adds df + p-values for fixed effects (Satterthwaite)
  library(ppcor)
  library(dplyr)
})

# =============================================================================
# CLI args (Snakemake-friendly)
# =============================================================================
parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)

  get_arg <- function(flag, default = NA_character_) {
    hit <- grep(paste0("^", flag, "="), args, value = TRUE)
    if (length(hit) == 0) return(default)
    sub(paste0("^", flag, "="), "", hit[[1]])
  }

  list(
    input  = get_arg("--input",  NA_character_),
    output = get_arg("--output", NA_character_)
  )
}

cli <- parse_args()
if (is.na(cli$input) || is.na(cli$output)) {
  stop("Missing required args. Usage: Rscript script.R --input=... --output=...")
}

# =============================================================================
# Config: column names
# =============================================================================
col_subject <- "subject"
col_duration <- "self_duration"
col_erp <- "tw1_erp_posterior"
col_alpha <- "tw1_alpha_posterior"
col_latency <- "latency"
col_other <- "other_duration"

DO_ZSCORE_WITHIN_SUBJECT <- FALSE

# =============================================================================
# Helpers
# =============================================================================
zscore_within_subject <- function(df, cols, subj_col) {
  df %>%
    group_by(.data[[subj_col]]) %>%
    mutate(across(all_of(cols), ~ as.numeric(scale(.x)))) %>%
    ungroup()
}

safe_sd0_check <- function(df, cols) {
  sds <- sapply(df[, cols, drop = FALSE], sd, na.rm = TRUE)
  if (any(is.na(sds)) || any(sds == 0)) {
    bad <- names(sds)[is.na(sds) | sds == 0]
    stop(paste0("Zero/NA variance in: ", paste(bad, collapse = ", "), "."))
  }
}

get_lrt <- function(m_null, m_full) {
  a <- anova(m_null, m_full)
  # second row corresponds to full model vs null
  list(
    chi2 = as.numeric(a$Chisq[2]),
    p_chi2 = as.numeric(a$`Pr(>Chisq)`[2])
  )
}

get_coef_stats <- function(m_full, term) {
  ct <- summary(m_full)$coefficients
  if (!(term %in% rownames(ct))) {
    stop(paste0("Term not found in fixed effects: ", term))
  }
  list(
    beta = as.numeric(ct[term, "Estimate"]),
    se   = as.numeric(ct[term, "Std. Error"]),
    t    = as.numeric(ct[term, "t value"]),
    p_t  = as.numeric(ct[term, "Pr(>|t|)"]),
    df   = if ("df" %in% colnames(ct)) as.numeric(ct[term, "df"]) else NA_real_
  )
}

make_formula <- function(lhs, rhs_terms, subject_col) {
  as.formula(paste(lhs, "~", paste(rhs_terms, collapse = " + "), "+ (1 |", subject_col, ")"))
}

# =============================================================================
# Load data
# =============================================================================
data <- read.csv(cli$input, stringsAsFactors = FALSE)

required_cols <- c(col_subject, col_duration, col_erp, col_alpha, col_latency, col_other)
missing_cols <- setdiff(required_cols, names(data))
if (length(missing_cols) > 0) {
  stop(paste("Missing required columns:", paste(missing_cols, collapse = ", ")))
}

data[[col_subject]] <- factor(data[[col_subject]])

data <- data %>%
  filter(
    !is.na(.data[[col_subject]]),
    !is.na(.data[[col_duration]]),
    !is.na(.data[[col_erp]]),
    !is.na(.data[[col_alpha]]),
    !is.na(.data[[col_latency]]),
    !is.na(.data[[col_other]])
  )

if (DO_ZSCORE_WITHIN_SUBJECT) {
  data <- zscore_within_subject(
    df = data,
    cols = c(col_duration, col_erp, col_alpha, col_latency, col_other),
    subj_col = col_subject
  )
}

cat("\n# ================= Data summary =================\n")
cat("Input:", cli$input, "\n")
cat("Rows (after NA filter):", nrow(data), "\n")
cat("Subjects:", nlevels(data[[col_subject]]), "\n")

# =============================================================================
# Results table accumulator
# =============================================================================
results <- tibble(
  test = character(),
  outcome = character(),
  term = character(),
  beta = numeric(),
  se = numeric(),
  t = numeric(),
  p = numeric(),
  chi2 = numeric(),
  df = numeric(),
  n = integer()
)

append_result <- function(test, outcome, term, beta, se, t, p, chi2, df, n) {
  results <<- bind_rows(
    results,
    tibble(
      test = test,
      outcome = outcome,
      term = term,
      beta = beta,
      se = se,
      t = t,
      p = p,
      chi2 = chi2,
      df = df,
      n = as.integer(n)
    )
  )
}

# =============================================================================
# TEST 1 (LRT): alpha ~ ERP + latency + other + (1|subject)
#   Null: alpha ~ latency + other + (1|subject)
#   Full: alpha ~ ERP + latency + other + (1|subject)
# =============================================================================
m1_null <- lmer(
  make_formula(col_alpha, c(col_latency, col_other), col_subject),
  data = data,
  REML = FALSE
)

m1_full <- lmer(
  make_formula(col_alpha, c(col_erp, col_latency, col_other), col_subject),
  data = data,
  REML = FALSE
)

lrt1 <- get_lrt(m1_null, m1_full)
cs1 <- get_coef_stats(m1_full, col_erp)

append_result(
  test = "TEST 1",
  outcome = col_alpha,
  term = col_erp,
  beta = cs1$beta,
  se = cs1$se,
  t = cs1$t,
  p = lrt1$p_chi2,      # p for LRT (model comparison)
  chi2 = lrt1$chi2,
  df = cs1$df,
  n = nrow(model.frame(m1_full))
)

cat("\n\n# ================= TEST 1: LRT (add ERP to alpha model) =================\n")
print(anova(m1_null, m1_full))
cat("\nFixed effect (ERP) in full model:\n")
print(summary(m1_full)$coefficients[col_erp, , drop = FALSE])

# =============================================================================
# TEST 2 (LRT): duration ~ ERP + alpha + latency + other + (1|subject)
#   Reduced: duration ~ ERP + latency + other + (1|subject)
#   Full:    duration ~ ERP + alpha + latency + other + (1|subject)
# =============================================================================
m2_reduced <- lmer(
  make_formula(col_duration, c(col_erp, col_latency, col_other), col_subject),
  data = data,
  REML = FALSE
)

m2_full <- lmer(
  make_formula(col_duration, c(col_erp, col_alpha, col_latency, col_other), col_subject),
  data = data,
  REML = FALSE
)

lrt2 <- get_lrt(m2_reduced, m2_full)
cs2 <- get_coef_stats(m2_full, col_alpha)

append_result(
  test = "TEST 2",
  outcome = col_duration,
  term = col_alpha,
  beta = cs2$beta,
  se = cs2$se,
  t = cs2$t,
  p = lrt2$p_chi2,      # p for LRT (model comparison)
  chi2 = lrt2$chi2,
  df = cs2$df,
  n = nrow(model.frame(m2_full))
)

cat("\n\n# ================= TEST 2: LRT (add alpha beyond ERP in duration model) =================\n")
print(anova(m2_reduced, m2_full))
cat("\nFixed effect (alpha) in full model:\n")
print(summary(m2_full)$coefficients[col_alpha, , drop = FALSE])

# =============================================================================
# TEST 3 (partial correlation): corr(ERP, alpha | duration + latency + other)
# =============================================================================
pc_data <- data[, c(col_erp, col_alpha, col_duration, col_latency, col_other)]
pc_data <- pc_data[complete.cases(pc_data), ]
safe_sd0_check(pc_data, c(col_erp, col_alpha, col_duration, col_latency, col_other))

pc <- pcor.test(
  x = pc_data[[col_erp]],
  y = pc_data[[col_alpha]],
  z = pc_data[, c(col_duration, col_latency, col_other), drop = FALSE]
)

# ppcor returns:
#   estimate (partial r), statistic (t), p.value, parameter (df)
append_result(
  test = "TEST 3",
  outcome = paste0("pcorr(", col_erp, ", ", col_alpha, ")"),
  term = "partial_r",
  beta = as.numeric(pc$estimate),        # "beta" slot reused as partial r
  se = NA_real_,
  t = as.numeric(pc$statistic),
  p = as.numeric(pc$p.value),
  chi2 = NA_real_,
  df = as.numeric(pc$parameter),
  n = nrow(pc_data)
)

cat("\n\n# ================= TEST 3: Partial correlation (ERP–alpha | duration + latency + other) =================\n")
print(pc)

# =============================================================================
# Write outputs
# =============================================================================
dir.create(dirname(cli$output), recursive = TRUE, showWarnings = FALSE)
write.csv(results, cli$output, row.names = FALSE)

cat("\n# ================= Wrote results =================\n")
cat("Output:", cli$output, "\n")
print(results)

cat("\n# =============================================================================\n")
cat("# Done.\n")
cat("# =============================================================================\n")
