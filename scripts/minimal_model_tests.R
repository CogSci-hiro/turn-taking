#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(lme4)
  library(ppcor)
})

set.seed(0)

REQUIRED_COLUMNS <- c(
  "subject",
  "self_duration",
  "tw1_erp_posterior",
  "tw1_alpha_posterior",
  "latency",
  "other_duration"
)

stopf <- function(message_text) {
  stop(message_text, call. = FALSE)
}

to_bool <- function(value) {
  if (is.logical(value)) return(isTRUE(value))
  tolower(as.character(value)) %in% c("true", "1", "yes", "y", "t")
}

format_num <- function(x, digits = 6L) {
  if (is.na(x)) return("NA")
  formatC(x, format = "f", digits = digits)
}

format_p <- function(x) {
  if (is.na(x)) return("NA")
  formatC(x, format = "e", digits = 3L)
}

format_md_num <- function(x) {
  if (is.na(x)) return("NA")
  if (x == 0) return("0")
  if (abs(x) < 1e-4 || abs(x) >= 1e5) {
    return(formatC(x, format = "e", digits = 3L))
  }
  formatC(x, format = "f", digits = 6L)
}

safe_z <- function(x) {
  sigma <- stats::sd(x, na.rm = TRUE)
  mu <- mean(x, na.rm = TRUE)
  if (is.na(sigma) || sigma == 0) {
    out <- rep(0, length(x))
    out[is.na(x)] <- NA_real_
    return(out)
  }
  as.numeric((x - mu) / sigma)
}

markdown_table_lines <- function(df) {
  out <- data.frame(df, stringsAsFactors = FALSE, check.names = FALSE)
  for (name in names(out)) {
    column <- out[[name]]
    if (is.numeric(column)) {
      out[[name]] <- vapply(column, format_md_num, character(1L))
    } else {
      out[[name]] <- ifelse(is.na(column), "NA", as.character(column))
    }
  }
  header <- paste0("| ", paste(names(out), collapse = " | "), " |")
  sep <- paste0("| ", paste(rep("---", ncol(out)), collapse = " | "), " |")
  rows <- if (nrow(out) == 0L) character(0) else apply(out, 1L, function(row) paste0("| ", paste(row, collapse = " | "), " |"))
  c(header, sep, rows)
}

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  parsed <- list(in_path = NULL, out_dir = "reports/minimal_model_tests", use_raw = FALSE)
  index <- 1L
  while (index <= length(args)) {
    key <- args[[index]]
    value <- if (index + 1L <= length(args)) args[[index + 1L]] else NULL
    if (key == "--in") parsed$in_path <- value
    if (key == "--outdir") parsed$out_dir <- value
    if (key == "--use_raw") parsed$use_raw <- to_bool(value)
    index <- index + 2L
  }
  if (is.null(parsed$in_path)) {
    stopf("Usage: minimal_model_tests.R --in <table.csv> --outdir <dir> --use_raw <true|false>")
  }
  parsed
}

resolve_snakemake_value <- function(values, key, default = NULL) {
  values_char <- as.character(values)
  values_names <- names(values_char)
  if (!is.null(values_names) && key %in% values_names) return(values_char[[key]])
  if (length(values_char) > 0L) values_char[[1L]] else default
}

resolve_io <- function() {
  if (!exists("snakemake")) return(parse_args())

  in_path <- resolve_snakemake_value(snakemake@input, "table", default = NULL)
  if (is.null(in_path)) stopf("Snakemake input must include named file 'table'.")

  out_full <- resolve_snakemake_value(snakemake@output, "full_report", default = NULL)
  out_csv <- resolve_snakemake_value(snakemake@output, "condensed_csv", default = NULL)
  out_md <- resolve_snakemake_value(snakemake@output, "condensed_md", default = NULL)
  if (any(is.null(c(out_full, out_csv, out_md)))) {
    stopf("Snakemake outputs must include full_report, condensed_csv, condensed_md.")
  }

  use_raw <- FALSE
  params_list <- snakemake@params
  if ("use_raw" %in% names(params_list)) {
    use_raw <- to_bool(params_list[["use_raw"]])
  }

  list(
    in_path = in_path,
    out_dir = dirname(out_full),
    use_raw = use_raw,
    output_paths = c(
      full_report = out_full,
      condensed_csv = out_csv,
      condensed_md = out_md
    )
  )
}

coefs_table <- function(model) {
  coef_df <- as.data.frame(summary(model)$coefficients)
  coef_df$term <- rownames(coef_df)
  rownames(coef_df) <- NULL
  coef_df <- coef_df[, c("term", "Estimate", "Std. Error", "t value"), drop = FALSE]
  names(coef_df) <- c("term", "estimate", "se", "t")
  coef_df
}

anova_table <- function(model_a, model_b) {
  out <- as.data.frame(anova(model_a, model_b))
  out$model <- rownames(out)
  rownames(out) <- NULL
  out[, c("model", setdiff(names(out), "model")), drop = FALSE]
}

extract_lrt <- function(anova_df) {
  p_col <- grep("Pr\\(>Chisq\\)", names(anova_df), value = TRUE)
  if (length(p_col) == 0L) stopf("Failed to extract LRT p-value column from anova output.")
  target_row <- nrow(anova_df)
  list(
    chisq = as.numeric(anova_df[[target_row, "Chisq"]]),
    df = as.numeric(anova_df[[target_row, "Df"]]),
    p_value = as.numeric(anova_df[[target_row, p_col[[1L]]]])
  )
}

coef_value <- function(model, term) {
  coef_mat <- summary(model)$coefficients
  if (!(term %in% rownames(coef_mat))) {
    stopf(sprintf("Fixed-effect term '%s' not found in model coefficients.", term))
  }
  as.numeric(coef_mat[term, "Estimate"])
}

coef_beta_se <- function(model, term) {
  coef_mat <- summary(model)$coefficients
  if (!(term %in% rownames(coef_mat))) {
    stopf(sprintf("Fixed-effect term '%s' not found in model coefficients.", term))
  }
  if (!("Std. Error" %in% colnames(coef_mat))) {
    stopf("Model coefficients are missing 'Std. Error' column.")
  }
  list(
    beta = as.numeric(coef_mat[term, "Estimate"]),
    se = as.numeric(coef_mat[term, "Std. Error"])
  )
}

sign_symbol <- function(value) {
  if (is.na(value)) return("NA")
  if (value > 0) return("+")
  if (value < 0) return("-")
  "0"
}

subset_complete <- function(data, columns, label) {
  keep <- stats::complete.cases(data[, columns, drop = FALSE])
  out <- droplevels(data[keep, , drop = FALSE])
  if (nrow(out) < 5L) {
    stopf(sprintf("Not enough complete rows for %s after NA filtering.", label))
  }
  if (length(unique(out$subject)) < 2L) {
    stopf(sprintf("Mixed model for %s requires at least 2 subjects after NA filtering.", label))
  }
  out
}

io <- resolve_io()
dir.create(io$out_dir, recursive = TRUE, showWarnings = FALSE)

output_paths <- io$output_paths
if (is.null(output_paths)) {
  output_paths <- c(
    full_report = file.path(io$out_dir, "full_report.md"),
    condensed_csv = file.path(io$out_dir, "condensed_table.csv"),
    condensed_md = file.path(io$out_dir, "condensed_table.md")
  )
}

for (path in output_paths) dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)

if (!file.exists(io$in_path)) {
  stopf(sprintf("Input CSV not found: %s", io$in_path))
}

df <- read.csv(io$in_path, stringsAsFactors = FALSE)
missing_required <- setdiff(REQUIRED_COLUMNS, names(df))
if (length(missing_required) > 0L) {
  stopf(sprintf(
    "Input table missing required columns: %s",
    paste(missing_required, collapse = ", ")
  ))
}

df$subject <- as.factor(df$subject)

numeric_cols <- c(
  "self_duration",
  "tw1_erp_posterior",
  "tw1_alpha_posterior",
  "latency",
  "other_duration"
)
for (col_name in numeric_cols) {
  df[[col_name]] <- as.numeric(df[[col_name]])
}

for (col_name in numeric_cols) {
  z_name <- paste0(col_name, "_z")
  df[[z_name]] <- ave(df[[col_name]], df$subject, FUN = safe_z)
}

use_raw <- isTRUE(io$use_raw)

duration_var <- if (use_raw) "self_duration" else "self_duration_z"
erp_var <- if (use_raw) "tw1_erp_posterior" else "tw1_erp_posterior_z"
alpha_var <- if (use_raw) "tw1_alpha_posterior" else "tw1_alpha_posterior_z"
latency_var <- if (use_raw) "latency" else "latency_z"
other_var <- if (use_raw) "other_duration" else "other_duration_z"

t1_null_formula <- sprintf("%s ~ %s + %s + (1|subject)", duration_var, latency_var, other_var)
t1_full_formula <- sprintf("%s ~ %s + %s + %s + (1|subject)", duration_var, erp_var, latency_var, other_var)
t2_null_formula <- sprintf("%s ~ %s + %s + (1|subject)", alpha_var, latency_var, other_var)
t2_full_formula <- sprintf("%s ~ %s + %s + %s + (1|subject)", alpha_var, erp_var, latency_var, other_var)
t3_reduced_formula <- sprintf("%s ~ %s + %s + %s + (1|subject)", duration_var, erp_var, latency_var, other_var)
t3_full_formula <- sprintf("%s ~ %s + %s + %s + %s + (1|subject)", duration_var, erp_var, alpha_var, latency_var, other_var)
t4_formula <- sprintf(
  "corr(%s, %s | %s, %s, %s)",
  erp_var,
  alpha_var,
  duration_var,
  latency_var,
  other_var
)

model_control <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000L))

data_t1 <- subset_complete(
  df,
  c("subject", duration_var, erp_var, latency_var, other_var),
  "Test 1"
)
data_t2 <- subset_complete(
  df,
  c("subject", alpha_var, erp_var, latency_var, other_var),
  "Test 2"
)
data_t3 <- subset_complete(
  df,
  c("subject", duration_var, erp_var, alpha_var, latency_var, other_var),
  "Test 3"
)

m1_null <- lmer(as.formula(t1_null_formula), data = data_t1, REML = FALSE, control = model_control)
m1_full <- lmer(as.formula(t1_full_formula), data = data_t1, REML = FALSE, control = model_control)
m2_null <- lmer(as.formula(t2_null_formula), data = data_t2, REML = FALSE, control = model_control)
m2_full <- lmer(as.formula(t2_full_formula), data = data_t2, REML = FALSE, control = model_control)
m3_reduced <- lmer(as.formula(t3_reduced_formula), data = data_t3, REML = FALSE, control = model_control)
m3_full <- lmer(as.formula(t3_full_formula), data = data_t3, REML = FALSE, control = model_control)

a1_df <- anova_table(m1_null, m1_full)
a2_df <- anova_table(m2_null, m2_full)
a3_df <- anova_table(m3_reduced, m3_full)

t1 <- extract_lrt(a1_df)
t2 <- extract_lrt(a2_df)
t3 <- extract_lrt(a3_df)

t1_erp <- coef_value(m1_full, erp_var)
t2_erp <- coef_value(m2_full, erp_var)
t3_erp <- coef_value(m3_full, erp_var)
t3_alpha <- coef_value(m3_full, alpha_var)
t1_stats <- coef_beta_se(m1_full, erp_var)
t2_stats <- coef_beta_se(m2_full, erp_var)
t3_stats <- coef_beta_se(m3_full, alpha_var)

pc_cols <- c(erp_var, alpha_var, duration_var, latency_var, other_var)
pc_data <- df[stats::complete.cases(df[, pc_cols]), pc_cols, drop = FALSE]
n_complete_pc <- nrow(pc_data)

if (n_complete_pc < 5L) {
  stopf("Partial correlation undefined: fewer than 5 complete cases in required columns.")
}

sd_values <- sapply(pc_data, stats::sd)
if (any(is.na(sd_values)) || any(sd_values == 0)) {
  zero_vars <- names(sd_values)[is.na(sd_values) | sd_values == 0]
  stopf(sprintf(
    "Partial correlation undefined: sd==0 after complete.cases for %s",
    paste(zero_vars, collapse = ", ")
  ))
}

pc <- ppcor::pcor.test(
  x = pc_data[[erp_var]],
  y = pc_data[[alpha_var]],
  z = pc_data[, c(duration_var, latency_var, other_var), drop = FALSE],
  method = "pearson"
)

pc_r <- as.numeric(pc$estimate)
pc_t <- as.numeric(pc$statistic)
pc_p <- as.numeric(pc$p.value)
pc_n <- as.integer(pc$n)
pc_df <- as.integer(pc$n - pc$gp - 2)

checklist <- data.frame(
  check = c(
    "ERP improves duration model? (p<0.05)",
    "ERP coefficient in duration full model positive?",
    "ERP improves alpha model? (p<0.05)",
    "ERP coefficient in alpha full model positive?",
    "Alpha improves duration beyond ERP? (p<0.05)",
    "Alpha coefficient in duration(full) negative?",
    "Partial corr ERP-alpha positive and p<0.05?"
  ),
  pass = c(
    t1$p_value < 0.05,
    t1_erp > 0,
    t2$p_value < 0.05,
    t2_erp > 0,
    t3$p_value < 0.05,
    t3_alpha < 0,
    (pc_r > 0) && (pc_p < 0.05)
  ),
  stringsAsFactors = FALSE
)
checklist$status <- ifelse(checklist$pass, "PASS", "FAIL")

condensed <- data.frame(
  test_id = c("T1", "T2", "T3", "T4"),
  outcome = c(
    "Add ERP to duration model",
    "Add ERP to alpha model",
    "Add alpha beyond ERP in duration model",
    "Partial corr ERP-alpha controlling duration+latency+other_duration"
  ),
  model_full_formula = c(
    t1_full_formula,
    t2_full_formula,
    t3_full_formula,
    t4_formula
  ),
  model_null_or_reduced_formula = c(
    t1_null_formula,
    t2_null_formula,
    t3_reduced_formula,
    sprintf("partial corr controls = %s, %s, %s", duration_var, latency_var, other_var)
  ),
  chisq_or_r = c(t1$chisq, t2$chisq, t3$chisq, pc_r),
  df = c(t1$df, t2$df, t3$df, pc_df),
  p_value = c(t1$p_value, t2$p_value, t3$p_value, pc_p),
  beta = c(t1_stats$beta, t2_stats$beta, t3_stats$beta, pc_r),
  se = c(t1_stats$se, t2_stats$se, t3_stats$se, NA_real_),
  direction_key_effect = c(
    sprintf("ERP %s", sign_symbol(t1_erp)),
    sprintf("ERP %s", sign_symbol(t2_erp)),
    sprintf("alpha %s; ERP %s", sign_symbol(t3_alpha), sign_symbol(t3_erp)),
    sprintf("r %s", sign_symbol(pc_r))
  ),
  n_complete = c(NA_integer_, NA_integer_, NA_integer_, pc_n),
  verdict = c(
    ifelse((t1$p_value < 0.05) && (t1_erp > 0), "PASS", "FAIL"),
    ifelse((t2$p_value < 0.05) && (t2_erp > 0), "PASS", "FAIL"),
    ifelse((t3$p_value < 0.05) && (t3_alpha < 0), "PASS", "FAIL"),
    ifelse((pc_p < 0.05) && (pc_r > 0), "PASS", "FAIL")
  ),
  stringsAsFactors = FALSE
)

utils::write.csv(condensed, output_paths[["condensed_csv"]], row.names = FALSE, na = "NA")
writeLines(markdown_table_lines(condensed), con = output_paths[["condensed_md"]])

required_missingness <- data.frame(
  column = REQUIRED_COLUMNS,
  missing_n = sapply(REQUIRED_COLUMNS, function(name) sum(is.na(df[[name]]))),
  missing_pct = sapply(REQUIRED_COLUMNS, function(name) mean(is.na(df[[name]])) * 100),
  stringsAsFactors = FALSE
)

fixed_t1 <- coefs_table(m1_full)
fixed_t2 <- coefs_table(m2_full)
fixed_t3 <- coefs_table(m3_full)

report_lines <- c(
  "# Minimal Model Tests Report",
  "",
  sprintf("- Input CSV: `%s`", io$in_path),
  sprintf("- Scale used: `%s`", if (use_raw) "raw columns" else "within-subject z-scored columns"),
  "- Random effect structure: random intercept only `(1|subject)`",
  "- LRT estimation mode: ML (`REML=FALSE`)",
  "",
  "## Data summary",
  sprintf("- Rows: %d", nrow(df)),
  sprintf("- Subjects: %d", length(unique(df$subject))),
  sprintf("- Complete cases used for partial correlation: %d", n_complete_pc),
  "",
  "### Missingness (required columns)",
  markdown_table_lines(required_missingness),
  "",
  "## Exact formulas",
  sprintf("- T1 null: `%s`", t1_null_formula),
  sprintf("- T1 full: `%s`", t1_full_formula),
  sprintf("- T2 null: `%s`", t2_null_formula),
  sprintf("- T2 full: `%s`", t2_full_formula),
  sprintf("- T3 reduced: `%s`", t3_reduced_formula),
  sprintf("- T3 full: `%s`", t3_full_formula),
  sprintf("- T4 partial correlation: `%s`", t4_formula),
  "",
  "## Test 1 (LRT): Add ERP to duration model",
  "### Raw ANOVA output",
  markdown_table_lines(a1_df),
  sprintf(
    "- One-line extract: ChiSq=%s, df=%s, p=%s, ERP sign=%s",
    format_num(t1$chisq, 6L),
    format_num(t1$df, 0L),
    format_p(t1$p_value),
    sign_symbol(t1_erp)
  ),
  "### Fixed effects (full model)",
  markdown_table_lines(fixed_t1),
  "",
  "## Test 2 (LRT): Add ERP to alpha model",
  "### Raw ANOVA output",
  markdown_table_lines(a2_df),
  sprintf(
    "- One-line extract: ChiSq=%s, df=%s, p=%s, ERP sign=%s",
    format_num(t2$chisq, 6L),
    format_num(t2$df, 0L),
    format_p(t2$p_value),
    sign_symbol(t2_erp)
  ),
  "### Fixed effects (full model)",
  markdown_table_lines(fixed_t2),
  "",
  "## Test 3 (LRT): Add alpha beyond ERP in duration model",
  "### Raw ANOVA output",
  markdown_table_lines(a3_df),
  sprintf(
    "- One-line extract: ChiSq=%s, df=%s, p=%s, alpha sign=%s, ERP sign=%s",
    format_num(t3$chisq, 6L),
    format_num(t3$df, 0L),
    format_p(t3$p_value),
    sign_symbol(t3_alpha),
    sign_symbol(t3_erp)
  ),
  "### Fixed effects (full model)",
  markdown_table_lines(fixed_t3),
  "",
  "## Test 4 (Partial correlation)",
  sprintf("- Formula: `%s`", t4_formula),
  sprintf("- r = %s", format_num(pc_r, 6L)),
  sprintf("- t = %s", format_num(pc_t, 6L)),
  sprintf("- df = %s", format_num(pc_df, 0L)),
  sprintf("- p-value = %s", format_p(pc_p)),
  sprintf("- n complete cases = %d", pc_n),
  "",
  "## Compatibility checklist",
  markdown_table_lines(checklist[, c("check", "status")]),
  "",
  "## Condensed manuscript table",
  markdown_table_lines(condensed),
  "",
  "## Manuscript paragraph validation",
  sprintf("- Claim 1 (T1): ERP -> duration positive + significant. See row `T1` (`%s`).", condensed$verdict[condensed$test_id == "T1"]),
  sprintf("- Claim 2 (T2): ERP -> alpha positive + significant. See row `T2` (`%s`).", condensed$verdict[condensed$test_id == "T2"]),
  sprintf("- Claim 3 (T3): alpha -> duration negative + significant controlling ERP. See row `T3` (`%s`).", condensed$verdict[condensed$test_id == "T3"]),
  sprintf("- Claim 4 (T4): partial r(ERP, alpha) positive + significant controlling behavior covariates. See row `T4` (`%s`).", condensed$verdict[condensed$test_id == "T4"])
)

writeLines(report_lines, con = output_paths[["full_report"]])
