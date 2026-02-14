#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(optparse)
  library(readr)
  library(dplyr)
  library(tibble)
  library(tidyr)
  library(purrr)
  library(stringr)
  library(lme4)
})

# ================================================================================================
# Utilities
# ================================================================================================

scale_z <- function(x) {
  # scale() returns a matrix with attributes; coerce to plain numeric for safety.
  as.numeric(scale(x))
}

ensure_dir <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE, showWarnings = FALSE)
}

safe_deparse <- function(x) paste0(deparse(x, width.cutoff = 500), collapse = "")

get_lme4_messages <- function(mod) {
  msg <- mod@optinfo$conv$lme4$messages
  if (is.null(msg)) return("")
  paste(msg, collapse = " | ")
}

model_overview_row <- function(model_id, outcome, predictor, roi, window, family, kind, mod) {
  tibble(
    model_id = model_id,
    kind = kind,  # "base" or "full"
    outcome = outcome,
    predictor = predictor,
    roi = roi,
    window = window,
    family = family,
    formula = safe_deparse(formula(mod)),
    n_obs = nobs(mod),
    n_subject = dplyr::n_distinct(lme4::getME(mod, "flist")$subject),
    AIC = AIC(mod),
    BIC = BIC(mod),
    logLik = as.numeric(logLik(mod)),
    is_singular = lme4::isSingular(mod, tol = 1e-4),
    lme4_messages = get_lme4_messages(mod)
  )
}

fixed_effects_rows <- function(model_id, kind, mod) {
  sm <- summary(mod)
  coefs <- as.data.frame(sm$coefficients)
  coefs$term <- rownames(coefs)
  rownames(coefs) <- NULL

  # lme4 summary columns are typically: Estimate, Std. Error, t value
  # We keep them generic in case of naming variants.
  coefs <- coefs %>%
    rename_with(~ str_replace_all(., "\\s+", "_")) %>%
    rename(
      estimate = Estimate,
      std_error = Std._Error,
      t_value = t_value
    )

  tibble(
    model_id = model_id,
    kind = kind,
    term = coefs$term,
    estimate = coefs$estimate,
    std_error = coefs$std_error,
    t_value = coefs$t_value
  )
}

fit_pair <- function(
  data,
  model_id,
  formula_base,
  formula_full,
  out_models,
  out_summaries,
  ctrl
) {
  base <- lmer(formula_base, data = data, REML = FALSE, control = ctrl)
  full <- lmer(formula_full, data = data, REML = FALSE, control = ctrl)

  # Save models
  saveRDS(base, file.path(out_models, paste0(model_id, "__base.rds")))
  saveRDS(full, file.path(out_models, paste0(model_id, "__full.rds")))

  # Save full summaries (text)
  writeLines(
    capture.output({
      cat("MODEL ID:", model_id, "\n")
      cat("KIND: base\n")
      cat("FORMULA:", safe_deparse(formula(base)), "\n\n")
      print(summary(base))
    }),
    con = file.path(out_summaries, paste0(model_id, "__base.txt"))
  )

  writeLines(
    capture.output({
      cat("MODEL ID:", model_id, "\n")
      cat("KIND: full\n")
      cat("FORMULA:", safe_deparse(formula(full)), "\n\n")
      print(summary(full))
    }),
    con = file.path(out_summaries, paste0(model_id, "__full.txt"))
  )

  cmp <- anova(base, full)  # LRT since REML=FALSE for both

  list(base = base, full = full, cmp = cmp)
}

# ================================================================================================
# CLI
# ================================================================================================

option_list <- list(
  make_option(c("--in"), dest = "in_path", type = "character", help = "Input CSV (trial table)"),
  make_option(c("--out"), dest = "out_dir", type = "character", help = "Output directory"),
  make_option(c("--zscore"), dest = "zscore", type = "logical", default = TRUE,
              help = "Z-score numeric variables (default: TRUE, legacy-like)"),
  make_option(c("--run_as_factor"), dest = "run_as_factor", type = "logical", default = FALSE,
              help = "Treat run as factor instead of z-scoring it (default: FALSE, legacy-like scaling)"),
  make_option(c("--optimizer"), dest = "optimizer", type = "character", default = "bobyqa",
              help = "lmer optimizer (default: bobyqa)"),
  make_option(c("--maxfun"), dest = "maxfun", type = "integer", default = 200000,
              help = "Optimizer maxfun (default: 200000)")
)

parser <- OptionParser(option_list = option_list)
opt <- parse_args(parser)

if (is.null(opt$in_path) || is.null(opt$out_dir)) {
  print_help(parser)
  stop("Missing required arguments: --in and --out", call. = FALSE)
}

# ================================================================================================
# I/O + deterministic preprocessing
# ================================================================================================

df <- readr::read_csv(opt$in_path, show_col_types = FALSE)

required_cols <- c(
  "subject", "run",
  "self_duration", "other_duration", "latency",
  "baseline_erp_anterior", "baseline_erp_posterior",
  "tw1_erp_anterior", "tw1_erp_posterior", "tw2_erp_anterior", "tw2_erp_posterior",
  "tw1_alpha_anterior", "tw1_alpha_posterior", "tw2_alpha_anterior", "tw2_alpha_posterior",
  "tw1_beta_anterior", "tw1_beta_posterior", "tw2_beta_anterior", "tw2_beta_posterior"
)

missing <- setdiff(required_cols, colnames(df))
if (length(missing) > 0) {
  stop(paste0("Input CSV missing required columns: ", paste(missing, collapse = ", ")), call. = FALSE)
}

# Ensure subject is a factor with stable ordering
df <- df %>%
  mutate(
    subject = factor(subject, levels = sort(unique(subject)))
  )

# run: legacy scaled it; we keep default legacy-like behavior (numeric -> zscore)
if (opt$run_as_factor) {
  df <- df %>% mutate(run = factor(run))
} else {
  df <- df %>% mutate(run = as.numeric(run))
}

# Z-scoring (legacy-like)
if (isTRUE(opt$zscore)) {
  # Only scale numeric columns; never touch factors/characters.
  # NOTE: timestamp is currently present but not used in models; scaling it would be harmless but confusing.
  numeric_cols <- df %>%
    select(where(is.numeric)) %>%
    colnames()

  numeric_cols <- setdiff(numeric_cols, "timestamp")

  df <- df %>%
    mutate(across(all_of(numeric_cols), scale_z))
}

# ================================================================================================
# Model grid
# ================================================================================================

# Predictors come directly from the columns
predictors <- tibble(
  predictor = c(
    "tw1_erp_anterior", "tw1_erp_posterior",
    "tw2_erp_anterior", "tw2_erp_posterior",
    "tw1_alpha_anterior", "tw1_alpha_posterior",
    "tw2_alpha_anterior", "tw2_alpha_posterior",
    "tw1_beta_anterior", "tw1_beta_posterior",
    "tw2_beta_anterior", "tw2_beta_posterior"
  )
) %>%
  mutate(
    window = if_else(str_detect(predictor, "^tw1_"), "tw1", "tw2"),
    family = case_when(
      str_detect(predictor, "_erp_") ~ "erp",
      str_detect(predictor, "_alpha_") ~ "alpha",
      str_detect(predictor, "_beta_") ~ "beta",
      TRUE ~ "unknown"
    ),
    roi = case_when(
      str_detect(predictor, "_anterior$") ~ "anterior",
      str_detect(predictor, "_posterior$") ~ "posterior",
      TRUE ~ "unknown"
    ),
    baseline_roi = case_when(
      roi == "anterior" ~ "baseline_erp_anterior",
      roi == "posterior" ~ "baseline_erp_posterior",
      TRUE ~ NA_character_
    )
  )

# Outcomes and their baseline covariates (legacy-ish, symmetric)
# You can change this later without touching the rest of the script.
outcomes <- tibble(
  outcome = c("self_duration", "latency"),
  base_covariates = c(
    # Predict self_duration from interaction-relevant nuisances
    "latency + other_duration + run",
    # Predict latency similarly, plus self_duration
    "self_duration + other_duration + run"
  )
)

grid <- tidyr::crossing(outcomes, predictors) %>%
  mutate(
    # Include baseline_erp_* only for erp-family predictors (since only erp baseline exists in the table)
    include_neural_baseline = (family == "erp")
  )

# ================================================================================================
# Output structure
# ================================================================================================

out_dir <- opt$out_dir
out_models <- file.path(out_dir, "models")
out_summaries <- file.path(out_dir, "summaries_full")
out_tables <- file.path(out_dir, "tables")

ensure_dir(out_dir)
ensure_dir(out_models)
ensure_dir(out_summaries)
ensure_dir(out_tables)

# Save session info for reproducibility
writeLines(capture.output(sessionInfo()), file.path(out_dir, "sessionInfo.txt"))

ctrl <- lmerControl(
  optimizer = opt$optimizer,
  optCtrl = list(maxfun = opt$maxfun)
)

# ================================================================================================
# Fit all models
# ================================================================================================

all_model_overview <- list()
all_fixed_effects  <- list()
all_lrt            <- list()

for (i in seq_len(nrow(grid))) {

  row <- grid[i, ]

  outcome      <- row$outcome[[1]]
  base_covs    <- row$base_covariates[[1]]
  predictor    <- row$predictor[[1]]
  roi          <- row$roi[[1]]
  window       <- row$window[[1]]
  family       <- row$family[[1]]
  baseline_roi <- row$baseline_roi[[1]]

  # ----------------------------------------------------------------------------
  # Construct model formulas
  # ----------------------------------------------------------------------------

  model_id <- paste(outcome, predictor, sep = "__")

  neural_baseline_term <- ""
  if (isTRUE(row$include_neural_baseline[[1]])) {
    neural_baseline_term <- paste0(" + ", baseline_roi)
  }

  formula_base_str <- paste0(
    outcome, " ~ ", base_covs,
    neural_baseline_term,
    " + (1|subject)"
  )

  formula_full_str <- paste0(
    outcome, " ~ ", base_covs,
    neural_baseline_term,
    " + ", predictor,
    " + (1|subject)"
  )

  formula_base <- as.formula(formula_base_str)
  formula_full <- as.formula(formula_full_str)

  # ----------------------------------------------------------------------------
  # Ensure base and full are fit on identical rows (drop NAs deterministically)
  # ----------------------------------------------------------------------------

  vars_needed <- c(
    outcome,
    "subject",
    "run",
    "self_duration",
    "other_duration",
    "latency",
    predictor
  )

  if (isTRUE(row$include_neural_baseline[[1]])) {
    vars_needed <- c(vars_needed, baseline_roi)
  }

  df_fit <- df %>%
    dplyr::select(all_of(unique(vars_needed))) %>%
    tidyr::drop_na()

  # ----------------------------------------------------------------------------
  # Fit models
  # ----------------------------------------------------------------------------

  res <- fit_pair(
    data          = df_fit,
    model_id      = model_id,
    formula_base  = formula_base,
    formula_full  = formula_full,
    out_models    = out_models,
    out_summaries = out_summaries,
    ctrl          = ctrl
  )

  base_mod <- res$base
  full_mod <- res$full
  cmp      <- res$cmp

  # ----------------------------------------------------------------------------
  # Store model summaries
  # ----------------------------------------------------------------------------

  all_model_overview[[length(all_model_overview) + 1]] <-
    model_overview_row(model_id, outcome, predictor, roi, window, family, "base", base_mod)

  all_model_overview[[length(all_model_overview) + 1]] <-
    model_overview_row(model_id, outcome, predictor, roi, window, family, "full", full_mod)

  all_fixed_effects[[length(all_fixed_effects) + 1]] <-
    fixed_effects_rows(model_id, "base", base_mod)

  all_fixed_effects[[length(all_fixed_effects) + 1]] <-
    fixed_effects_rows(model_id, "full", full_mod)

  # ----------------------------------------------------------------------------
  # Likelihood Ratio Test (robust extraction)
  # ----------------------------------------------------------------------------

  cmp_row <- as.data.frame(cmp)

  # Different R versions label df column differently
  df_col <- if ("Chi Df" %in% colnames(cmp_row)) {
    "Chi Df"
  } else if ("Df" %in% colnames(cmp_row)) {
    "Df"
  } else {
    NA_character_
  }

  ok_two_rows <- nrow(cmp_row) >= 2
  has_chisq   <- "Chisq" %in% colnames(cmp_row)
  has_p       <- "Pr(>Chisq)" %in% colnames(cmp_row)
  has_df      <- !is.na(df_col)

  if (ok_two_rows && has_chisq && has_df) {

    # Standard extraction from anova table
    lrt_stat <- as.numeric(cmp_row$Chisq[2])
    lrt_df   <- as.numeric(cmp_row[[df_col]][2])
    lrt_p    <- if (has_p) as.numeric(cmp_row$`Pr(>Chisq)`[2]) else NA_real_

  } else {

    # Fallback: compute LRT manually from log-likelihoods
    ll0 <- as.numeric(logLik(base_mod))
    ll1 <- as.numeric(logLik(full_mod))

    df0 <- attr(logLik(base_mod), "df")
    df1 <- attr(logLik(full_mod), "df")

    lrt_stat <- 2 * (ll1 - ll0)
    lrt_df   <- df1 - df0
    lrt_p    <- if (is.finite(lrt_stat) && lrt_df > 0) {
      stats::pchisq(lrt_stat, df = lrt_df, lower.tail = FALSE)
    } else {
      NA_real_
    }
  }

  delta_aic    <- AIC(full_mod) - AIC(base_mod)
  delta_bic    <- BIC(full_mod) - BIC(base_mod)
  delta_loglik <- as.numeric(logLik(full_mod) - logLik(base_mod))

  # ----------------------------------------------------------------------------
  # Store LRT summary row
  # ----------------------------------------------------------------------------

  all_lrt[[length(all_lrt) + 1]] <- tibble(
    model_id      = model_id,
    outcome       = outcome,
    predictor     = predictor,
    roi           = roi,
    window        = window,
    family        = family,
    n_used        = nrow(df_fit),
    lrt_chisq     = lrt_stat,
    lrt_df        = lrt_df,
    lrt_p         = lrt_p,
    delta_AIC     = delta_aic,
    delta_BIC     = delta_bic,
    delta_logLik  = delta_loglik
  )

}


models_df <- bind_rows(all_model_overview) %>%
  arrange(outcome, family, window, roi, kind, model_id)

fixed_df <- bind_rows(all_fixed_effects) %>%
  arrange(model_id, kind, term)

lrt_df <- bind_rows(all_lrt) %>%
  arrange(outcome, family, window, roi, model_id)

readr::write_csv(models_df, file.path(out_tables, "models.csv"))
readr::write_csv(fixed_df, file.path(out_tables, "fixed_effects.csv"))
readr::write_csv(lrt_df, file.path(out_tables, "lrt_comparisons.csv"))
