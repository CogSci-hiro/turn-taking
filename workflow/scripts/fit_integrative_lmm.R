#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(lme4)
  library(ppcor)
})


parse_cli_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  parsed <- list(in_path = NULL, out_dir = NULL)
  index <- 1L
  while (index <= length(args)) {
    key <- args[[index]]
    value <- if (index + 1L <= length(args)) args[[index + 1L]] else NULL
    if (key == "--in") parsed$in_path <- value
    if (key == "--out") parsed$out_dir <- value
    index <- index + 2L
  }
  if (is.null(parsed$in_path) || is.null(parsed$out_dir)) {
    stop("Usage: fit_integrative_lmm.R --in <table.csv> --out <integration_dir>", call. = FALSE)
  }
  parsed
}


resolve_io <- function() {
  if (!exists("snakemake")) return(parse_cli_args())

  outputs <- as.character(snakemake@output)
  input_path <- as.character(snakemake@input[[1]])
  find_output <- function(name, fallback_index) {
    hit <- which(basename(outputs) == name)
    if (length(hit) >= 1L) return(outputs[[hit[[1L]]]])
    outputs[[fallback_index]]
  }

  list(
    in_path = input_path,
    output_paths = c(
      joint_model = find_output("joint_model.csv", 1L),
      interactions = find_output("interactions.csv", 2L),
      random_slope = find_output("random_slope.csv", 3L),
      partial_correlations = find_output("partial_correlations.csv", 4L)
    )
  )
}


safe_z <- function(x) {
  sigma <- stats::sd(x, na.rm = TRUE)
  mu <- mean(x, na.rm = TRUE)
  if (is.na(sigma) || sigma == 0) return(rep(0, length(x)))
  as.numeric((x - mu) / sigma)
}


normalize_predictors <- function(df, predictor_cols) {
  df %>%
    group_by(subject) %>%
    mutate(across(all_of(predictor_cols), safe_z)) %>%
    ungroup()
}


ensure_output_paths <- function(io) {
  if (!is.null(io$output_paths)) return(io$output_paths)

  out_dir <- io$out_dir
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  c(
    joint_model = file.path(out_dir, "joint_model.csv"),
    interactions = file.path(out_dir, "interactions.csv"),
    random_slope = file.path(out_dir, "random_slope.csv"),
    partial_correlations = file.path(out_dir, "partial_correlations.csv")
  )
}


tidy_fixed_effects <- function(model) {
  if (requireNamespace("broom.mixed", quietly = TRUE)) {
    return(broom.mixed::tidy(model, effects = "fixed"))
  }

  coefs <- as.data.frame(summary(model)$coefficients)
  coefs$term <- rownames(coefs)
  rownames(coefs) <- NULL

  stat_col <- intersect(c("t value", "z value"), names(coefs))
  statistic <- if (length(stat_col) >= 1L) coefs[[stat_col[[1L]]]] else rep(NA_real_, nrow(coefs))

  tibble::tibble(
    term = coefs$term,
    estimate = as.numeric(coefs$Estimate),
    std.error = as.numeric(coefs$`Std. Error`),
    statistic = as.numeric(statistic)
  )
}


add_pvalue_column <- function(tbl) {
  if (!("p.value" %in% names(tbl))) tbl$p.value <- NA_real_
  tbl
}


tidy_fixed <- function(model, model_name) {
  tidy_fixed_effects(model) %>%
    add_pvalue_column() %>%
    mutate(model = model_name) %>%
    dplyr::select(model, term, estimate, std.error, statistic, p.value)
}


fit_model <- function(lmer_formula, lm_formula, data, ctrl) {
  if (dplyr::n_distinct(data$subject) > 1L) {
    mixed_try <- tryCatch(
      lmer(lmer_formula, data = data, REML = FALSE, control = ctrl),
      error = function(err) err
    )
    if (!inherits(mixed_try, "error")) return(mixed_try)
  }
  stats::lm(lm_formula, data = data)
}


to_partial_row <- function(label, x, y, z_df) {
  test <- ppcor::pcor.test(x = x, y = y, z = as.matrix(z_df))
  tibble::tibble(
    model = label,
    term = label,
    estimate = as.numeric(test$estimate),
    std.error = NA_real_,
    statistic = as.numeric(test$statistic),
    p.value = as.numeric(test$p.value),
    n = as.integer(test$n),
    gp = as.integer(test$gp)
  )
}


io <- resolve_io()
output_paths <- ensure_output_paths(io)
for (path in output_paths) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
}

df <- readr::read_csv(io$in_path, show_col_types = FALSE)
required <- c("subject", "self_duration", "other_duration", "latency")
missing_required <- setdiff(required, names(df))
if (length(missing_required) > 0) {
  stop(
    paste0("Input table missing required columns: ", paste(missing_required, collapse = ", ")),
    call. = FALSE
  )
}

erp_cols <- grep("^tw[12]_erp_", names(df), value = TRUE)
alpha_cols <- grep("^tw[12]_alpha_", names(df), value = TRUE)
if (length(erp_cols) == 0L || length(alpha_cols) == 0L) {
  stop("Input table must include tw1/tw2 ERP and alpha predictor columns.", call. = FALSE)
}

model_df <- df %>%
  mutate(
    subject = factor(subject),
    ERP = rowMeans(dplyr::across(all_of(erp_cols)), na.rm = FALSE),
    alpha = rowMeans(dplyr::across(all_of(alpha_cols)), na.rm = FALSE)
  ) %>%
  group_by(subject) %>%
  mutate(latency_group = if_else(latency <= median(latency, na.rm = TRUE), "fast", "slow")) %>%
  ungroup() %>%
  mutate(latency_group = factor(latency_group, levels = c("fast", "slow"))) %>%
  normalize_predictors(c("ERP", "alpha", "latency", "other_duration")) %>%
  drop_na(subject, self_duration, ERP, alpha, latency, latency_group, other_duration)

ctrl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

model_a <- fit_model(
  self_duration ~ ERP + alpha + latency + other_duration + (1 | subject),
  self_duration ~ ERP + alpha + latency + other_duration,
  data = model_df,
  ctrl = ctrl
)
model_b <- fit_model(
  self_duration ~ ERP * alpha + latency + other_duration + (1 | subject),
  self_duration ~ ERP * alpha + latency + other_duration,
  data = model_df,
  ctrl = ctrl
)
model_c <- fit_model(
  self_duration ~ ERP + alpha * latency_group + other_duration + (1 | subject),
  self_duration ~ ERP + alpha * latency_group + other_duration,
  data = model_df,
  ctrl = ctrl
)
model_d <- fit_model(
  self_duration ~ ERP + alpha + latency + other_duration + (1 + alpha | subject),
  self_duration ~ ERP + alpha + latency + other_duration,
  data = model_df,
  ctrl = ctrl
)

joint_model <- tidy_fixed(model_a, "Model_A_joint_main_effects")
interactions <- bind_rows(
  tidy_fixed(model_b, "Model_B_erp_alpha_interaction"),
  tidy_fixed(model_c, "Model_C_alpha_latency_group_interaction")
)
random_slope <- tidy_fixed(model_d, "Model_D_random_slope_alpha")

pcor_data <- model_df %>%
  dplyr::select(self_duration, ERP, alpha, latency, other_duration) %>%
  drop_na()

partial_correlations <- bind_rows(
  to_partial_row(
    "alpha_duration_control_ERP",
    x = pcor_data$alpha,
    y = pcor_data$self_duration,
    z_df = pcor_data %>% dplyr::select(ERP)
  ),
  to_partial_row(
    "alpha_duration_control_ERP_latency_other_duration",
    x = pcor_data$alpha,
    y = pcor_data$self_duration,
    z_df = pcor_data %>% dplyr::select(ERP, latency, other_duration)
  ),
  to_partial_row(
    "ERP_alpha_control_duration_latency_other_duration",
    x = pcor_data$ERP,
    y = pcor_data$alpha,
    z_df = pcor_data %>% dplyr::select(self_duration, latency, other_duration)
  )
)

readr::write_csv(joint_model, output_paths[["joint_model"]])
readr::write_csv(interactions, output_paths[["interactions"]])
readr::write_csv(random_slope, output_paths[["random_slope"]])
readr::write_csv(partial_correlations, output_paths[["partial_correlations"]])
