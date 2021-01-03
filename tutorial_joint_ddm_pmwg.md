Tutorial script for multiple-task joint modeling with the PMwG sampler
================

## Overview

This code constructs a joint model of two simulated behavioural
datasets. It can be extended to N tasks and multiple data sources per
subject (e.g., EEG + choice-RT + fMRI). The code and variable names in
this script are intended to be as general as possible to allow easy
modification and to let users ‘plug in’ their own experimental designs
and models as straightforwardly as possible.

#### The building blocks

The script starts by creating an experimental design matrix (an N x M
data frame representing N trials of M experimental factors). The design
matrix encodes the features of the experimental design (e.g., subject,
stimulus, condition, etc.) on each trial. This is useful for simulating
and plotting data, and for computing variables based on your particular
experimental design. As such, the most important objects in the script
are the data frames `data`, `design`, and `sims`. These all have the
same form as the design matrix (i.e., `data` and `sims` are the design
matrix with additional columns of observed or simulated outcomes, such
as ‘rt’ and ‘response’. These objects can thus be passed interchangeably
to the likelihood and sampling functions for different purposes (e.g.,
fitting to data, simulating out of a model, generating hypothetical
data).

#### The likelihood function

At the heart of this code is the likelihood function. The likelihood
function iterates over each trial in the design and computes either 1)
the likelihood of the current observation in `data` given the current
parameter values, or 2) a simulated observation generated from the model
given the current parameters and cell of the experimental design.

The bulk of the work when building a new model is writing the likelihood
function. It is easiest to first write (and debug) the ‘stand alone’
likelihood of the model as applied to a single trial (row) in the data.
From there you can easily iterate over trials using a for loop.
Likelihood functions for many designs can be made faster by
‘vectorizing’ over observations that share a cell in the design
matrix (e.g., with a single call to `dLBA()`, `dnorm()`, `ddiffusion()`,
etc.). However, the loop method has the advantage of being more
intuitive and easier to debug in in the initial stages of writing a
model (despite its longer compute time).

Constructing a joint likelihood (e.g., the likelihood of N models of N
data sets) simply involves putting each model’s likelihood loop inside
the same likelihood function and passing that function a named list of
data frames (e.g., the list `data` containing data frames `data$task1`,
`data$task2`, `data$task3`, etc.). The returned likelihood is then just
the grand sum of each model’s individual summed likelihood. Functions
for simulating out of the models are computed in the same manner as the
single case and are returned as a list of data frames (one for each
task/model) at the end of the function.

#### The sampler

Running the Particle Metropolis within Gibbs (PMwG) sampler is
straightforward - you simply pass the `data` (or `sims`) object, the
likelihood function, and a vector of parameter names as arguments. See
<https://newcastlecl.github.io/samplerDoc/> for further information
about the sampler.

When working with a single model, covariation among parameters is
accounted for hierarchically, via a multivariate normal distribution.
The multivariate normal has a mean vector containing the hyper-level
means of each parameter and variance-covariance matrix representing
parameter correlations. When working with multiple models simultaneously
(i.e., joint modeling), the mean vector is simply the union
(concatenation) of each model’s individual mean vector (e.g., `all.means
<- c(m1.means, m2.means)`). The covariance matrix is similarly
constructed based on this extended means vector. Variance shared among
different tasks is thus accounted for hierarchically under the same
overarching multivariate normal.

**TO DO**: Add convenience functions for plotting parameter samples,
model fits, posterior predictives/parameter recovery/model exploration,
heatmap of hyper var-covar matrix, etc.

## Setup environment

#### Clear workspace

``` r
rm(list=ls())
```

#### Set working directory

``` r
getwd()
main_dir <- c("~/Documents/Github/PMwG")
setwd(file.path(main_dir))
```

#### Create subdirectory structure

This will not overwrite if dir already exists.

``` r
sub_dirs <- list("samples", "plots", "data")
lapply(sub_dirs, function (x) dir.create(file.path(main_dir, x), showWarnings = FALSE))
```

#### Load packages

``` r
library(tidyverse)
library(rtdists)
library(pmwg)
```

#### Set seed for random number reproducibility

``` r
set.seed(101)
```

#### Create some empty storage objects

``` r
design <- list()
data <- list()
p_names <- list()
p_vector <- list()
sims <- list()
```

We will fill these in with important bits and pieces needed by the
likelihood and sampling functions as we go. These data objects are the
‘building blocks’ of how we will pass data to and from the likelihood
and sampling functions.

We will mostly work with lists of data frames (or vectors) because this
allows us to generalize the code to an arbitrary number of
tasks/models/data sources when doing joint modeling.

To this end, all the important data objects we will work with conform to
the `tidyverse` principles of tidy data and can thus be used as is with
the `purrr` and `broom` libraries for performing the same computation on
many objects. This is especially useful for modeling, plotting, and
summarizing joint models and data spanning multiple tasks and subjects.

## Setup Task 1

Before building the combined joint model, we will first build up and
test all the necessary pieces (e.g., design, data, likelihood) for each
task/model separately. We will combine them in the final step once we
are confident the individual pieces each work correctly.

#### Make experimental design matrix for Task 1

Here we create an N x M data frame of our experimental design. Each row
represents one observation or trial and each column is a factor whose
levels code the experimental conditions for each trial. For example, the
following generates a design matrix for a simple 2 x 2 (stimulus x cue
condition) speed-accuracy trade-off experiment with column factors
`subject`, `stim`, and `cond1`. Fully factorial designs are particularly
easy to create by passing vectors of subjects and factor levels to
`expand_grid()`. More complex and unbalanced designs, and designs
containing randomization or sequential effects, may need to be
constructed manually (e.g., by assigning to cells the output of
`sample()` or `ifelse()`).

``` r
n_subs_t1 <- 5  # n subjects
n_obs_t1 <- 40  # n observations (trials) per subject per design cell
design$t1 <- expand_grid(subject = factor(rep(1:n_subs_t1, each = n_obs_t1)), 
                         stim = factor(c("s1", "s2")),
                         cond1 = factor(c("spd", "acc"))
)
design$t1
```

Here we pass our factor levels to `expand_grid()` to generate a
factorial combination of all subjects, stimuli, and speed-accuracy cue
conditions. Note that your empirical data should also be in a similarly
tidy format with column names and factor levels that match those in your
likelihood function.

#### Create a vector of parameter names

Now we want to create a named vector containing all the unique estimated
parameters in our design. First we extract the factor levels over which
the parameters will vary. This is done by passing a vector of factor
levels to `str_c()`, which appends them to the name of the target
parameter. This creates a unique text string label for each parameter in
the design. The likelihood and sampling functions use this to lookup
parameters and pull out corresponding values. Ultimately, this lets us
pass multiple similar models to the sampling functions without naming
conflicts. Here we let a and v vary over condition and keep t0 fixed
(i.e., we don’t pass it any factor levels via `unique()`).

``` r
p_names$t1 <- c(str_c("t1.a", unique(design$t1$cond1), sep = "."), 
                str_c("t1.v", unique(design$t1$cond1), sep = "."),
                str_c("t1.t0", sep = ".")
                )
p_names$t1
```

#### Set mean parameter values to simulate from (or to use as priors)

Note that (for now) the values of unestimated constants are set inside
the likelihood function.

``` r
p_vector$t1 <- c(2.0, 3.0,  # a's
                 1.0, 1.5,  # v's
                 0.2        # t0
                 )
names(p_vector$t1) <- p_names$t1
p_vector$t1
names(p_vector$t1)
```

Everything looks good.

Now we can construct the likelihood function for Task 1.

## Likelihood function Task 1

Although the following function is quite long, it is very modular. It is
made of three self-explanatory code blocks:

1.  Check inputs
2.  Likelihood loops
3.  Prepare output

Adding a model to the function involves adding its likelihood loop to
the ‘Likelihood loops’ block and appending the result to the `data` and
`out` objects before returning the simulated data or computing the
summed log-likelihood.

Notice that the relevant data is indexed by name (e.g., `data$t1$rt`).
This makes adding additional data sources easy - you simply append the
new data source to the data list and give it a name (e.g.,
`data$t1_EEG`, `data$t2_BOLD`). Then refer to it by that name when
writing the likelihood function.

#### Likelihood function for diffusion model of Task 1

``` r
ll_t1_ddm <- function(x, data, sample = FALSE) {
  # This function takes the following inputs:
  # 1. x = the (log of) a named vector of starting parameters.
  # 2. data = an experimental data frame or the experimental 
  #    design matrix created above if simulating data (i.e., 
  #    if sample = TRUE).
  #
  # This function returns the following outputs:
  # 1. The sum of the log-likelihoods (if sample = FALSE)
  # 2. A data frame of simulated data (if sample = TRUE)
  
  # Check inputs ------------------------------------------------------------
  
  # Undo the log transform
  x <- exp(x) 
  # Perform parameter checks
  if (any(data$t1$rt < x["t1.t0"])) {
    return(-1e10)
  }
  
  if (sample) {
    # Create output objects for simulated data
    data$t1$rt <- rep(NA, nrow(data$t1))
    data$t1$response <- rep(NA, nrow(data$t1))
  } else {
    # Create output objects for likelihoods
    out <- list()
    out$t1 <- numeric(nrow(data$t1))
  }
  
  # Likelihood loops --------------------------------------------------------
  
  # Task 1 loop -------------------------------------------------------------
  
  # Now we iterate over each trial in the data (or design matrix) and 
  # either simulate an outcome or calculate the likelihood given the 
  # current parameters
  for (i in 1:nrow(data$t1)) {
    # Extract the parameters that correspond to the current trial
    t1.a = x[paste("t1.a", data$t1$cond1[i], sep = ".")]
    t1.v = x[paste("t1.v", data$t1$cond1[i], sep = ".")]
    t1.t0 = x[paste("t1.t0")]
    
    if (sample) {
      # Simulate a trial from the model using the current parameters
      tmp <- rdiffusion(n = 1,
                        a = t1.a,
                        v = t1.v,
                        t0 = t1.t0
      )
      data$t1$rt[i] <- tmp$rt
      data$t1$response[i] <- tmp$response
    } else {
      # Calculate the likelihood (density) for the current parameters
      out$t1[i] <- ddiffusion(rt = data$t1$rt[i],
                              response = data$t1$response[i],
                              a = t1.a,
                              v = t1.v,
                              t0 = t1.t0
      )
    }
  }
  
  # Task 2 loop -------------------------------------------------------------
  
  # Here is where you insert additional likelihood loops for joint modeling.
  # Remember to append results produced here to `data` and `out` below.
  
  # Prepare output ----------------------------------------------------------
  
  if (sample) {
    # Output simulated data
    data$t1$response <- factor(data$t1$response)
    return(data$t1) # Remove index if returning multiple
  } else {
    # Remove bad likelihoods
    out <- out$t1
    bad <- (out < 1e-10) | (!is.finite(out))
    out[bad] <- 1e-10
    # Output summed log-likelihood
    out <- sum(log(out))
    return(out)
  }
}
```

## Test likelihood function

Here we check that our likelihood function returns the expected items
(i.e., a simulated data frame or a summed log-likelihood).

#### Simulate data from the design matrix

``` r
sims$t1 <- ll_t1_ddm(x = log(p_vector$t1), 
                     data = design, 
                     sample = TRUE
                     )
sims$t1
```

#### Return likelihood

``` r
ll_t1_ddm(x = log(p_vector$t1), 
          data = sims, 
          sample = FALSE
          )
```

## Explore data

#### Load data

Here we would usually read in and tidy up our empirical data to get it
into a similar format as the data frame we just simulated and stored in
`sims`. For now let’s just use the simulated data as our `data`.

``` r
data <- sims
data$t1
```

#### Cell counts

``` r
table(data$t1$stim, data$t1$cond1)
```

#### Mean RT

``` r
data$t1 %>% 
  group_by(stim, cond1) %>% 
  summarise(rt = mean(rt))
```

#### RT histogram

``` r
data$t1 %>%
  ggplot(aes(x = rt, fill = stim)) +
  geom_histogram(stat = 'bin') +
  facet_wrap(. ~ cond1) +
  scale_x_continuous(limits = c(0, 3)) +
  theme_minimal()
```

**TO DO**: Add other useful data checks here

Now we can do the same for Task 2…

## Setup Task 2

#### Make experimental design matrix for Task 2

``` r
n_subs_t2 <- 5
n_obs_t2 <- 40
design$t2 <- expand_grid(subject = factor(rep(1:n_subs_t2, each = n_obs_t2)), 
                         stim = factor(c("s1", "s2")),
                         cond1 = factor(c("spd", "acc"))
)
design$t2
```

#### Create a vector of parameter names

``` r
p_names$t2 <- c(str_c("t2.a", unique(design$t2$cond1), sep = "."),
                str_c("t2.v", unique(design$t2$cond1), sep = "."),
                str_c("t2.t0", sep = ".")
                )
p_names$t2
```

#### Set mean parameter values to simulate from (or to use as priors)

Note that (for now) the values of unestimated constants are set inside
the likelihood function. Here we will just set them to half of model 1’s
values.

``` r
p_vector$t2 <- c(2.0, 3.0,  # a's
                 1.0, 1.5,  # v's
                 0.2        # t0
                 )/2
names(p_vector$t2) <- p_names$t2
p_vector$t2
```

## Likelihood function Task 2

#### Likelihood function for diffusion model of Task 2

``` r
ll_t2_ddm <- function(x, data, sample = FALSE) {
  # This function takes the following inputs:
  # 1. x = the (log of) a named vector of starting parameters.
  # 2. data = an experimental data frame or the experimental 
  #    design matrix created above if simulating data (i.e., 
  #    if sample = TRUE).
  #
  # This function returns the following outputs:
  # 1. The sum of the log-likelihoods (if sample = FALSE)
  # 2. A data frame of simulated data (if sample = TRUE)
  
  # Check inputs ------------------------------------------------------------
  
  # Undo the log transform
  x <- exp(x) 
  # Perform parameter checks
  if (any(data$t2$rt < x["t2.t0"])) {
    return(-1e10)
  }
  
  if (sample) {
    # Create output objects for simulated data
    data$t2$rt <- rep(NA, nrow(data$t2))
    data$t2$response <- rep(NA, nrow(data$t2))
  } else {
    # Create output objects for likelihoods
    out <- list()
    out$t2 <- numeric(nrow(data$t2))
  }
  
  # Likelihood loops --------------------------------------------------------
  
  # Task 2 loop -------------------------------------------------------------
  
  # Now we iterate over each trial in the data (or design matrix) and 
  # either simulate an outcome or calculate the likelihood given the 
  # current parameters
  for (i in 1:nrow(data$t2)) {
    # Extract the parameters that correspond to the current trial
    t2.a = x[paste("t2.a", data$t2$cond1[i], sep = ".")]
    t2.v = x[paste("t2.v", data$t2$cond1[i], sep = ".")]
    t2.t0 = x[paste("t2.t0")]
    
    if (sample) {
      # Simulate a trial from the model using the current parameters
      tmp <- rdiffusion(n = 1,
                        a = t2.a,
                        v = t2.v,
                        t0 = t2.t0
      )
      data$t2$rt[i] <- tmp$rt
      data$t2$response[i] <- tmp$response
    } else {
      # Calculate the likelihood (density) for the current parameters
      out$t2[i] <- ddiffusion(rt = data$t2$rt[i],
                              response = data$t2$response[i],
                              a = t2.a,
                              v = t2.v,
                              t0 = t2.t0
      )
    }
  }
  
  # Prepare output ----------------------------------------------------------
  
  if (sample) {
    # Output simulated data
    data$t2$response <- factor(data$t2$response)
    return(data$t2) # Remove index if returning multiple
  } else {
    # Remove bad likelihoods
    out <- out$t2
    bad <- (out < 1e-10) | (!is.finite(out))
    out[bad] <- 1e-10
    # Output summed log-likelihood
    out <- sum(log(out))
    return(out)
  }
}
```

## Test likelihood function

#### Simulate data from the design matrix

``` r
sims$t2 <- ll_t2_ddm(x = log(p_vector$t2), 
                     data = design, 
                     sample = TRUE
                     )
sims$t2
```

#### Return likelihood

``` r
ll_t2_ddm(x = log(p_vector$t2), 
          data = sims, 
          sample = FALSE
          )
```

## Explore data

Again we treat `sims` as `data`

``` r
data <- sims
data$t2
```

#### Cell counts

``` r
table(data$t2$stim, data$t2$cond1)
```

#### Mean RT

``` r
data$t2 %>% 
  group_by(stim, cond1) %>% 
  summarise(rt = mean(rt))
```

#### RT histogram

``` r
data$t2 %>%
  ggplot(aes(x = rt, fill = stim)) +
  geom_histogram(stat = 'bin') +
  facet_wrap(. ~ cond1) +
  scale_x_continuous(limits = c(0, 3)) +
  theme_minimal()
```

## Create joint model

Now we are ready to combine the two models into a single likelihood.
First let’s make a combined parameter vector containing all parameters.

#### Make combined parameter vector

``` r
p_vector <- c(p_vector$t1, p_vector$t2)
p_vector
```

## Setup for sampling

#### Get the names of the parameters we want to sample

``` r
pars <- names(p_vector)
pars
```

#### Set priors for the mean and variance of hyper-level multivariate normal

``` r
priors <- list(
  theta_mu_mean = rep(0, length(pars)),
  theta_mu_var = diag(rep(1, length(pars)))
)
priors
```

## Joint likelihood function

#### Likelihood function for joint diffusion model of Tasks 1 & 2

``` r
ll_joint_ddm <- function(x, data, sample = FALSE) {
  # This function takes the following inputs:
  # 1. x = the (log of) a named vector of starting parameters
  # 2. data = a named list of data frames or experimental design  
  #    matrices if simulating data (i.e., if sample = TRUE)
  #
  # This function returns the following outputs:
  # 1. The sum of the log-likelihoods (if sample = FALSE)
  # 2. A list of data frames of simulated data (if sample = TRUE)
  
  # Check inputs ------------------------------------------------------------
  
  # Undo the log transform
  x <- exp(x) 
  # Perform parameter checks 
  # Task 1 checks
  if (any(data$t1$rt < x["t1.t0"])) {
    return(-1e10)
  }
  # Task 2 checks
  if (any(data$t2$rt < x["t2.t0"])) {
    return(-1e10)
  }
  
  if (sample) {
    # Create output objects for simulated data
    # Task 1 sim outputs
    data$t1$rt <- rep(NA, nrow(data$t1))
    data$t1$response <- rep(NA, nrow(data$t1))
    # Task 2 sim outputs
    data$t2$rt <- rep(NA, nrow(data$t2))
    data$t2$response <- rep(NA, nrow(data$t2))
  } else {
    # Create output objects for likelihoods
    out <- list()
    out$t1 <- numeric(nrow(data$t1))
    out$t2 <- numeric(nrow(data$t2))
  }
  
  # Likelihood loops --------------------------------------------------------
  
  # Task 1 loop -------------------------------------------------------------
  
  # Now we iterate over each trial in the data (or design matrix) and 
  # either simulate an outcome or calculate the likelihood given the 
  # current parameters
  for (i in 1:nrow(data$t1)) {
    # Extract the parameters that correspond to the current trial
    t1.a = x[paste("t1.a", data$t1$cond1[i], sep = ".")]
    t1.v = x[paste("t1.v", data$t1$cond1[i], sep = ".")]
    t1.t0 = x[paste("t1.t0")]
    
    if (sample) {
      # Simulate a trial from the model using the current parameters
      tmp_t1 <- rdiffusion(n = 1,
                           a = t1.a,
                           v = t1.v,
                           t0 = t1.t0
      )
      data$t1$rt[i] <- tmp_t1$rt
      data$t1$response[i] <- tmp_t1$response
    } else {
      # Calculate the likelihood (density) given the current parameters
      out$t1[i] <- ddiffusion(rt = data$t1$rt[i],
                              response = data$t1$response[i],
                              a = t1.a,
                              v = t1.v,
                              t0 = t1.t0
      )
    }
  }
  
  # Task 2 loop -------------------------------------------------------------
  
  # Again we iterate over each trial and either simulate an outcome
  # or calculate the likelihood given the current parameters
  for (i in 1:nrow(data$t2)) {
    # Extract the parameters that correspond to the current trial
    t2.a = x[paste("t2.a", data$t2$cond1[i], sep = ".")]
    t2.v = x[paste("t2.v", data$t2$cond1[i], sep = ".")]
    t2.t0 = x[paste("t2.t0")]
    
    if (sample) {
      # Simulate a trial from the model using the current parameters
      tmp_t2 <- rdiffusion(n = 1,
                           a = t2.a,
                           v = t2.v,
                           t0 = t2.t0
      )
      data$t2$rt[i] <- tmp_t2$rt
      data$t2$response[i] <- tmp_t2$response
    } else {
      # Calculate the likelihood (density) given the current parameters
      out$t2[i] <- ddiffusion(rt = data$t2$rt[i],
                              response = data$t2$response[i],
                              a = t2.a,
                              v = t2.v,
                              t0 = t2.t0
      )
    }
  }
  
  # Prepare output ----------------------------------------------------------
  
  if (sample) {
    # Output simulated data
    data$t1$response <- factor(data$t1$response)
    data$t2$response <- factor(data$t2$response)
    return(data)
  } else {
    # Remove bad likelihoods
    out_all <- c(out$t1, out$t2)
    bad <- (out_all < 1e-10) | (!is.finite(out_all))
    out_all[bad] <- 1e-10
    # Output summed log-likelihood
    # print(paste("ll t1: ", sum(log(out$t1))))  # model 1 ll
    # print(paste("ll t2: ", sum(log(out$t2))))  # model 2 ll
    out_all <- sum(log(out_all))
    return(out_all)
  }
}
```

## Test likelihood function

#### Simulate data from the design matrix

``` r
sims <- ll_joint_ddm(x = log(p_vector), data = data, sample = TRUE)
sims
```

#### Return likelihood

``` r
ll_joint_ddm(x = log(p_vector), data = sims, sample = FALSE)
```

## Setup sampling

#### Initialize sampler

``` r
sampler <- pmwgs(
  data = data,
  pars = pars,
  prior = priors,
  ll_func = ll_joint_ddm
)
```

#### Set start points (optional)

``` r
start_points <- list(
  mu = log(p_vector),
  sig2 = diag(rep(.01, length(pars)))
)
```

#### Initialize start points

``` r
sampler <- init(
  sampler, 
  start_mu = start_points$mu,
  start_sig = start_points$sig2
)
```

## Run sampling

Note that for large models, the following stages are best run in
parallel on a research computing/server grid.

#### Stage 1: Burn-in

``` r
burned <- run_stage(
  sampler, 
  stage = "burn",
  iter = 500,
  particles = 2000,
  epsilon = .5, 
  n_cores = 10
)
```

#### Stage 2: Adaptation

``` r
adapted <- run_stage(
  burned, 
  stage = "adapt", 
  iter = 10000, 
  particles = 2000,
  epsilon = .5,
  n_cores = 10
)
```

#### Stage 3: Sampling

``` r
sampled <- run_stage(
  adapted, 
  stage = "sample",
  iter = 1000, 
  particles = 200,
  epsilon = .5,
  n_cores = 10
)
```

#### Save to samples directory

``` r
save(sampled, file = "samples/samples_ddm_combined_test.RData")
```

#### Next steps

Now that we have our samples, we would now run sampling diagnositics,
check model fit by simulating posterior predictives, and exploring
relationships between tasks and models. Future versions of this tutorial
will include convenience functions and example code for performing these
analyses.

See <https://newcastlecl.github.io/samplerDoc/> for further information
about the PMwG sampler.
