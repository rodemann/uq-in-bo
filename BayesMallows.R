# Load the plackettLuce library
library(PlackettLuce)

# Create a dataset with preferences data
data <- data.frame(
  X_1_1 = c(1.2, 2.5, 3.8, 4.1, 5.0),
  X_1_2 = c(3.5, 1.8, 4.2, 2.9, 4.8),
  Preference = c(1, 0, 1, 0, 1)  # 1 for preference, 0 for no preference
)

# Define a utility function that models the preference probabilities
utility_function <- function(X_1_1, X_1_2, params) {
  utility_1 <- params[1] * X_1_1
  utility_2 <- params[1] * X_1_2
  probability <- exp(utility_1) / (exp(utility_1) + exp(utility_2))
  return(probability)
}

# Fit the Plackett-Luce model
plackett_luce_fit <- PlackettLuce(data, utility_function, control = list(maxit = 1000))

# Get the estimated parameter values, which represent the effect of X_1
estimated_params <- plackett_luce_fit$par

# Now you can use the estimated parameter(s) to estimate the effect of X_1
print(paste("Estimated effect of X_1:", estimated_params[1]))
