library(mlrMBO)
library(ggplot2)
library(dplyr)
library(smoof)
library(iml)
library(readr)
library(readxl)
library(mvtnorm)
source("R/makeMBOInfillCritUACB.R")
source("R/initCrit.InfillCritUACB.R")
source("R/makeMBOInfillCritRACB.R")
source("R/initCrit.InfillCritRACB.R")
source("R/ShapleyMBO.R")
#source("R/_Explore_Exploit_Measures/xplxpl-jr.R")


simulated_data_x <- read_csv("C:/Users/Julian Rodemann/Downloads/simulated_data_x.csv")
simulated_data_pref <- read_csv("C:/Users/Julian Rodemann/Downloads/simulated_data_pref.csv")
# clean redundancies
simulated_data_pref = simulated_data_pref[,-1]
simulated_data_x = simulated_data_x[,-1]
simulated_data_x = simulated_data_x[-201,]

# standardize 
simulated_data_x$LOG = (simulated_data_x$LOG - mean(simulated_data_x$LOG))/sd(simulated_data_x$LOG)  
simulated_data_x$LIG = (simulated_data_x$LIG - mean(simulated_data_x$LIG))/sd(simulated_data_x$LIG)  

# #learn (hyper)surrogate 
# lrn_hyper = makeLearner("regr.km", predict.type = "se", covtype = "exp", control = list(trace = FALSE))
# lrn_hyper = makeLearner("regr.randomForest", predict.type = "se")


# utility function
mean_vector <- rnorm(2)
# Generate a random 2x2 covariance matrix
cov_matrix <- matrix(rnorm(4), nrow = 2)
# Ensure the covariance matrix is positive definite (if necessary)
cov_matrix <- cov_matrix %*% t(cov_matrix)
# get cor matrix
V = cov_matrix
solve(diag(sqrt(diag(V)))) %*% V %*% solve(diag(sqrt(diag(V))))


# Python Original, (c) Philipp
# # Get the correct dimnesions
# d = self.bounds.shape[0]
# # Sample mean vector
# mu = self.rs.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(1, d))
# # Sample covariance matrices
# cv = ds.make_spd_matrix(n_dim=d, random_state=self.rs) * scale # make_psd_matrix(d, scale, rs)
# 
# 
# def _generate_mean_covariance(self, scale):
#   """Generate random mean vector and PSD covariance matrix with dimension d"""
# 
# # Get the correct dimnesions
# d = self.bounds.shape[0]
# # Sample mean vector
# mu = self.rs.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(1, d))
# # Sample covariance matrices
# cv = ds.make_spd_matrix(n_dim=d, random_state=self.rs) * scale # make_psd_matrix(d, scale, rs)
# 
# return mu, cv
# 
# def make_function(self, k=5, scale=2e3):
#   """Utility function factory """
# mu, cv = self._generate_mean_covariance(scale)
# 
# return lambda x: k*scale*mvn.pdf(x, mean=mu.flatten(), cov=cv)
# 




utility = function(x){
  # bivariate kernel density estim
  #Compositional::mkde(x %>% as.matrix)
  # dummy
  #3*x[1] - x[2]^2
  - dmvnorm(simulated_data_x, mean = mean_vector, sigma = cov_matrix )
}
# Evaluate the PDF of the multivariate normal distribution at the specified data points

utilities = utility(simulated_data_x)
#utilities = apply(simulated_data_x, 1, utility)

hyper_data = cbind(utilities, simulated_data_x)
# subsample
hyper_data = hyper_data[sample.int(200,20),]

hyper_estim = makeRegrTask(data = hyper_data, target = "utilities")
#hyper_model = train(lrn_hyper, hyper_estim)
#features = hyper_model$features
features = names(simulated_data_x)

var_function = function(x) 4*abs(x)

estimated_hyper = function(x){
#browser()
x = matrix(x, nrow = 1)
sd = var_function(x) %>% sum %>% sqrt 
newdata = as.data.frame(x)
names(newdata) = features
#mean = predict(hyper_model, newdata = newdata) %>% getPredictionResponse()
#mean = apply(newdata, 1, utility)
mean = utility(simulated_data_x)
rnorm(1, mean = mean, sd = sd)
}


obj_fun = makeSingleObjectiveFunction(name = "exo utility", 
                                    fn = estimated_hyper, has.simple.signature = TRUE,
                                    par.set = makeNumericParamSet(
                                      len = 2, id = "x", 
                                      lower = 0, upper = max(simulated_data_x),
                                      vector = TRUE),
                                      minimize = TRUE 
)


budget = 20
init_design_size = 100
parameter_set = getParamSet(obj_fun)

# same design for all approaches
design <- generateDesign(n = init_design_size, par.set = parameter_set, fun = lhs::randomLHS)

ctrl <- makeMBOControl(final.method = "best.true.y", final.evals = 5)

# set Control Argument of BO 
ctrl = makeMBOControl(propose.points = 1L,
                    store.model.at = 1:(budget+1))

ctrl = setMBOControlTermination(ctrl, iters = budget)
infill_crit = makeMBOInfillCritUACB(cb.lambda = 5, 
                                  cb.rho = 1,
                                  cb.alpha = 10,
                                  base_kernel= "matern3_2", 
                                  imprecision= 1, 
                                  noise_proxy_fun = var_function)

infill_crit = makeMBOInfillCritRACB(cb.lambda = 0.5, 
                                  cb.alpha = 0.4,
                                  noise_proxy_fun = var_function)
infill_crit = makeMBOInfillCritCB(cb.lambda = 2)

ctrl = setMBOControlInfill(ctrl, crit = infill_crit, opt = "focussearch", 
                         opt.focussearch.points = 5000, opt.focussearch.maxit = 1)


lrn = makeLearner("regr.km", predict.type = "se", covtype = "powexp", control = list(trace = FALSE))
#lrn = makeLearner("regr.randomForest", predict.type = "se")

# ensure numerical stability in km {DiceKriging} cf. github issue and recommendation by Bernd Bischl 
y = apply(design, 1, obj_fun)
Nuggets = 1e-8*var(y)
lrn = setHyperPars(learner = lrn, nugget=Nuggets)


res_mbo = mbo(fun = obj_fun, design = design, control = ctrl, learner = lrn)

#raw_shapleys = ShapleyMBO_racb(res.mbo = res_mbo, contribution = TRUE, noise_proxy_fun = var_function)
shapleys = ShapleyMBO(res.mbo = res_mbo, iter.interest = 1:budget, contribution = TRUE)

shapleys_sel <- select(shapleys, "iter","feature", "phi_mean_scaled", "phi_se_scaled", "phi_cb")

shapleys

shapleys_sel

# x1_ind = subset(1:nrow(shapleys_sel), 1:nrow(shapleys_sel) %% dim == 1)
# x2_ind = subset(1:nrow(shapleys), 1:nrow(shapleys) %% dim == 2)
# x3_ind = subset(1:nrow(shapleys), 1:nrow(shapleys) %% dim == 3)
# x4_ind = subset(1:nrow(shapleys), 1:nrow(shapleys) %% dim == 0)
# 
# shapleys$phi_mean_scaled[x1_ind] %>% mean
# shapleys$phi_mean_scaled[x2_ind] %>% mean
# shapleys$phi_mean_scaled[x3_ind] %>% mean
# shapleys$phi_mean_scaled[x4_ind] %>% mean
# 
# shapleys$phi_se_scaled[x1_ind] %>% mean
# shapleys$phi_se_scaled[x2_ind] %>% mean
# shapleys$phi_se_scaled[x3_ind] %>% mean
# shapleys$phi_se_scaled[x4_ind] %>% mean
# 
# shapleys$phi_noise_scaled[x1_ind] %>% mean
# shapleys$phi_noise_scaled[x2_ind] %>% mean
# shapleys$phi_noise_scaled[x3_ind] %>% mean
# shapleys$phi_noise_scaled[x4_ind] %>% mean
