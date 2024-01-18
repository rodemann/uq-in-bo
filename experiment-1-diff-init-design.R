# compare agent with shapleys to agent without 
# both agents have own interal utility model which is simulated by another BO
# agents have different knowledge than automatic BO
# agent 1 intervenes each 3rd intervention (or randomly?)
# agent 2 intervenes only if shapleys fulfill certain conditions
# e.g., internal shapleys suggest lifting gain to be more relevant than lowering gain
# i.e., only intervene if shapley ratio diverges from internally learned ratio
# (similar setup as in human-bo team paper)

#TODO possible extension: third agent that uses information on proposals but not on shapleys

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

# first create ground truth by estimating hypersurrogate utility from data
# simulated_data_x <- read_csv("C:/Users/Julian Rodemann/Downloads/simulated_data_x.csv")
# simulated_data_pref <- read_csv("C:/Users/Julian Rodemann/Downloads/simulated_data_pref.csv")
# save(simulated_data_x, file = "data/simulated_data_x.csv")
# save(simulated_data_pref, file = "data/simulated_data_pref.csv")

load(file = "data/simulated_data_pref.Rds")
load(file = "data/simulated_data_x.Rds")
# 
# simulated_data_x <- read_csv("data/simulated_data_x.csv")
# simulated_data_pref <- read_csv("data/simulated_data_pref.csv")


# clean redundancies
simulated_data_pref = simulated_data_pref[,-1]
simulated_data_x = simulated_data_x[,-1]
simulated_data_x = simulated_data_x[-201,]

# standardize 
simulated_data_x$LOG = (simulated_data_x$LOG - mean(simulated_data_x$LOG))/sd(simulated_data_x$LOG)  
simulated_data_x$LIG = (simulated_data_x$LIG - mean(simulated_data_x$LIG))/sd(simulated_data_x$LIG)  

# #learn (hyper)surrogate 
lrn_hyper = makeLearner("regr.km", predict.type = "se", covtype = "exp", control = list(trace = FALSE))
#lrn_hyper = makeLearner("regr.randomForest", predict.type = "se")

# 
# # utility function
# mean_vector <- rnorm(2)
# # Generate a random 2x2 covariance matrix
# cov_matrix <- matrix(rnorm(4), nrow = 2)
# # Ensure the covariance matrix is positive definite (if necessary)
# cov_matrix <- cov_matrix %*% t(cov_matrix)
# # get cor matrix
# V = cov_matrix
# solve(diag(sqrt(diag(V)))) %*% V %*% solve(diag(sqrt(diag(V))))
# 
# utility = function(x){
#   # bivariate kernel density estim
#   #Compositional::mkde(x %>% as.matrix)
#   # dummy
#   #3*simulated_data_x$LOG - simulated_data_x$LIG^2
#   - dmvnorm(simulated_data_x, mean = mean_vector, sigma = cov_matrix )
# }
# # Evaluate the PDF of the multivariate normal distribution at the specified data points


#utilities = utility(simulated_data_x)
#utilities = apply(simulated_data_x, 1, utility)
utilities = 3*simulated_data_x$LOG - simulated_data_x$LIG^6


hyper_data = cbind(utilities, simulated_data_x)
# subsample
hyper_data = hyper_data[sample.int(200,40),]

Nuggets = 1e-8
lrn = setHyperPars(learner = lrn_hyper, nugget=Nuggets)


hyper_estim = makeRegrTask(data = hyper_data, target = "utilities")
hyper_model = train(lrn_hyper, hyper_estim)
#features = hyper_model$features
features = names(simulated_data_x)

var_function = function(x) 2*abs(x)

estimated_hyper = function(x){
  #browser()
  x = matrix(x, nrow = 1)
  sd = var_function(x) %>% sum %>% sqrt 
  newdata = as.data.frame(x)
  names(newdata) = features
  mean = predict(hyper_model, newdata = newdata) %>% getPredictionResponse()
  #mean = apply(newdata, 1, utility)
  #mean = utility(simulated_data_x)
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
################
# end of hypersurrogate 
################

###############
# begin BO
##
budget = 40
init_design_size = 10
parameter_set = getParamSet(obj_fun)


####
## auto BO
# same design for all approaches
design <- generateDesign(n = init_design_size, par.set = parameter_set, fun = lhs::randomLHS)
# same final evaluation method
#ctrl <- makeMBOControl(final.method = "best.true.y", final.evals = 5)
infill_crit = makeMBOInfillCritCB(cb.lambda = 1)
# set Control Argument of BO 
# store all models (here; 1:2 because only one iter internally)  
ctrl = makeMBOControl(propose.points = 1L, store.model.at = 1:2)
ctrl = setMBOControlInfill(ctrl, crit = infill_crit, opt = "focussearch", 
                           opt.focussearch.points = 2000, opt.focussearch.maxit = 1)
lrn = makeLearner("regr.km", predict.type = "se", covtype = "powexp", control = list(trace = FALSE))
#lrn = makeLearner("regr.randomForest", predict.type = "se")
# ensure numerical stability in km {DiceKriging} cf. github issue and recommendation by Bernd Bischl 
y = apply(design, 1, obj_fun)
Nuggets = 1e-8*var(y)
lrn = setHyperPars(learner = lrn, nugget=Nuggets)

plot(design$x1, y)
plot(design$x2, y)




####
## agent simulation 
init_design_size_agent = 20
design_agent <- generateDesign(n = init_design_size_agent, par.set = parameter_set, fun = lhs::maximinLHS)
#ctrl <- makeMBOControl(final.method = "best.true.y", final.evals = 5)
infill_crit_agent = makeMBOInfillCritCB(cb.lambda = 4)
# set Control Argument of BO 
# store all models (here; 1:2 because only one iter internally)  
ctrl_agent = makeMBOControl(propose.points = 1L, store.model.at = 1:2)
ctrl_agent = setMBOControlInfill(ctrl, crit = infill_crit, opt = "focussearch", 
                                 opt.focussearch.points = 2000, opt.focussearch.maxit = 1)
lrn = makeLearner("regr.km", predict.type = "se", covtype = "gauss", control = list(trace = FALSE))
#lrn = makeLearner("regr.randomForest", predict.type = "se")
# ensure numerical stability in km {DiceKriging} cf. github issue and recommendation by Bernd Bischl 
y = apply(design, 1, obj_fun)
Nuggets = 1e-8*var(y)
lrn = setHyperPars(learner = lrn, nugget=Nuggets)


ctrl_agent = setMBOControlTermination(ctrl_agent, iters = 1)


# initial setup of agent
initial_iters_agent = 200
ctrl_agent = makeMBOControl(propose.points = 1L, store.model.at = 1:initial_iters_agent)
ctrl_agent = setMBOControlTermination(ctrl_agent, iters = initial_iters_agent)
res_mbo_agent = mbo(fun = obj_fun, design = design_agent, control = ctrl_agent, learner = lrn_agent, show.info = F)
shapleys = ShapleyMBO(res.mbo = res_mbo_agent, iter.interest = 1:initial_iters_agent, contribution = TRUE)

x1_ind = subset(1:nrow(shapleys), 1:nrow(shapleys) %% 2 == 1)
x2_ind = subset(1:nrow(shapleys), 1:nrow(shapleys) %% 2 == 0)
mean_shapley_x1 = shapleys$phi_mean_scaled[x1_ind] %>% mean
mean_shapley_x2 = shapleys$phi_mean_scaled[x2_ind] %>% mean

shapley_ratio_agent = mean_shapley_x1 / mean_shapley_x2




baseline_results = list()
shapleyBO_results = list()
baseline_opts = list()
shapley_opts = list()
number_interventions_tot = list()

sample = generateDesign(n = 2000, par.set = parameter_set, fun = lhs::maximinLHS)

n_exp= 10
designs_list = list()
for (i in 1:n_exp) {
  design <- sample[sample.int(nrow(sample),init_design_size),]
  designs_list[[i]] = design
}
designs_list_shapley = list()
for (i in 1:n_exp) {
  design <- sample[sample.int(nrow(sample),init_design_size),]
  designs_list_shapley[[i]] = design
}

for (i in 1:n_exp) {
  #browser()
  design = designs_list[[i]]
  #source("exp-baseline-agent.R")
  # this script runs the baseline BO for an agent who has no access to shapleyBO
  #
  ## BO loop
  for (b in 1:budget) {
    #browser()
    ctrl = setMBOControlTermination(ctrl, iters = 1)
    res_mbo = mbo(fun = obj_fun, design = design, control = ctrl, learner = lrn, show.info = F)
    mbo_design = res_mbo$opt.path %>% as.data.frame()
    if(b %% 3 == 0){
      ctrl_agent = setMBOControlTermination(ctrl_agent, iters = 1)
      res_mbo_agent = mbo(fun = obj_fun, design = design_agent, control = ctrl_agent, learner = lrn_agent, show.info = F)
      mbo_design_agent = res_mbo_agent$opt.path %>% as.data.frame()
      proposal_agent = mbo_design_agent[nrow(mbo_design_agent),1:2]
      design = rbind(design, proposal_agent)
    }else{
      design = mbo_design[,1:2]
    }
  }
  # optimum
  #res_mbo$y
  baseline_res = res_mbo
  
  
  design = designs_list_shapley[[i]]
  print(design)
  #source("exp-shapley-agent.R")
  # this script runs the BO for an agent who has access to shapleyBO
  #
  number_interventions = 0
  ## BO loop
  for (b in 1:budget) {
    ctrl = setMBOControlTermination(ctrl, iters = 1)
    ctrl_agent = makeMBOControl(propose.points = 1L, store.model.at = 1:2)
    # auto BO:
    res_mbo = mbo(fun = obj_fun, design = design, control = ctrl, learner = lrn, show.info = F)
    mbo_design = res_mbo$opt.path %>% as.data.frame()
    shapleys = ShapleyMBO(res.mbo = res_mbo, iter.interest = 1, contribution = TRUE)
    if(shapleys$phi_mean_scaled[1] == 0 & shapleys$phi_mean_scaled[2] == 0){
      shapley_ratio_bo = 1
    }else{
      shapley_ratio_bo = shapleys$phi_mean_scaled[1] / shapleys$phi_mean_scaled[2]
    }
    # human bo interface: does the proposal align with agent's knowledge?
    if(shapley_ratio_agent/shapley_ratio_bo > 2 |
       shapley_ratio_agent/shapley_ratio_bo < 0.5 ){
      number_interventions = number_interventions +1
      ctrl_agent = setMBOControlTermination(ctrl_agent, iters = 1)
      # agent BO:
      res_mbo_agent = mbo(fun = obj_fun, design = design_agent, control = ctrl_agent, learner = lrn_agent, show.info = F)
      mbo_design_agent = res_mbo_agent$opt.path %>% as.data.frame()
      proposal_agent = mbo_design_agent[nrow(mbo_design_agent),1:2]
      design = rbind(design, proposal_agent)
      # possible extension: update shapleys
    }else{
      design = mbo_design[,1:2]
    }
  }
  shapley_res = res_mbo
  
  baseline_results[[i]] = baseline_res
  shapleyBO_results[[i]] = shapley_res
  number_interventions_tot[[i]] = number_interventions
  baseline_opts[[i]] = baseline_res$y
  shapley_opts[[i]] = shapley_res$y
  print(i)
}

baseline_opts %>% unlist %>% mean
shapley_opts %>% unlist %>% mean





#baseline_results[[10]]

## possible extensions: different infill crits
# infill_crit = makeMBOInfillCritUACB(cb.lambda = 5, 
#                                     cb.rho = 1,
#                                     cb.alpha = 10,
#                                     base_kernel= "matern3_2", 
#                                     imprecision= 1, 
#                                     noise_proxy_fun = var_function)
# 
# infill_crit = makeMBOInfillCritRACB(cb.lambda = 0.5, 
#                                     cb.alpha = 0.4,
#                                     noise_proxy_fun = var_function)


