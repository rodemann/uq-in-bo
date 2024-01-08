library(mlrMBO)
library(ggplot2)
library(dplyr)
library(smoof)
library(iml)
library(readr)
library(readxl)
source("R/makeMBOInfillCritUACB.R")
source("R/initCrit.InfillCritUACB.R")
source("R/ShapleyMBO.R")
source("R/ShapleyMBO_racb.R")

#source("R/_Explore_Exploit_Measures/xplxpl-jr.R")

fun = smoof::makeAlpine02Function(2)
#fun = smoof::makeAlpine01Function(2)

sf_optimization_data <- read_excel("data/sf_optimization_data.xlsx", 
                                   sheet = "Discrete sweep", col_names = FALSE)

sample_patient_1 <- sf_optimization_data[3:5,3:11] %>% t()  

sample_patient_1 <- apply(sample_patient_1, 2, as.numeric) %>% as.data.frame()

colnames(sample_patient_1) <- c("step_size_commanded", "step_size_measured", "metabolic_cost")
#rownames(sample_patient_1) <- 1:9


#learn (hyper)surrogate 
lrn_hyper = makeLearner("regr.km", predict.type = "se", covtype = "matern3_2", control = list(trace = FALSE))

#lrn_hyper = makeLearner("regr.GPfit", predict.type = "response")

hyper_data = sample_patient_1[,2:3] 
hyper_estim = makeRegrTask(data = hyper_data, target = "metabolic_cost")
hyper_model = train(lrn_hyper, hyper_estim)
features = hyper_model$features

var_function = function(x) 4*abs(x)

estimated_hyper = function(x){
  # x = matrix(x, nrow = 1)
  sd = var_function(x) %>% sqrt
  newdata = as.data.frame(x)
  names(newdata) = features
  mean = predict(hyper_model, newdata = newdata) %>% getPredictionResponse()
  rnorm(1, mean = mean, sd = sd)
  
}




obj_fun = makeSingleObjectiveFunction(name = "metabolic cost", 
                                      fn = estimated_hyper, has.simple.signature = TRUE,
                                      par.set = makeNumericParamSet(
                                        len = 1, id = "x", 
                                        lower = -25, upper = 25,
                                        vector = TRUE)
)


budget = 10
init_design_size = 10
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
ctrl = setMBOControlInfill(ctrl, crit = infill_crit, opt = "focussearch", 
                           opt.focussearch.points = 1000, opt.focussearch.maxit = 1)


lrn = makeLearner("regr.km", predict.type = "se", covtype = "matern3_2", control = list(trace = FALSE))
# ensure numerical stability in km {DiceKriging} cf. github issue and recommendation by Bernd Bischl 
y = apply(design, 1, obj_fun)
Nuggets = 1e-8*var(y)
lrn = setHyperPars(learner = lrn, nugget=Nuggets)


res_mbo = mbo(fun = obj_fun, design = design, control = ctrl, learner = lrn)

raw_shapleys = ShapleyMBO_racb(res.mbo = res_mbo, contribution = TRUE, noise_proxy_fun = var_function)

#xplxpl(res_mbo)


