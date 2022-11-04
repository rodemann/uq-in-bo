library(mlrMBO)
library(ggplot2)
library(dplyr)
library(smoof)
library(iml)

source("R/makeMBOInfillCritRACB.R")
source("R/initCrit.InfillCritRACB.R")
source("R/ShapleyMBO_racb.R")
source("R/ShapleyMBO.R")
source("R/_Explore_Exploit_Measures/xplxpl-jr.R")


dim = 2
fun = smoof::makeAlpine02Function(dim)
#fun = smoof::makeAlpine01Function(dim)
fun = smoof::makeHyperEllipsoidFunction(dim)
# autoplot(fun)
# plot3D(fun)

var_function = function(x) 12*abs(x[1]-15) + 0.4*abs(x[2]-15)

obj_fun = function(x) {
  rnorm(1, mean = fun(x), sd = var_function(x) %>% sqrt())
}

obj_fun = makeSingleObjectiveFunction(name = "noisy 2D parable", 
                                      fn = obj_fun, has.simple.signature = TRUE,
                                      par.set = makeNumericParamSet(
                                        len = dim, id = "x", 
                                        lower = rep(-15, dim), upper = rep(15, dim),
                                        vector = TRUE)
)


# test for one feature only --> envokes this error:
#Only 1 feature was provided. The iml package is only useful and works for multiple features. 
# obj_fun = makeCosineMixtureFunction(1)
# obj_fun = convertToMinimization(obj_fun)

#viz
plot3D(obj_fun, length.out = 100)
autoplot(obj_fun)


## noise estimation
parameter_set = getParamSet(obj_fun)
noise_design <- generateDesign(n = 60, par.set = parameter_set, fun = lhs::randomLHS)

noise_evals = 100
get_noise = function(x) {
  replicate(noise_evals, obj_fun(x)) %>% var()
}
noise = apply(noise_design, 1, get_noise)
#learn noise by GP
lrn_noise = makeLearner("regr.km", covtype = "powexp", predict.type = "response", optim.method = "gen", 
                  control = list(trace = FALSE), config = list(on.par.without.desc = "warn"))

lrn_noise = makeLearner("regr.GPfit", predict.type = "response")

noise_data = cbind(noise_design, noise)
noise_estim = makeRegrTask(data = noise_data, target = "noise")
noise_model = train(lrn_noise, noise_estim)
features = noise_model$features

estimated_noise = function(x){
  x = matrix(x, nrow = 1)
  newdata = as.data.frame(x)
  names(newdata) = features
  predict(noise_model, newdata = newdata) %>% getPredictionResponse()
}



## start BO
budget = 4
init_design_size = 100

# same design for all approaches
design <- generateDesign(n = init_design_size, par.set = parameter_set, fun = lhs::randomLHS)
ctrl <- makeMBOControl(final.method = "best.true.y", final.evals = 5)


# set Control Argument of BO 
ctrl = makeMBOControl(propose.points = 1L,
                      store.model.at = 1:(budget+1))

ctrl = setMBOControlTermination(ctrl, iters = budget)
infill_crit = makeMBOInfillCritRACB(cb.lambda = 2, 
                                    cb.alpha = 1,
                                    noise_proxy_fun = estimated_noise)

#infill_crit = makeMBOInfillCritCB(cb.lambda = 1)

ctrl = setMBOControlInfill(ctrl, crit = infill_crit, opt = "focussearch", 
                           opt.focussearch.points = 1000, opt.focussearch.maxit = 1)

lrn = makeLearner("regr.km", covtype = "powexp", predict.type = "se", optim.method = "gen", 
                  control = list(trace = FALSE), config = list(on.par.without.desc = "warn"))
# ensure numerical stability in km {DiceKriging} cf. github issue and recommendation by Bernd Bischl 
y = apply(design, 1, obj_fun)
Nuggets = 1e-8*var(y)
lrn = setHyperPars(learner = lrn, nugget=Nuggets)

res_mbo = mbo(fun = obj_fun, design = design, control = ctrl, learner = lrn)

shapleys = ShapleyMBO_racb(res.mbo = res_mbo, contribution = TRUE, noise_proxy_fun = var_function)

#shapleys = ShapleyMBO(res.mbo = res_mbo, iter.interest = 1:3, contribution = TRUE)

shapleys <- select(shapleys, "iter","feature", "phi_mean_scaled", "phi_se_scaled", "phi_noise_scaled","phi_cb")




x1_ind = subset(1:nrow(shapleys), 1:nrow(shapleys) %% 2 == 1)
x2_ind = subset(1:nrow(shapleys), 1:nrow(shapleys) %% 2 == 0)

shapleys$phi_mean_scaled[x1_ind] %>% mean
shapleys$phi_mean_scaled[x2_ind] %>% mean

shapleys$phi_se_scaled[x1_ind] %>% mean
shapleys$phi_se_scaled[x2_ind] %>% mean

shapleys$phi_noise_scaled[x1_ind] %>% mean
shapleys$phi_noise_scaled[x2_ind] %>% mean

shapleys
