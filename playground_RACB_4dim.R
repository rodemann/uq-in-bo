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


dim = 4
fun = smoof::makeAlpine02Function(dim)
#fun = smoof::makeAlpine01Function(dim)
fun = smoof::makeHyperEllipsoidFunction(dim)
# autoplot(fun)
# plot3D(fun)

var_function = function(x) 12*abs(x[1]-15) + 2*abs(x[2]-15)

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


#plot3D(obj_fun, length.out = 100)
#autoplot(obj_fun)

budget = 40
init_design_size = 40
parameter_set = getParamSet(obj_fun)

# same design for all approaches
design <- generateDesign(n = init_design_size, par.set = parameter_set, fun = lhs::randomLHS)
ctrl <- makeMBOControl(final.method = "best.true.y", final.evals = 5)


# set Control Argument of BO 
ctrl = makeMBOControl(propose.points = 1L,
                      store.model.at = 1:(budget+1))

ctrl = setMBOControlTermination(ctrl, iters = budget)
infill_crit = makeMBOInfillCritRACB(cb.lambda = 2, 
                                    cb.alpha = 1,
                                    noise_proxy_fun = var_function)

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




x1_ind = subset(1:nrow(shapleys), 1:nrow(shapleys) %% dim == 1)
x2_ind = subset(1:nrow(shapleys), 1:nrow(shapleys) %% dim == 2)
x3_ind = subset(1:nrow(shapleys), 1:nrow(shapleys) %% dim == 3)
x4_ind = subset(1:nrow(shapleys), 1:nrow(shapleys) %% dim == 0)

shapleys$phi_mean_scaled[x1_ind] %>% mean
shapleys$phi_mean_scaled[x2_ind] %>% mean
shapleys$phi_mean_scaled[x3_ind] %>% mean
shapleys$phi_mean_scaled[x4_ind] %>% mean

shapleys$phi_se_scaled[x1_ind] %>% mean
shapleys$phi_se_scaled[x2_ind] %>% mean
shapleys$phi_se_scaled[x3_ind] %>% mean
shapleys$phi_se_scaled[x4_ind] %>% mean

shapleys$phi_noise_scaled[x1_ind] %>% mean
shapleys$phi_noise_scaled[x2_ind] %>% mean
shapleys$phi_noise_scaled[x3_ind] %>% mean
shapleys$phi_noise_scaled[x4_ind] %>% mean

shapleys
