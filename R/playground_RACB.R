library(mlrMBO)
library(ggplot2)
library(dplyr)
library(smoof)
library(iml)

source("R/makeMBOInfillCritRACB.R")
source("R/initCrit.InfillCritRACB.R")
source("R/ShapleyMBO.R")
source("R/_Explore_Exploit_Measures/xplxpl-jr.R")

fun = smoof::makeAlpine02Function(2)
#fun = smoof::makeAlpine01Function(2)

var_function = function(x) 0.2*abs(x[1]-10)
obj_fun = function(x) {
  rnorm(1, mean = fun(x), sd = var_function(x) %>% sqrt())
}
obj_fun = makeSingleObjectiveFunction(name = "noisy parable", 
                                      fn = obj_fun, has.simple.signature = TRUE,
                                      par.set = makeNumericParamSet(
                                            len = 2, id = "x", 
                                            lower = rep(0, 2), upper = rep(10, 2),
                                            vector = TRUE)
                                      )


budget = 3
init_design_size = 10
parameter_set = getParamSet(obj_fun)

# same design for all approaches
design <- generateDesign(n = init_design_size, par.set = parameter_set, fun = lhs::randomLHS)

ctrl <- makeMBOControl(final.method = "best.true.y", final.evals = 5)

# set Control Argument of BO 
ctrl = makeMBOControl(propose.points = 1L,
                      store.model.at = 1:(budget+1))

ctrl = setMBOControlTermination(ctrl, iters = budget)
infill_crit = makeMBOInfillCritRACB(cb.lambda = 5, 
                                    cb.alpha = 2,
                                    noise_proxy_fun = var_function)

ctrl = setMBOControlInfill(ctrl, crit = infill_crit, opt = "focussearchSavepts", 
                           opt.focussearch.points = 200, opt.focussearch.maxit = 1)

lrn = makeLearner("regr.km", covtype = "powexp", predict.type = "se", optim.method = "gen", 
                  control = list(trace = FALSE), config = list(on.par.without.desc = "warn"))
# ensure numerical stability in km {DiceKriging} cf. github issue and recommendation by Bernd Bischl 
y = apply(design, 1, obj_fun)
Nuggets = 1e-8*var(y)
lrn = setHyperPars(learner = lrn, nugget=Nuggets)

res_mbo = mbo(fun = obj_fun, design = design, control = ctrl, learner = lrn)

shapleys = ShapleyMBO(res.mbo = res_mbo, iter.interest = 1:3, contribution = TRUE, noise_proxy_fun = var_function)

select(shapleys, "iter","feature", "phi_mean", "phi_se", "phi_noise", "phi_cb")


