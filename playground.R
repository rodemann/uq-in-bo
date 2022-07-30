library(mlrMBO)
library(ggplot2)
library(dplyr)
library(smoof)
source("R/makeMBOInfillCritUACB.R")
source("R/initCrit.InfillCritUACB.R")

fun = smoof::makeAlpine02Function(1)

var_function = function(x) 0.2*abs(x-5)
obj_fun = function(x) {
  rnorm(1, mean = fun(x), sd = var_function(x) %>% sqrt())
}
obj_fun = makeSingleObjectiveFunction(name = "noisy_parable", 
                                      fn = obj_fun, has.simple.signature = TRUE,
                                      par.set = makeNumericParamSet("x", 1, 0, 20), noisy = TRUE)
# visualize the function
autoplot(obj_fun, length.out = 1000)


budget = 25
init_design_size = 100
parameter_set = getParamSet(obj_fun)

# same design for all approaches
design <- generateDesign(n = init_design_size, par.set = parameter_set, fun = lhs::randomLHS)

ctrl <- makeMBOControl(final.method = "best.true.y", final.evals = 5)

# set Control Argument of BO 
ctrl = makeMBOControl(propose.points = 1L)
ctrl = setMBOControlTermination(ctrl, iters = budget)
infill_crit = makeMBOInfillCritUACB(cb.lambda = 5, 
                                    cb.rho = 0,
                                    cb.alpha = 10,
                                    base_kernel= "powexp", 
                                    imprecision= 10, 
                                    noise_proxy_fun = var_function)

ctrl = setMBOControlInfill(ctrl, crit = infill_crit, opt = "focussearch", 
                           opt.focussearch.points = 200, opt.focussearch.maxit = 1)

lrn = makeLearner("regr.km", covtype = "powexp", predict.type = "se", optim.method = "gen", 
                  control = list(trace = FALSE), config = list(on.par.without.desc = "warn"))
# ensure numerical stability in km {DiceKriging} cf. github issue and recommendation by Bernd Bischl 
y = apply(design, 1, obj_fun)
Nuggets = 1e-8*var(y)
lrn = setHyperPars(learner = lrn, nugget=Nuggets)

res_mbo = mbo(fun = obj_fun, design = design, control = ctrl, learner = lrn)


res_mbo

