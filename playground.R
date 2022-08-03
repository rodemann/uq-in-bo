library(mlrMBO)
library(ggplot2)
library(dplyr)
library(smoof)
source("R/makeMBOInfillCritUACB.R")
source("R/initCrit.InfillCritUACB.R")
source("R/ShapleyMBO.R")

#fun = smoof::makeAlpine02Function(1)
fun = smoof::makeAlpine01Function(2)

var_function = function(x) 0.2*abs(x-10)
obj_fun = function(x) {
  rnorm(1, mean = fun(x), sd = var_function(x) %>% sqrt())
}
obj_fun = makeSingleObjectiveFunction(name = "noisy_parable", 
                                      fn = obj_fun, has.simple.signature = TRUE,
                                      par.set = makeNumericParamSet("x", 1, -10, 10), noisy = TRUE)

percent.noise = 0.05
# approximate the sd of the function
noise.free.fun = smoof::makeHyperEllipsoidFunction(dimensions = 4L)
sample = ParamHelpers::generateDesign(n = 10000, par.set = getParamSet(noise.free.fun), fun = lhs::randomLHS)
values = apply(sample, 1, noise.free.fun)
sd = sd(values)
# sd of the noise is 5% of the estimated sd of the function (moderate noise)
noise.sd = percent.noise * sd
# define noisy Objective Function
obj_fun = smoof::makeSingleObjectiveFunction(
  name = "noisy 4d Hyper-Ellipsoid",
  id = "hyper_ellipsoid_4d_5%noise",
  description = "4d Hyper-Ellipsoid with artificially added gaussian noise.
    The sd of the noise is 5% of sd of the noise free HypEll, eps ~ N(0, 0.05 * sd(nf.fun))",
  fn = function(x, sd = noise.sd) { #see makeHyperEllipsoidFunction() of the smoof package
    n = length(x)
    eps = rnorm(1, 0, sd) # Gaussian noise
    sum(1:n * x^2) + eps # the fist term is taken from the source code of makeHyperEllipsoidFunction()
  }, 
  par.set = makeNumericParamSet(
    len = 4, id = "x", # 4 dimensional 
    lower = rep(-5.12, 4), upper = rep(5.12, 4),
    vector = TRUE),
  global.opt.params = rep(0, 4),
  global.opt.value = 0,
  noisy = TRUE
)

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

ShapleyMBO(res.mbo = res_mbo, iter.interest = 1)


res_mbo

