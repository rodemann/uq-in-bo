source("R/makeMBOInfillCritUACB.R")

# this extends the method initCrit to the GLCB Infill Criterion 

initCrit.InfillCritUACB = function(crit, fun, design, learner, control) {

  cb.lambda = crit$params$cb.lambda
  if (is.null(cb.lambda))
    cb.lambda = ifelse(mlrMBO:::isSimpleNumeric(getParamSet(fun)), 1, 2)
  cb.rho = crit$params$cb.rho
  if (is.null(cb.rho))
    cb.rho = 1
  imprecision = crit$params$imprecision
  if (is.null(imprecision))
    imprecision = 100
  cb.alpha = crit$params$cb.alpha
  if (is.null(cb.alpha))
    cb.alpha = 1
  base_kernel = learner$par.vals$covtype
  noise_proxy_fun = crit$params$noise_proxy_fun

  crit = makeMBOInfillCritUACB(cb.lambda, cb.rho, cb.alpha, base_kernel,
                               imprecision, noise_proxy_fun)
  mlrMBO:::initCritOptDirection(crit, fun)
}