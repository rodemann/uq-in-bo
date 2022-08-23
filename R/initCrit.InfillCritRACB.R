source("R/makeMBOInfillCritRACB.R")

# this extends the method initCrit to the GLCB Infill Criterion 

initCrit.InfillCritRACB = function(crit, fun, design, learner, control) {
  
  cb.lambda = crit$params$cb.lambda
  if (is.null(cb.lambda))
    cb.lambda = ifelse(mlrMBO:::isSimpleNumeric(getParamSet(fun)), 1, 2)
  cb.alpha = crit$params$cb.alpha
  if (is.null(cb.alpha))
    cb.alpha = 1
  noise_proxy_fun = crit$params$noise_proxy_fun
  
  crit = makeMBOInfillCritRACB(cb.lambda, cb.alpha, noise_proxy_fun)
  mlrMBO:::initCritOptDirection(crit, fun)
}