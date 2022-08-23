# make Infill Criterion (Acquistion Function) Generalized Lower Confidence Bound
library(mlrMBO)
library(BBmisc)

makeMBOInfillCritRACB = function(cb.lambda = NULL, 
                                 cb.alpha = NULL,
                                 noise_proxy_fun = NULL) {
  
  # input checking
  assertNumber(cb.lambda, lower = 0, null.ok = TRUE)
  force(cb.lambda)
  assertNumber(cb.alpha, lower = 0, null.ok = TRUE)
  force(cb.alpha)
  assertFunction(noise_proxy_fun, null.ok = TRUE)
  force(noise_proxy_fun)
  
  
  # create function
  makeMBOInfillCrit(
    fun = function(points, models, control, par.set, designs, iter, progress, attributes = FALSE) {
      
      model <- models[[1L]]

      maximize.mult = if (control$minimize) 1 else -1
      p = predict(model, newdata = points)$data
      
      noise_proxy_fun_apply <- function(newdata) {
        apply(newdata, 1, noise_proxy_fun)
      }
      
      noise_proxy <- noise_proxy_fun_apply(points)
      
      # actual AF
      res = maximize.mult * p$response - cb.lambda * p$se + cb.alpha * noise_proxy
      
      if (attributes) {
        res = setAttribute(res, "crit.components",
                           data.frame(se = p$se, mean = p$response, 
                                      lambda = cb.lambda, 
                                      alpha = cb.alpha,
                                      noise = noise_proxy))
        
      }
      return(res)
    },
    name = "Risk Averse Confidence bound",
    id = "racb",
    components = c("se", "mean", "lambda", "alpha", "noise"),
    params = list(cb.lambda = cb.lambda, cb.alpha = cb.alpha, noise_proxy_fun = noise_proxy_fun),
    opt.direction = "objective",
    requires.se = TRUE
  )
}




