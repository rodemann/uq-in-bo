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

