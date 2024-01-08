# this script runs the BO for an agent who has access to shapleyBO
#


number_interventions = 0
## BO loop
for (b in 1:budget) {
  ctrl = setMBOControlTermination(ctrl, iters = 1)
  ctrl_agent = makeMBOControl(propose.points = 1L, store.model.at = 1:2)
  res_mbo = mbo(fun = obj_fun, design = design, control = ctrl, learner = lrn, show.info = F)
  mbo_design = res_mbo$opt.path %>% as.data.frame()
  shapleys = ShapleyMBO(res.mbo = res_mbo, iter.interest = 1, contribution = TRUE)
  if(shapleys$phi_mean_scaled[1] == 0 & shapleys$phi_mean_scaled[2] == 0){
    shapley_ratio_bo = 1
  }else
  shapley_ratio_bo = shapleys$phi_mean_scaled[1] / shapleys$phi_mean_scaled[2]
  # human bo interface: does the proposal align with agent's knowledge?
  if(shapley_ratio_agent/shapley_ratio_bo > 2 |
     shapley_ratio_agent/shapley_ratio_bo < 0.5 ){
    number_interventions = number_interventions +1
    ctrl_agent = setMBOControlTermination(ctrl_agent, iters = 1)
    res_mbo_agent = mbo(fun = obj_fun, design = design_agent, control = ctrl_agent, learner = lrn_agent, show.info = F)
    mbo_design_agent = res_mbo_agent$opt.path %>% as.data.frame()
    proposal_agent = mbo_design_agent[nrow(mbo_design_agent),1:2]
    design = rbind(design, proposal_agent)
    # possible extension: update shapleys
  }else
  design = mbo_design[,1:2]
}
# optimum
res_mbo$y

number_interventions

# 
# #raw_shapleys = ShapleyMBO_racb(res.mbo = res_mbo, contribution = TRUE, noise_proxy_fun = var_function)
# shapleys = ShapleyMBO(res.mbo = res_mbo, iter.interest = 1, contribution = TRUE)
# 
# shapleys_sel <- select(shapleys, "iter","feature", "phi_mean_scaled", "phi_se_scaled", "phi_cb")
# 
# shapleys
# 
# shapleys_sel
# 
# x1_ind = subset(1:nrow(shapleys_sel), 1:nrow(shapleys) %% 2 == 1)
# x2_ind = subset(1:nrow(shapleys), 1:nrow(shapleys) %% 2 == 0)
# mean_shapley_x1 = shapleys$phi_mean_scaled[x1_ind] %>% mean
# mean_shapley_x2 = shapleys$phi_mean_scaled[x2_ind] %>% mean
# 
# 
# 
# shapleys$phi_mean_scaled[x3_ind] %>% mean
# shapleys$phi_mean_scaled[x4_ind] %>% mean
# 
# shapleys$phi_se_scaled[x1_ind] %>% mean
# shapleys$phi_se_scaled[x2_ind] %>% mean
# shapleys$phi_se_scaled[x3_ind] %>% mean
# shapleys$phi_se_scaled[x4_ind] %>% mean
# 
# shapleys$phi_noise_scaled[x1_ind] %>% mean
# shapleys$phi_noise_scaled[x2_ind] %>% mean
# shapleys$phi_noise_scaled[x3_ind] %>% mean
# shapleys$phi_noise_scaled[x4_ind] %>% mean
