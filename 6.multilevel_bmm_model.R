rm(list=ls())
library(loo)
library(lmerTest)
library(sjPlot)
library(glmmTMB)

########################################################### scale covariates
df <- loadRData("./data/df_model_update.Rda") %>% 
  select(-c(r2_bc, resid_var_bc, iso_a3, sif_coefficient, majcrop1, majcrop2,  
            total_gridcells, avg_gridcells))  

y <- c("r2", "resid_var")
rand <- c("country", "cropname")
x_char <- setdiff(names(df)[sapply(df, is.character)], rand)
x_num <- setdiff(colnames(df), c(y, names(df)[sapply(df, is.character)]))
df[,x_num] <- sapply(df[,x_num], scale)
# df$avg_farmsize <- df$avg_farmsize*100

covariates <- paste(c(x_char, x_num), collapse= " + ")

df <- filter(df, r2>0 & r2<1)
full<-as.formula(paste("r2 ~ ", covariates, "+ (1 | cropname) + (1|country)", sep=""))

########################################################### backward selection & anova
full_model <- glmmTMB(full, data = df, glmmTMB::beta_family())
tab_model(full_model)

f1 <- update(full,    ~ . -lc)
m1 <- glmmTMB(f1, data = df, glmmTMB::beta_family())
anova(full_model, m1)

f2 <- update(full, ~ .  - lc - max_yield - avg_yield)
m2 <- glmmTMB(f2, data=df, glmmTMB::beta_family())
anova(m1, m2)

f3 <- update(full, ~ . - lc - max_yield - avg_yield - corrupt_change - corrupt_trend)
m3 <- glmmTMB(f3, data=df, glmmTMB::beta_family())
anova(m2, m3)

f4 <- update(full, ~ . - lc - max_yield - avg_yield - corrupt_change - corrupt_trend - max_csif - avg_gdp_cap - gdp_cap_trend) 
m4 <- glmmTMB(f4, data=df, glmmTMB::beta_family())
anova(m3, m4)

f5 <- update(full, ~ . - lc - max_yield - avg_yield - corrupt_change - corrupt_trend - corrupt_score - max_csif - avg_gdp_cap - gdp_cap_trend) 
m5 <- glmmTMB(f5, data=df, glmmTMB::beta_family())
anova(m4, m5)

########################################################### brms
f4 <- update(full, ~ . - lc - max_yield - avg_yield - corrupt_change - corrupt_trend - max_csif - avg_gdp_cap - gdp_cap_trend) 
model <- brm(f4, data=df, family=zero_inflated_beta(), seed= 121624)
save(model, file= "./models/model_0109.Rda")

model <- loadRData("./models/model_0109.Rda")
tab_model(model)

pl <- c(
  `(Intercept)` = "Intercept",
  avg_csif = "Avg. CSIF proxy",
  total_harvarea = "Total harvested area",
  avg_farmsize = "Avg. harvested area",
  cropland_fraction = "Cropland fraction",
  whichlagyield_lag = "Lag offset: lag",
  whichlagyield_lead = "Lag offset: lead",
  inc_classLowincome= "Class: Low income",
  inc_classLowermiddleincome = "Class: Lower middle income",
  inc_classUppermiddleincome ="Class: Upper middle income",
  corrupt_score = "Corruption",
  flag_pct = "Flagged percentage"
)

as.data.frame(1:12, rownames(summary(model)$fixed))
tab_model(model,
          pred.labels=pl, order.terms = c(1, 10, 7:9, 2:3, 4:6, 11,12), 
          rm.terms="phi", dv.labels = "Correspondence Index",
          title = "Global all-crop mixed model", file="./models/model_0109.html")


########################################################### sensitivity analysis: flagged
model <- loadRData("./models/model_0109.Rda")
full<-as.formula(paste("r2 ~ ", covariates, "+ (1 | cropname) + (1|country)", sep=""))
f4 <- update(full, ~ . - lc - max_yield - avg_yield - corrupt_change - corrupt_trend - max_csif - avg_gdp_cap - gdp_cap_trend) 

flag_og <- loadRData("./data/df_model_update.Rda") %>% 
  select(country, cropname, flag_pct)
df2 <- left_join(df %>% select(-"flag_pct"), flag_og, by=c("country", "cropname"))

model_50flags <- brm(update(f4, ~ . - flag_pct), data=filter(df2, flag_pct<=0.5), 
                     family=zero_inflated_beta(), seed= 121624)
save(model_50flags, file="./models/model50_0109.Rda")

model_0flags <- brm(update(f4, ~ . - flag_pct), data=filter(df2, flag_pct==0), 
                     family=zero_inflated_beta(), seed= 121624)
save(model_0flags, file="./models/model0_0109.Rda")


sjPlot::tab_model(model, model_50flags, model_0flags, 
                  pred.labels=pl, order.terms = c(1, 10, 7:9, 2:3, 4:6, 11,12), rm.terms="phi", dv.labels = "Correspondence Index", title = "Global all-crop mixed model: flag sensitivity analysis", file="./models/flag_0109.html")






