rm(list=ls())
library(lmerTest)
library(sjPlot)

bmm <- loadRData("./models/model_0109.Rda")$formula
formula <- as.formula(r2 ~ whichlag + inc_class + sif_coefficient + total_harvarea + avg_farmsize + cropland_fraction + avg_csif + flag_pct)

bycrop <- loadRData("./data/df_model_update.Rda")

results <- bycrop %>%
  group_by(cropname) %>%
  do(model = lm(formula, data = .)) %>%
  mutate(adj_r_squared = summary(model)$adj.r.squared) %>% select(cropname, adj_r_squared)

compute_adj_r2 <- function(df) {
  model = lm(formula, data = df)
  return(summary(model)$adj.r.squared*100)
}

# Perform bootstrap
set.seed(928) # for reproducibility
n_bootstrap <- 1000 # number of bootstrap samples
bootstrap_results <- bycrop %>%
  group_by(cropname) %>%
  do(tibble(bootstrap_r2 = replicate(n_bootstrap, compute_adj_r2(sample_n(., n(), replace = TRUE)))))

save(bootstrap_results, file="./data/bootstrap_results.Rda")


long_format <- bootstrap_results %>%
  unnest(bootstrap_r2) %>%
  rename(adjusted_r2 = bootstrap_r2) %>% group_by(cropname) %>% 
  mutate(mean = mean(adjusted_r2)) %>% 
  ungroup() 

long_format <- long_format[order(long_format$mean, decreasing= TRUE),]
long_format$cropname <- factor(long_format$cropname, levels= unique(long_format$cropname))

unique(long_format[, c("cropname", "mean")])

?geom_vline
ggplot(long_format, aes(x = adjusted_r2, fill = cropname)) +
  geom_density(alpha = 0.5) + geom_vline(aes(xintercept=mean)) + 
  labs(title =   "Bootstrapped percentage of \n variance explained by model",
       x = "Percentage of variance explained (%)",
       y = "Density") +
  facet_wrap(~cropname, ncol=1,
             strip.position = "left")  + 
  geom_vline(xintercept=50, linetype="dashed",color="gray") +
  theme_blank() + 
  theme(
    strip.text.y.left = element_text(angle = 0, hjust=c(1,1)),
    axis.text.y = element_blank(),
    axis.title.y = element_blank(),
    legend.position="none",
    text = element_text(size=18))+
  scale_y_discrete(
    expand = c(0,0)) + theme(
      panel.spacing = unit(0,'lines'))

ggsave("./plots/boostraped_adj_r2.png", height=12, width=7.5)
