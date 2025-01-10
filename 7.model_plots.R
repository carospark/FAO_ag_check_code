rm(list=ls())
library(loo)
library(lmerTest)
library(sjPlot)
model <- loadRData("./models/model_0109.Rda")

pp_check(model)
ggsave("./plots/post_predictive_check_update.png", height=4, width=3)

plots <- invisible(plot(model, ask= FALSE))

# Save all plots automatically
for (i in seq_along(plots[1:3])) {
  ggsave(filename = paste0("./plots/brm_plot_", i, ".png"), plot = plots[[i]], width = 5, height = 6, dpi = 300)
}

###############################################################################################
df <- loadRData("./data/df_model_update.Rda") %>% 
  select(-c(r2_bc, resid_var_bc, iso_a3, sif_coefficient, majcrop1, majcrop2,  
            total_gridcells, avg_gridcells))  
y <- c("r2", "resid_var")
rand <- c("country", "cropname")
x_char <- setdiff(names(df)[sapply(df, is.character)], rand)
x_num <- setdiff(colnames(df), c(y, names(df)[sapply(df, is.character)]))
df[,x_num] <- sapply(df[,x_num], scale)
covariates <- paste(c(x_char, x_num), collapse= " + ")
df <- filter(df, r2>0 & r2<1)
full<-as.formula(paste("r2 ~ ", covariates, sep=""))
f4 <- update(full, ~ . - lc - max_yield - avg_yield - corrupt_change - corrupt_trend - max_csif - avg_gdp_cap - gdp_cap_trend) 

compute_adj_r2 <- function(df) {
  model = lm(f4, data = df)
  return(summary(model)$adj.r.squared*100)}


set.seed(1216) 
n_bootstrap <- 1000 
bootstrap_results <- df %>%
  group_by(cropname) %>%
  do(tibble(bootstrap_r2 = replicate(n_bootstrap, compute_adj_r2(sample_n(., n(), replace = TRUE)))))

save(bootstrap_results, file="./data/bootstrap_results_update.Rda")

long_format <- bootstrap_results %>%
  unnest(bootstrap_r2) %>%
  rename(adjusted_r2 = bootstrap_r2) %>% group_by(cropname) %>% 
  mutate(mean = mean(adjusted_r2)) %>% 
  ungroup() 

long_format <- long_format[order(long_format$mean, decreasing= TRUE),]
long_format$cropname <- factor(long_format$cropname, levels= unique(long_format$cropname))


ggplot(long_format, aes(x = adjusted_r2, fill = cropname)) +
  geom_density(alpha = 0.5) + geom_vline(aes(xintercept=mean)) + 
  labs(title =   "Bootstrapped percentage of \n variance explained by model",
       x = "Percentage of variance explained (%)",
       y = "Density") +
  facet_wrap(~cropname, ncol=1,
             strip.position = "left")  + 
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

ggsave("./plots/boostraped_adj_r2_updated.png", height=12, width=7.5)


###############################################################################################

library(corrplot)
View(df)

nofact <- df[,x_num] %>% select(-gdp_cap_trend)
numeric_df <- nofact[sapply(nofact, is.numeric)]
cor_matrix <- cor(numeric_df, use = "complete.obs")  
png("./plots/corrplot.png")
corrplot(cor_matrix, method = 'square', order = 'FPC', type = 'lower', diag = FALSE) 
dev.off()

ggsave("./plots/covariate_corrplot_update.png")




beep(5)
