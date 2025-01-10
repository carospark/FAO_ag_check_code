rm(list=ls())

df <- unique(loadRData("./data/df_model_update.Rda") %>% select(country, cropname, r2, inc_class) )
df$r2 <- sqrt(df$r2)
path <- "./data/combined_update.feather"
combined <- arrow::read_feather(path) %>% select("country", "cropname", "yield")
df <- left_join(df, combined) %>% filter(!is.na(yield)) %>% mutate(yield = yield/1000)
df$inc_class <- factor(df$inc_class, levels= c("High income", "Upper middle income", "Lower middle income", "Low income"))

cbPalette <- c( "gold",  "#0072B2", "#CC79A7", "lightgreen")

ggplot(data= df, aes(x=r2, y=yield)) + geom_point(aes(color=inc_class), alpha=0.6) +   facet_wrap(~cropname, scales="free") +theme_classic() + theme(legend.title=element_blank(), strip.background = element_rect(fill="lightgray", color=NA)) + scale_color_manual(values=cbPalette) + xlab("FAO-CSIF correlation") + ylab("FAO-reported yields (t/ha)")
ggsave("./plots/var_fao_raw_update.png", height=6.5, width=10)


# Function to calculate Coefficient of Variation (CV)
calc_cv <- function(data) {
  mean_value <- mean(data, na.rm = TRUE)
  sd_value <- sd(data, na.rm = TRUE)
  cv <- (sd_value / mean_value) * 100  
  return(cv)
}

dfavg <- df %>% group_by(country, cropname, r2, inc_class) %>% summarise(cv = calc_cv(yield))
ggplot(data= dfavg, aes(x=r2, y=cv)) + geom_point(aes(color=inc_class)) + facet_wrap(~cropname, scales="free")  + theme_classic()+ theme(legend.title=element_blank(), strip.background = element_rect(fill="lightgray", color=NA)) + scale_color_manual(values=cbPalette) + xlab("FAO-CSIF correlation") + ylab("Coefficient of Variation (CV) of FAO-reported yields")+  geom_smooth(method = "glm", formula = y~x, method.args = list(family = gaussian(link = 'log')), color="gray50", fill="gray")
ggsave("./plots/var_fao_avg_update.png", height=6.5, width=10)

