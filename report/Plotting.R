###### Script used to produce the plot of model performances ######

library(tidyverse)

# N hidden layers
mods <- c(1, 2, 3, 4, 5)
# Accuracies
accs <- c(53.33, 52.35, 52.81, 53.55, 53.52)
# Macro-F1 scores
f1s <-  c(38.32, 32.93, 43.59, 38.75, 42.00)
# Precisions
prec <- c(39.04, 36.84, 41.27, 42.22, 46.80)
# Recalls
rec <-  c(42.11, 31.58, 49.44, 44.11, 43.27)
# Running times
rtim <- c("00:22:27", "00:25:43", "00:27:04", "00:28:40", "00:25:41")

# Create a dataframe
m <- tibble(HiddenLayers=mods, 
            Accuracy=accs,
            MacroF1=f1s,
            Precision=prec,
            Recall=rec,
            RunTime=hms::as.hms(rtim)) %>%
  mutate(RunTime=as.numeric(lubridate::hms(RunTime)) / 60) 

# Plot 1: Runtime vs no of hidden layers 
p1 <- m %>% 
  ggplot(aes(x=HiddenLayers, y=RunTime)) + 
  geom_line() + 
  geom_point() +
  labs(x="Number of Hidden Layers", y="Minutes") +
  theme(plot.title=element_text(size=12)) + 
  ggtitle("Running Time on SAGA")

m %>% 
  gather(key="Metric", value="Value", -HiddenLayers) %>%
  ggplot(aes(x = HiddenLayers, y = Value) ) + 
  geom_line(aes(color = Metric)) + 
  geom_point(aes(color = Metric)) + 
  facet_wrap(~Metric, scales = "free_y", ncol=1, 
             strip.position = "top", 
             labeller = as_labeller(c(HiddenLayers="Number of Hidden Layers", 
                                      Accuracy="Accuracy, %",
                                      MacroF1="Macro-F1, %",
                                      Precision="Precision, %",
                                      Recall="Recall, %",
                                      RunTime="Minutes Running Time")))  +
  ylab(NULL) +
  xlab("Number of Hidden Layers") +
  theme(legend.position = "none")

