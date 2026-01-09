rm(list = ls())
library(ggVennDiagram)
library(tidyverse)
Sys.setenv(LANGUAGE='en')

d_level <- read_csv('result/ASIA_2Classes_feature.csv') %>% .[-1]
c_level <- read_csv('result/ASIA_C_feature.csv') %>% .[-1]
b_level <- read_csv('result/ASIA_B_feature.csv') %>% .[-1]
a_level <- read_csv('data/Em_A_lasso_result.csv') 
terminal <- read_csv('result/SA_bool.csv')
lasso <- read_csv('data/lasso_result.csv')

original_data <- read_csv('data/SCI_data_original.csv') %>% .[-1]
colnames(original_data) <- gsub('-','_',colnames(original_data))
terminal_name <- colnames(original_data)[terminal$index]

venn_list <- list(`ASIA-D` = colnames(d_level),
                  `ASIA-C` = colnames(c_level),
                  `ASIA-B` = colnames(b_level),
                  `ASIA-A` = a_level$x,
                  `ASIA-T` = terminal_name,
                  `LASSO` = lasso$x)
name_list <- c('Em-2Class','Em-C','Em-B','Em-A','Em-3Class','LASSO')

ggVennDiagram(venn_list,
              category.names = name_list,
              label = "count", 
              label_color = "black",
              label_alpha = 0,
              edge_lty = "solid", 
              edge_size = 0.5) +
  scale_fill_gradient(low="white",high = "#b9292b",name = "Feature Numbers")+
  scale_color_manual(values = c('grey','grey','grey','grey','grey','grey'))
ggsave('figure/Features_Venn.pdf',width = 10,height = 10)

