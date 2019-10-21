checkpoint:::checkpoint(
	snapshotDate = utils::packageDescription("RevoUtils")$MRANDate, 
	R.version = "3.5.3")

library(tidyverse)
library(ggplot2)
# library(reticulate)
library(dplyr)
# library(readr)
library(readxl)
library(lubridate)
library(scales)
library(investr)
library(fdrtool)
library(foreach)
source("src/visualization_functions.r")


## Setting and default value
extrafont::loadfonts(device="win")
current = as.Date("2019-10-21")
version = "0.2.0"
hh = 4.5


if (!dir.exists(file.path("Results/figures"))) {
  dir.create(file.path("Results/figures"))
}


## Loading results data
tam =
    bind_rows(read_excel("Results/data/testing_df.xlsx"),
                read_excel("Results/data/training_df.xlsx")) %>%
    mutate(date = as.Date(date),
           begin = as.Date("1989-04-25"),
           end = as.Date("1989-06-04"),
           event = "1989 Tiananman") %>% 
    select(-...1)
hk14 =
    read_excel("Results/data/predict_df_HK2014.xlsx") %>%
    mutate(date = as.Date(date),
           begin = as.Date("2014-09-26"),
           end = as.Date("2014-12-15"),
           event = "2014 Hong Kong") %>%
    select(-...1)
hk19 =
    read_excel("Results/data/predict_df_HK2019.xlsx") %>%
    mutate(date = as.Date(date),
           begin = as.Date("2019-06-09"),
           end = current,
           event = "2019 Hong Kong") %>%
    select(-...1)


#########################
# figures for sum stats #
#########################

df_sum = rbind(tam, hk14, hk19) %>%
    select(c(event, begin, end, date, id, page)) %>%
    group_by(event, begin, end, date, id, page) %>%
    filter(row_number()==1)

aggregate = df_sum %>%
    group_by(event) %>% summarise(n_articles = n(),
                                  earliest = min(date),
                                  latest = max(date))

timeline = df_sum %>%
    group_by(event, begin, end, date) %>%
    summarise(n_articles = n(),
              n_fronts = sum(page<=1)) %>% ungroup() %>%
    group_by(event, begin, end) %>%
    complete(date = seq.Date(begin[1], end[1], by="day"), fill = list(n_articles=0, n_fronts=0)) %>%
    ungroup() %>%
    mutate(days_since = date-begin) %>%
    select(c(event, days_since, n_articles, n_fronts))

plot1 = ggplot(timeline, aes(x = days_since, y = n_articles, group = event, colour = event)) +
    geom_line(aes(linetype = event)) +
    scale_x_continuous(limits=c(0, 150), breaks = seq(0,150,by=30)) +
    scale_y_continuous(limits=c(0, 12), breaks = seq(0,12,by=4)) +
    xlab("Number of days since beginning") +
    ylab("Number of relevant articles") +
    scale_color_manual(values = c("black", "blue", "red")) +
    scale_linetype_manual(values=c("solid", "dotted", "longdash")) +
    theme_bw() +
    theme(panel.grid.minor=element_blank(), panel.grid.major=element_blank(),
          legend.direction = "horizontal", legend.position="bottom", legend.title=element_blank(),
          legend.spacing.x = unit(0.2, 'in'))

ggsave(file.path("Results/figures","summary_stats_1.png"), plot = plot1, width=(3/2)*hh, height=hh)

plot2 = ggplot(timeline, aes(x = days_since, y = n_fronts, group = event, colour = event)) +
    geom_line(aes(linetype = event)) +
    scale_x_continuous(limits=c(0, 150), breaks = seq(0,150,by=30)) +
    scale_y_continuous(limits=c(0, 12), breaks = seq(0,12,by=4)) +
    xlab("Number of days since beginning") +
    ylab("Number of relevant front-page articles") +
    scale_color_manual(values = c("black", "blue", "red")) +
    scale_linetype_manual(values=c("solid", "dotted", "longdash")) +
    theme_bw() +
    theme(panel.grid.minor=element_blank(), panel.grid.major=element_blank(),
          legend.direction = "horizontal", legend.position="bottom", legend.title=element_blank(),
          legend.spacing.x = unit(0.2, 'in'))

ggsave(file.path("Results/figures","summary_stats_2.png"), plot = plot2, width=(3/2)*hh, height=hh)


########################
# figures for mappings #
########################

## scatter plot of tam before summarization
tam_fitted = ggplot(data = tam, aes(x = predict, y = days_since)) + 
    geom_point(shape = 1, color = 'black') +
    scale_x_continuous(limits=c(0, 43), breaks = seq(0,40,by=10)) +
    scale_y_continuous(limits=c(0, 43), breaks = seq(0,40,by=10)) +
    xlab("Fitted time of publication (days since beginning)") +
    ylab("Time of publication (days since beginning)") +
    theme_bw()+
    theme(panel.grid.minor=element_blank(), panel.grid.major=element_blank()) + 
    geom_abline(intercept=0, slope=1, linetype = "dashed", color="black", size = 1)

ggsave(file.path("Results/figures","tiananmen_fitted.png"), plot = tam_fitted, width=(3/2)*hh, height=(3/2)*hh)


## Detect the search range of hyper-parameters
sentences_search_range = tam %>%
    group_by(days_since,id) %>% summarize(n_sentence = n()) %>% ungroup() %>%
    summarize(min=min(n_sentence), max=max(n_sentence)) %>%
    (function(x) {seq(x$min,x$max)})
	
articles_search_range = tam %>%
    distinct(days_since,id) %>%
    group_by(days_since) %>% summarize(n_article = n()) %>%
    summarize(min=min(n_article), max=max(n_article)) %>%
    (function(x) {seq(x$min,x$max)})

## Search for hyper-parameters
all_MSE = 
    foreach(i = sentences_search_range, .combine = bind_rows, .multicombine = TRUE) %:% foreach(j=articles_search_range, .combine = bind_rows, .multicombine = TRUE) %do% {
        tibble(max_sentences=i,max_articles=j, mse=model_MSE(tam,i,j))
    } %>% arrange(mse)

# all_MSE %>% arrange(mse) %>% head(10) %>% View
# ggplot(data=all_MSE, aes(x=max_sentences, y=max_articles, fill=mse) ) + geom_tile()
selected_model = all_MSE %>% arrange(mse) %>% head(1)


plot_tam = figure_tam(tam,
                      max_sentences=selected_model$max_sentences,
                      max_articles=selected_model$max_articles)
ggsave(file.path("Results/figures","fig_tiananmen.png"), plot = plot_tam, width=(3/2)*hh, height=(3/2)*hh)


plot_hk19 = figure_hk(tam, hk19,
                      max_sentences=selected_model$max_sentences,
                      max_articles=selected_model$max_articles,
                      tail_len=Inf, color_choice = "red",
                      as.Date("2019-06-09"), current, with_events=FALSE)
ggsave(file.path("Results/figures","fig_hk19.png"), plot = plot_hk19, width=(3/2)*hh, height=hh)


plot_hk14 = figure_hk(tam, hk14,
                      max_sentences=selected_model$max_sentences,
                      max_articles=selected_model$max_articles,
                      tail_len=Inf, color_choice = "blue",
                      as.Date("2014-09-26"), as.Date("2014-12-15"), with_events=FALSE)
ggsave(file.path("Results/figures","fig_hk14.png"), plot = plot_hk14, width=(3/2)*hh, height=hh)


plot_hk19_w_events = figure_hk(tam, hk19,
                             max_sentences=selected_model$max_sentences,
                             max_articles=selected_model$max_articles,
                             tail_len=Inf, color_choice = "red",
                             as.Date("2019-06-09"), current, with_events=TRUE)
ggsave(file.path("Results/figures","fig_hk19_w_events.png"), plot = plot_hk19_w_events, width=(3/2)*hh, height=hh)



tmp_tam = tam %>% proc_data(max_sentences=selected_model$max_sentences,max_articles=selected_model$max_articles) 
tmp_hk19 = hk19 %>% proc_data(max_sentences=selected_model$max_sentences,max_articles=selected_model$max_articles) 
tmp_hk14 = hk14 %>% proc_data(max_sentences=selected_model$max_sentences,max_articles=selected_model$max_articles) 

loess_model = loess(formula = days_since~predict, data=tmp_tam)

predict_tiananmen_date(loess_model, tmp_hk19) %>% 
	select(date_actual, date_tiananmen) %>% 
	write_csv(path= paste0("results/data/PCI-Crackdown-HK2019_v",version,"_",as.character(current),".csv"))

predict_tiananmen_date(loess_model, tmp_hk14) %>% 
	select(date_actual, date_tiananmen) %>% 
	write_csv(path= paste0("results/data/PCI-Crackdown-HK2014_v",version,"_",as.character(current),".csv"))


