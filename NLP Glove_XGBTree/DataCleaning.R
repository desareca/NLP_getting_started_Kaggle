library(tidyverse)
library(data.table)
library(ggplot2)
library(ggpubr)
library(caret)
library(caretEnsemble)
library(ranger)
library(Metrics)
library(lubridate)
library(tm)
library(syuzhet)
library(tidytext)
library(ggcorrplot)
library(kohonen)
library(RColorBrewer)
library(plotly)
library(text2vec)
library(slam)
library(glmnet)


# ----------------------------------------------------------------------------------------------------
# Carga de datos
# ----------------------------------------------------------------------------------------------------

news <- data.frame(text = readLines("en_US.news.txt", encoding = "UTF-8"),
                   stringsAsFactors = FALSE)
blogs <- data.frame(text = readLines("en_US.blogs.txt", encoding = "UTF-8"),
                    stringsAsFactors = FALSE)
twitter <- data.frame(text = readLines("en_US.twitter.txt", encoding = "UTF-8"),
                      stringsAsFactors = FALSE)

train <- rbind(news, blogs, twitter) # hay alrededor de 3M de observaciones
rm(news, blogs, twitter)

# ----------------------------------------------------------------------------------------------------
# Limpieza y Vectorizacion del texto
# ----------------------------------------------------------------------------------------------------
# prep_fun = tolower
cleanText <- function(text, stopwords = TRUE, language = "english", remText = NULL){
      text = gsub("http "," ",text) # limpia paginas http
      text = gsub("https "," ",text) # limpia paginas https
      text = gsub("com "," ",text) # limpia direcciones
      text = gsub("co "," ",text) # limpia direcciones
      text = gsub("org "," ",text) # limpia direcciones
      text = gsub("[[:punct:]]"," ",text) # elimina puntuacion
      text = gsub("\\w*[0-9]+\\w*\\s*", " ",text) #elimina numeros
      text = stringr::str_replace_all(text, "\\p{quotation mark}", "") # elimina comillas
      text = gsub("\\n", " ",text) # elimina saltos de linea
      text = stringr::str_replace_all(text,"[\\s]+", " ")
      text = stringr::str_replace_all(text," $", "") # elimina espacios finales
      # elimina emojis
      text = gsub("<\\w+>","",iconv(text, from = "UTF-8", to = "latin1", sub = "byte"))
      text = gsub("<\\w+>","",iconv(text, from = "UTF-8", to = "latin1", sub = "byte"))
      text = gsub("-", "",text) # elimina giones
      text = tolower(text) # transforma a minuscula
      text = removeWords(text, letters) # elimina letras sueltas
      text = stringr::str_replace_all(text," $", "") # elimina espacios finales
      if(stopwords){text = removeWords(text, stopwords(language))} # elimina stopwords 
      if(is.null(remText)){text = removeWords(text, remText)} # elimina palabras especificas 
      text = stripWhitespace(text) # quita espacios en blanco repetidos
      text = stringr::str_replace_all(text,"^ ", "") # elimina espacios iniciales
      text = stemDocument(text) # Stem words
      return(text)
}
it_train = itoken(train$text, 
                  preprocessor = cleanText, 
                  tokenizer = word_tokenizer,
                  n_chunks = 5,
                  # ids = train$id, 
                  progressbar = TRUE)

t1 = Sys.time()
vocab = create_vocabulary(it_train, ngram = c(1L, 1L))
print(difftime(Sys.time(), t1, units = 'min')) # toma alrededor de  11 mins

saveRDS(vocab, "vocab.rds")
# hay 391.796 1 grams (sin stopwords).
# filtro donde apareza una palabra por lo menos en 10 de las observaciones totales y 
# 10 de cada observacion



# Calculo dtm, dtm_tfidf y tcm 
vocab <- readRDS("vocab.rds")

vocab_prune = vocab %>% 
      prune_vocabulary(term_count_min = 10, doc_count_min = 10) 
# Con esto redujimos a 50247 terminos.


t1 = Sys.time()
vectorizer = vocab_vectorizer(vocab_prune)
tfidf = TfIdf$new()
dtm = create_dtm(it_train, vectorizer)
dtm_tfidf = fit_transform(dtm, tfidf)
tcm <- create_tcm(it_train, vectorizer)
print(difftime(Sys.time(), t1, units = 'min')) # demoro aprox 24 mins

saveRDS(dtm,"DTM.rds")
saveRDS(dtm_tfidf,"DTMtfidf.rds")
saveRDS(tfidf,"model_tfidf.rds")
# para aplicar tfidf, dtm_test_tfidf = transform(dtm_test_tfidf, tfidf)
saveRDS(tcm,"TCM.rds")
rm(dtm, dtm_tfidf, it_train, vectorizer, vocab_prune, tcm, tfidf, train, vocab, cleanText)

# ----------------------------------------------------------------------------------------------------
# Glove 
# ----------------------------------------------------------------------------------------------------
# calculo glove
tcm <- readRDS("TCM.rds")
t1 = Sys.time()
glove = GlobalVectors$new(rank = 200, x_max = 10, shuffle = TRUE)
word_vectors = glove$fit_transform(tcm, n_iter = 50)
print(difftime(Sys.time(), t1, units = 'min')) # demoro aprox 24 mins
saveRDS(word_vectors,"word_vectors.rds")
rm(tcm, glove, word_vectors)

# berlin = wv_main["man", , drop = FALSE] - 
#    wv_main["father", , drop = FALSE] + 
#    wv_main["mother", , drop = FALSE]
# cos_sim = sim2(x = wv_main, y = berlin, method = "cosine", norm = "l2")
# head(sort(cos_sim[,1], decreasing = TRUE), 10)



#Ajuste de pesos de palabras en documento.
dtm <- readRDS("DTM.rds")
dtm_tfidf <- readRDS("DTMtfidf.rds")
word_vectors <- readRDS("word_vectors.rds")


nword <- slam::row_sums(dtm, na.rm = T)
rm(dtm)
#vamos a filtrar por documentos con mas de 4 palabras
# dim 3336695
dtm_tfidf <- dtm_tfidf[nword>=5,]
# dim 2364639
t1 = Sys.time()
docCod <- matprod_simple_triplet_matrix(dtm_tfidf %>% as.simple_triplet_matrix(),
                                        word_vectors %>% as.simple_triplet_matrix()
                                        )/nword[nword>=5]

print(difftime(Sys.time(), t1, units = 'min')) # demoro aprox 1 mins
saveRDS(docCod,"docCod.rds")


# ----------------------------------------------------------------------------------------------------
# SOM 
# ----------------------------------------------------------------------------------------------------
training <- readRDS("docCod.rds")

set.seed(0)
Id = sample(c(1:dim(training)[1]), round(dim(training)[1]*20/100))
training<- list(text = training[Id,])


start.time <- Sys.time()
dim <- 20
set.seed(0)
somText <- supersom(training$text, rlen = 200, alpha = c(0.9, 0.01), mode = "pbatch", 
                    normalizeDataLayers = TRUE, radius = floor(dim*0.8), keep.data = T,
                    grid = somgrid(xdim =  dim, ydim =  dim, topo = "rectangular",
                                   neighbourhood.fct = "gaussian", toroidal = FALSE))

print(difftime(Sys.time(), start.time, units = 'min')) # demora aprox 78 mins
# saveRDS(somText, "somText.rds")
saveRDS(list(somText = somText, Id = Id), "somText_wdata.rds")


# revision del modelo som
# somText <- readRDS("somText.rds")
somText <- readRDS("somText_wdata.rds")$somText
colors <- colorRampPalette(brewer.pal(n = 10, name = "Paired")[c(10,8,1)])
par(mfrow = c(1,1))
#grafico de entrenamiento, muestra la distancia media del codebook durante el entrenamiento.
plot(somText, type = "changes", shape = "straight")

par(mfrow = c(2,2))
# muestra el codebook resultante
plot(somText, type = "codes", palette.name = colors, codeRendering = "lines", shape = "straight")

# # muestra el conteo de observaciones mapeadas por cada neurona, conteo 0 neurona en gris.
plot(somText, type = "counts", palette.name = colors, heatkey = TRUE, shape = "straight")

# muestra la suma de las distancias de las neuronas vecinas. Es la U-matrix
plot(somText, type = "dist.neighbours", palette.name = colors, shape = "straight")

# # muestra las distancia de las observaciones al codebook final, mientras menor distancia
# # mejor representación de las observaciones.
plot(somText, type = "quality", palette.name = colors, heatkey = TRUE, shape = "straight")

# ----------------------------------------------------------------------------------------------------
# analisis de sentimientos 
# ----------------------------------------------------------------------------------------------------
dtm <- readRDS("DTM.rds")
dtm_tfidf <- readRDS("DTMtfidf.rds")
vocab <- readRDS("vocab.rds")
vocab_prune = vocab %>% 
   prune_vocabulary(term_count_min = 10, doc_count_min = 10) 


sentCalc <- function(name, row.names = TRUE){
   text <- name %>% t() %>% apply(1,function(x) get_sentiment(char_v = x, method = "syuzhet"))
   # cambiamos method nrc por syuzhet
   if(row.names){text <- text %>% as.data.frame(row.names = name)}
   if(!row.names){text <- text %>% as.data.frame(row.names = c(1:length(name)))}
   return(text)
}

# calcula el valor del sentimiento para cada termino (aprox 0.16 mins)
t1 = Sys.time()
em <- vocab_prune[,1] %>% sentCalc()
print(difftime(Sys.time(), t1, units = 'min'))

(em %>% apply(1, function(x){sum(x)>0}) %>% sum())#*100/nrow(em) 
# hay 927 terminos que presentan algun sentimiento
# lo que representa el 1.8% aproximadamente del total (ojo con esto).


# filtrar documentos con por lo menos 5 terminos
nword <- slam::row_sums(dtm, na.rm = T)
dtm <- dtm[nword>=5,]
dtm_tfidf <- dtm_tfidf[nword>=5,]


# calcula el valor del sentimieno para cada observacion
emText <- matrix(ncol = ncol(em), nrow = 0)
# se ejecuta en bloques para aprovechar memoria y disminuir tiempo de ejecucion
# se toman multiplos del numero de columnas que se ajusten a la memoria del pc
t1 = Sys.time()
pb <- txtProgressBar(min = 0, max = 226, style = 3)
seq <- 0
for (k in 1:226) {
   seq <- max(seq) + seq.int(1,10463,1) 
   # emText <- rbind(emText, (as.matrix(dtm[seq,]))%*%as.matrix(em))
   emText <- rbind(emText, (as.matrix(dtm_tfidf[seq,]))%*%as.matrix(em))
   setTxtProgressBar(pb, k)
}
close(pb)
# emText <- rbind(emText, (as.matrix(dtm[c(dim(dtm)[1],dim(dtm)[1]),]))%*%as.matrix(em))
emText <- rbind(emText, (as.matrix(dtm_tfidf[c(dim(dtm_tfidf)[1],dim(dtm_tfidf)[1]),]))%*%as.matrix(em))
emText <- emText[-dim(emText)[1],]
emText <- emText/nword[nword>=5]
print(difftime(Sys.time(), t1, units = 'min'))# aprox. 19 min

saveRDS(list(emText = emText, em = em),
        "emText.rds")


emText <- readRDS("emText.rds")$emText

length(emText)
dim(dtm)



# Ahora vamos a revisar la distribucion de sentimientos
data.frame(V1 = emText[sample(c(1:length(emText)), round(length(emText)*10/100))]) %>% 
   ggplot(aes(V1)) +
   geom_histogram(aes(y = ..density.., fill = ..count..),
                  fill="#6633FF", bins = 100, alpha = 0.75) +
   stat_function(fun = dnorm, colour = "red",
                 args = list(mean = mean(emText),
                             sd = sd(emText))) +
   labs(y = "counts", x = "Sentiments", title = "Histogram sentiments")

data.frame(V1 = emText[sample(c(1:length(emText)), round(length(emText)*1/100))]) %>% 
   ggplot(aes(sample = V1)) +
   stat_qq(col = "#6633FF") + stat_qq_line(col="red") + 
   ggtitle("Gráfico Q-Q Normal")

# la mayoria de los sentimientos son neutros, fuera de estos sentimientos neutros
# tenemos que hay un pequeño sesgo hacia los sentimientos positivos.
data.frame(V1 = emText[emText!=0][sample(c(1:length(emText[emText!=0])), round(length(emText[emText!=0])*10/100))]) %>% 
   ggplot(aes(V1)) +
   geom_histogram(aes(y = ..density.., fill = ..count..),
                  fill="#6633FF", bins = 100, alpha = 0.75) +
   stat_function(fun = dnorm, colour = "red",
                 args = list(mean = mean(emText[emText!=0]),
                             sd = sd(emText[emText!=0]))) +
   labs(y = "counts", x = "Sentiments", title = "Histogram sentiments")


data.frame(V1 = emText[emText!=0][sample(c(1:length(emText[emText!=0])), round(length(emText[emText!=0])*1/100))]) %>% 
   ggplot(aes(sample = V1)) +
   stat_qq(col = "#6633FF") + stat_qq_line(col="red") + 
   ggtitle("Gráfico Q-Q Normal")

summary(emText)
summary(emText[emText!=0])


# ----------------------------------------------------------------------------------------------------
# Clasificador de sentimientos utilizando Glove
# ----------------------------------------------------------------------------------------------------
emText <- readRDS("emText.rds")$emText
# fSent <- function(x){ifelse(x==0,"neutro", ifelse(x>0, "positivo", "negativo"))}
# emText <- emText %>% sapply(function(x) 1*(x>0))
# emText <- emText %>% sapply(fSent)

set.seed(0)
Id = sample(c(1:length(emText)), round(length(emText)*25/100))
NFOLDS = 5


t1 = Sys.time()
glmnet_glove = cv.glmnet(x = readRDS("docCod.rds")[Id,], 
                         # y = emText[Id],
                         y = 1*(emText[Id]>0),
                         family = "binomial",
                         type.measure = "auc",
                         nfolds = NFOLDS,
                         trace.it = 1)
print(difftime(Sys.time(), t1, units = 'sec'))

t1 = Sys.time()
glmnet_dtm = cv.glmnet(x = readRDS("DTM.rds")[Id,], 
                       # y = emText[Id],
                       y = 1*(emText[Id]>0),
                       family = "binomial",
                       type.measure = "auc",
                       nfolds = NFOLDS,
                       trace.it = 1)
print(difftime(Sys.time(), t1, units = 'sec'))


t1 = Sys.time()
glmnet_dtmtfidf = cv.glmnet(x = readRDS("DTMtfidf.rds")[Id,], 
                            # y = emText[Id],
                            y = 1*(emText[Id]>0),
                            family = "binomial",
                            type.measure = "auc",
                            nfolds = NFOLDS,
                            trace.it = 1)
print(difftime(Sys.time(), t1, units = 'sec'))


plot(glmnet_glove)
plot(glmnet_dtm)
plot(glmnet_dtmtfidf)



# Clasificacion con SOM
somText <- readRDS("somText_wdata.rds")
Id <- somText$Id
somText <- somText$somText
emText <- readRDS("emText.rds")$emText
colors <- colorRampPalette(brewer.pal(n = 10, name = "Paired")[c(10,8,1)])


# funcion para clasificar
distClass <- function(model, target){
   dist <- table(target, model$unit.classif)
   dist <- dist %>% apply(2, function(x){x/sum(x)})
   dist <- dist[,order(as.numeric(colnames(dist)))]
   
   dimX <- somText[["grid"]][["xdim"]]
   dimY <- somText[["grid"]][["ydim"]]
   
   
   col0 <- c(1:(dimX*dimY))[-unique(model$unit.classif)]
   if(length(col0)){
      dist <- cbind(dist, matrix(0, nrow = nrow(dist), ncol = length(col0)))
      colnames(dist)[(ncol(dist) - length(col0) + 1):ncol(dist)] <- col0
   }
   dist <- dist[,order(as.numeric(colnames(dist)))]
   return(list(dist = dist, col0 = col0))
}
# Cluster en SOM por sentimiento
dist <- list()
dist[[1]] <- distClass(somText, 1*(emText[Id]<0))
dist[[2]] <- distClass(somText, 1*(emText[Id]==0))
dist[[3]] <- distClass(somText, 1*(emText[Id]>0))

tget <- c("negative", "neutral", "positive")

par(mfrow = c(1,3))
c(1:3) %>% 
   sapply(function(j){plot(somText, type = "property", property = dist[[j]]$dist[2,], main=tget[j],
                           palette.name = colors, heatkey = TRUE, shape = "straight")})





# probar donde caen algunas observaciones
news <- data.frame(text = readLines("en_US.news.txt", encoding = "UTF-8"),
                   stringsAsFactors = FALSE)
blogs <- data.frame(text = readLines("en_US.blogs.txt", encoding = "UTF-8"),
                    stringsAsFactors = FALSE)
twitter <- data.frame(text = readLines("en_US.twitter.txt", encoding = "UTF-8"),
                      stringsAsFactors = FALSE)

train <- rbind(news, blogs, twitter) # hay alrededor de 3M de observaciones
rm(news, blogs, twitter)

nword <- slam::row_sums(readRDS("DTM.rds"), na.rm = T)

train <- train[nword>=5,]
rm(nword)

predText <- predict(somText, 
                    newdata = list(text = readRDS("docCod.rds")[Id,][1:100,] %>% apply(1,as.matrix) %>% t()))

pred <- predText$predictions[[1]]%*%t(predText$unit.predictions[[1]])
pred[is.na(pred)] <- 0

border2 <- dist %>% sapply(function(x, n=4){
   n = n-1
   p = c(1:n)/n
   res = 0
   for (i in 1:n) {res = res + (x$dist[2,]>=p[i])/n}
   return(res)
}) %>% t()

# plot obs aleatoria
obs = sample.int(nrow(pred),1)
par(mfrow = c(1,3))
c(1:3) %>%
   sapply(function(j){
      plot(somText, type = "property",
           # property = border2[j,],
           property = dist[[j]]$dist[2,],
           main=tget[j],
           palette.name = colors, heatkey = TRUE, shape = "straight")
      add.cluster.boundaries(somText, pred[obs,]*(pred[obs,]>=1*max(pred[obs,])),
                             col = "green", lwd = 3)
   })
title(main = paste0("\n", ifelse(length(words(train[obs]))>25,
                                 paste0(train[obs] %>% substr(1,sum(nchar(words(train[obs])[1:25]))), "..."),
                                 train[obs])),
      outer = TRUE, cex.main = 1)


barP <- c(1:3) %>% 
   sapply(function(x){dist[[x]]$dist[2,which.max(pred[obs,])]})
names(barP) <- c("negative", "neutral", "positive")

data.frame(Sentiment = names(barP), Level = barP) %>% 
   ggplot(aes(Level, Sentiment)) + geom_col(fill = colors(1)) + xlim(c(0,1)) + 
   coord_flip() +
   labs(title = paste0(ifelse(length(words(train[obs]))>25,
                              paste0(train[obs] %>% substr(1,sum(nchar(words(train[obs])[1:25]))), "..."),
                              train[obs])))

paste0("(sent: ",round(emText[obs],2),") - ",train[obs])







# ----------------------------------------------------------------------------------------------------
# Visualizacion target texto de prueba en SOM
# ----------------------------------------------------------------------------------------------------
colors <- colorRampPalette(brewer.pal(n = 10, name = "Paired")[c(10,8,1)])
colors <- colorRampPalette(c("#6633FF","darkorange","#88CC00"))

# cargar base de datos 
train <- fread("train.csv") %>% select(text, target)
(table(train$target)/dim(train)[1]) %>% round(4) # relativamente balanceado


cleanText <- function(text, stopwords = TRUE, language = "english", remText = NULL){
   text = gsub("http "," ",text) # limpia paginas http
   text = gsub("https "," ",text) # limpia paginas https
   text = gsub("com "," ",text) # limpia direcciones
   text = gsub("co "," ",text) # limpia direcciones
   text = gsub("org "," ",text) # limpia direcciones
   text = gsub("[[:punct:]]"," ",text) # elimina puntuacion
   text = gsub("\\w*[0-9]+\\w*\\s*", " ",text) #elimina numeros
   text = stringr::str_replace_all(text, "\\p{quotation mark}", "") # elimina comillas
   text = gsub("\\n", " ",text) # elimina saltos de linea
   text = stringr::str_replace_all(text,"[\\s]+", " ")
   text = stringr::str_replace_all(text," $", "") # elimina espacios finales
   # elimina emojis
   text = gsub("<\\w+>","",iconv(text, from = "UTF-8", to = "latin1", sub = "byte"))
   text = gsub("<\\w+>","",iconv(text, from = "UTF-8", to = "latin1", sub = "byte"))
   text = gsub("-", "",text) # elimina giones
   text = tolower(text) # transforma a minuscula
   text = removeWords(text, letters) # elimina letras sueltas
   text = stringr::str_replace_all(text," $", "") # elimina espacios finales
   if(stopwords){text = removeWords(text, stopwords(language))} # elimina stopwords 
   if(is.null(remText)){text = removeWords(text, remText)} # elimina palabras especificas 
   text = stripWhitespace(text) # quita espacios en blanco repetidos
   text = stringr::str_replace_all(text,"^ ", "") # elimina espacios iniciales
   text = stemDocument(text) # Stem words
   return(text)
}

textCod <- function(text, vocab, term_count_min = 10, doc_count_min = 10, cleanF, tfidf, word_vectors){
   # cargar y filtrar vocab
   vocab_prune = vocab %>% 
      prune_vocabulary(term_count_min = 10, doc_count_min = 10) 
   # limpiar y crear dtm con el nuevo texto
   it_train = itoken(train$text, 
                     preprocessor = cleanText, 
                     tokenizer = word_tokenizer,
                     n_chunks = 5,
                     progressbar = FALSE)
   
   vectorizer = vocab_vectorizer(vocab_prune)
   dtm = create_dtm(it_train, vectorizer)
   # ajustar dtm con tfidf calculado anteriormente
   dtm_tfidf = fit_transform(dtm, tfidf)
   # Codificar en base a Glove calculdo anteriormente
   nword <- slam::row_sums(dtm, na.rm = T)
   #vamos a filtrar por documentos con mas de 0 palabras
   dtm_tfidf <- dtm_tfidf[nword>0,]
   docCod <- matprod_simple_triplet_matrix(dtm_tfidf %>% as.simple_triplet_matrix(),
                                           word_vectors %>% as.simple_triplet_matrix()
   )/nword[nword>0]
   return(list(docCod = docCod, nword = nword))
}


docCod <- textCod(text = train$text,
                   vocab = readRDS("vocab.rds"),
                   term_count_min = 10,
                   doc_count_min = 10,
                   cleanF = cleanText,
                   tfidf = readRDS("model_tfidf.rds"),
                   word_vectors = readRDS("word_vectors.rds"))




# Clasificacion con SOM (PENDIENTE)
somText <- readRDS("somText_wdata.rds")$somText
predText <- predict(somText, 
                    newdata = list(text = docCod$docCod))


#### AJUSTAR ####

predClass <- predText$unit.classif[docCod$nword>0]
classMatrix <- c(1:(somText$grid$xdim*somText$grid$ydim)) %>% 
   sapply(function(x) sum(predClass==x, na.rm = T))

plot(classMatrix/max(classMatrix))



predClass0 <- predText$unit.classif[train$target[docCod$nword>0]==0]
classMatrix0 <- c(1:(somText$grid$xdim*somText$grid$ydim)) %>% 
   sapply(function(x) sum(predClass0==x, na.rm = T))

plot(classMatrix0/max(classMatrix))


predClass1 <- predText$unit.classif[train$target[docCod$nword>0]==1]
classMatrix1 <- c(1:(somText$grid$xdim*somText$grid$ydim)) %>% 
   sapply(function(x) sum(predClass1==x, na.rm = T))



classM %>% is.na() %>% sum()


classM <- classMatrix1/(classMatrix1+classMatrix0)
classM[is.na(classM)] <- 0.5

classM <- classMatrix1/max(classMatrix)

range(classM)
plot(somText, type = "property",
     property = classM,
     palette.name = colors, 
     heatkey = TRUE, shape = "straight", na.color = colors(1))


predM <- predClass %>% sapply(function(x) classM[x])

auc(train$target[docCod$nword>0], predM)

view(data.frame(target = train$target[docCod$nword>0],predM = predM))



train$target[docCod$nword>0][predM==0] %>% mean(na.rm = T)

train$target[docCod$nword>0][predM==1] %>% mean(na.rm = T)



glmnet_glove = cv.glmnet(x = docCod$docCod, 
                         y = train$target[docCod$nword>0], 
                         family = "binomial",
                         type.measure = "auc",
                         nfolds = 3,
                         trace.it = 1)
plot(glmnet_glove)




