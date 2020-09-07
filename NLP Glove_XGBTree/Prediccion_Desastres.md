---
title: "GloVe y XGBTree para predecir desastres en Twitter"
author: "Desareca"
date: "06-09-2020"
output:
  html_document: 
    code_folding: hide
    df_print: default
    highlight: tango
    keep_md: yes
    theme: united
    toc: yes
    toc_float: yes
---

<style>
.list-group-item.active, .list-group-item.active:focus, .list-group-item.active:hover {
    background-color: #F84040;
}
</style>



# Resumen

El objetivo de este notebook es representar oraciones mediante el modelo **GloVe**. Este modelo normalmente se utiliza para representar palabras con muy buenos resultados, en este caso se hará una ponderación de los vectores de cada palabra para lograr un vector de cada oración tratada.

Una vez establecida la representación se determinan modelos para predecir la veracidad de desastres informados en varios textos. 

Para ello se utilizan dos datasets, el primero es un dataset peque?o con mensajes informando desastres en *twitter*, con esta base de datos se implementan los modelos. 

El segundo dataset es más grande y se utiliza para generar el modelo **GloVe**, esto es para lograr mayor generalización en la representación vectorial.



# GloVe (Global Vectors)

**GloVe** , acuñado de Global Vectors, es un modelo para la representación de palabras distribuidas. El modelo es un algoritmo de aprendizaje no supervisado para obtener representaciones vectoriales de palabras. Esto se logra mapeando palabras en un espacio significativo donde la distancia entre palabras está relacionada con la similitud semántica. El entrenamiento se realiza en estadísticas globales de co-ocurrencia palabra-palabra agregadas de un corpus, y las representaciones resultantes muestran interesantes subestructuras lineales del espacio vectorial de palabras [(Wikipedia)](https://en.wikipedia.org/wiki/GloVe_(machine_learning)).

El algoritmo **GloVe** consta de los siguientes pasos:

1- Recopilación de estadísticas de co-ocurrencia de palabras en una forma de matriz de co-ocurrencia $X$. Cada elemento $X_{ij}$ de dicha matriz representa la frecuencia con la que aparece la palabra $i$ en el contexto de la palabra $j$. Por lo general, se escanea el corpus de la siguiente manera: para cada término se buscan términos de contexto dentro de un ?rea definida por un tamaño de ventana antes del término y un tamaño de ventana después del término. También se da menos peso a las palabras más distantes, usualmente usando esta fórmula:

$$decay = \frac{1}{offset}$$

2- Definir restricciones suaves para cada par de palabras:

$$w_{i}^{T}w_j+b_i+b_j = log(X_{ij})$$

Donde:

  - $w_i$: vector palabra principal.
  - $w_j$: vector palabra contexto.
  - $b_i$: sesgo palabra principal.
  - $b_j$: sesgo palabra contexto.


3- Definir una función de costos:


$$J=\sum_{i=1}^{V}\sum_{j=1}^{V}f(X_{ij})(w_{i}^{T}w_j+b_i+b_j-logX_{ij})^2$$

Donde $f$ es una función de ponderación que ayuda a evitar el aprendizaje solo de pares de palabras extremadamente comunes. Los autores de **GloVe** eligen la siguiente función:


$$f(X_{ij})=\begin{cases} (\frac{X_{ij}}{x_{max}})^\alpha & \text{si X_{ij} > XMAX} \\1 & \text{en otro caso}\end{cases}$$

Para mayor información revisar ["GloVe: Global Vectors for Word Representation"](https://nlp.stanford.edu/pubs/glove.pdf), ["GloVe Word Embeddings"](http://text2vec.org/glove.html).


# Exploración Datasets.

## Real or Not? NLP with Disaster Tweets.



Twitter es un importante canal de comunicación en tiempos de emergencia. La ubicuidad de los teléfonos inteligentes permite a las personas anunciar una emergencia que están observando en tiempo real. Debido a esto, más agencias están interesadas en monitorear *Twitter* (es decir, organizaciones de ayuda ante desastres y agencias de noticias). Sin embargo, no siempre está claro si las palabras de una persona realmente anuncian un desastre. Este dataset representa una muestra de mensajes de emergencia en *Twitter* identificando si los desastres informados son reales o no.

A continuación se muestran las primeras 6 observaciones del dataset, que muestra que consta de 5 columnas:

- **id**: un identificador único para cada tweet. 
- **keyword**: una palabra clave en particular del tweet (puede estar en blanco).
- **location**: la ubicación desde la que se envió el tweet (puede estar en blanco).
- **text**: el texto del tweet. 
- **target**: indica si un tweet trata sobre un desastre real (1) o no (0). 


```r
dataDisaster <- fread("train.csv")
htmlTable(dataDisaster %>% head(),
          caption = "Tabla 1. Muestra del dataset.",
          tfoot = "&dagger; primeras 6 observaciones",
          col.rgroup = c("none","#FC7C7C"),
          css.cell = "padding-left: .5em; padding-right: .2em;")
```

<table class='gmisc_table' style='border-collapse: collapse; margin-top: 1em; margin-bottom: 1em;' >
<thead>
<tr><td colspan='6' style='text-align: left;'>
Tabla 1. Muestra del dataset.</td></tr>
<tr>
<th style='border-bottom: 1px solid grey; border-top: 2px solid grey;'> </th>
<th style='border-bottom: 1px solid grey; border-top: 2px solid grey; text-align: center;'>id</th>
<th style='border-bottom: 1px solid grey; border-top: 2px solid grey; text-align: center;'>keyword</th>
<th style='border-bottom: 1px solid grey; border-top: 2px solid grey; text-align: center;'>location</th>
<th style='border-bottom: 1px solid grey; border-top: 2px solid grey; text-align: center;'>text</th>
<th style='border-bottom: 1px solid grey; border-top: 2px solid grey; text-align: center;'>target</th>
</tr>
</thead>
<tbody>
<tr>
<td style='text-align: left;'>1</td>
<td style='padding-left: .5em; padding-right: .2em; text-align: center;'>1</td>
<td style='padding-left: .5em; padding-right: .2em; text-align: center;'></td>
<td style='padding-left: .5em; padding-right: .2em; text-align: center;'></td>
<td style='padding-left: .5em; padding-right: .2em; text-align: center;'>Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all</td>
<td style='padding-left: .5em; padding-right: .2em; text-align: center;'>1</td>
</tr>
<tr style='background-color: #fc7c7c;'>
<td style='background-color: #fc7c7c; text-align: left;'>2</td>
<td style='padding-left: .5em; padding-right: .2em; background-color: #fc7c7c; text-align: center;'>4</td>
<td style='padding-left: .5em; padding-right: .2em; background-color: #fc7c7c; text-align: center;'></td>
<td style='padding-left: .5em; padding-right: .2em; background-color: #fc7c7c; text-align: center;'></td>
<td style='padding-left: .5em; padding-right: .2em; background-color: #fc7c7c; text-align: center;'>Forest fire near La Ronge Sask. Canada</td>
<td style='padding-left: .5em; padding-right: .2em; background-color: #fc7c7c; text-align: center;'>1</td>
</tr>
<tr>
<td style='text-align: left;'>3</td>
<td style='padding-left: .5em; padding-right: .2em; text-align: center;'>5</td>
<td style='padding-left: .5em; padding-right: .2em; text-align: center;'></td>
<td style='padding-left: .5em; padding-right: .2em; text-align: center;'></td>
<td style='padding-left: .5em; padding-right: .2em; text-align: center;'>All residents asked to 'shelter in place' are being notified by officers. No other evacuation or shelter in place orders are expected</td>
<td style='padding-left: .5em; padding-right: .2em; text-align: center;'>1</td>
</tr>
<tr style='background-color: #fc7c7c;'>
<td style='background-color: #fc7c7c; text-align: left;'>4</td>
<td style='padding-left: .5em; padding-right: .2em; background-color: #fc7c7c; text-align: center;'>6</td>
<td style='padding-left: .5em; padding-right: .2em; background-color: #fc7c7c; text-align: center;'></td>
<td style='padding-left: .5em; padding-right: .2em; background-color: #fc7c7c; text-align: center;'></td>
<td style='padding-left: .5em; padding-right: .2em; background-color: #fc7c7c; text-align: center;'>13,000 people receive #wildfires evacuation orders in California </td>
<td style='padding-left: .5em; padding-right: .2em; background-color: #fc7c7c; text-align: center;'>1</td>
</tr>
<tr>
<td style='text-align: left;'>5</td>
<td style='padding-left: .5em; padding-right: .2em; text-align: center;'>7</td>
<td style='padding-left: .5em; padding-right: .2em; text-align: center;'></td>
<td style='padding-left: .5em; padding-right: .2em; text-align: center;'></td>
<td style='padding-left: .5em; padding-right: .2em; text-align: center;'>Just got sent this photo from Ruby #Alaska as smoke from #wildfires pours into a school</td>
<td style='padding-left: .5em; padding-right: .2em; text-align: center;'>1</td>
</tr>
<tr style='background-color: #fc7c7c;'>
<td style='background-color: #fc7c7c; border-bottom: 2px solid grey; text-align: left;'>6</td>
<td style='padding-left: .5em; padding-right: .2em; background-color: #fc7c7c; border-bottom: 2px solid grey; text-align: center;'>8</td>
<td style='padding-left: .5em; padding-right: .2em; background-color: #fc7c7c; border-bottom: 2px solid grey; text-align: center;'></td>
<td style='padding-left: .5em; padding-right: .2em; background-color: #fc7c7c; border-bottom: 2px solid grey; text-align: center;'></td>
<td style='padding-left: .5em; padding-right: .2em; background-color: #fc7c7c; border-bottom: 2px solid grey; text-align: center;'>#RockyFire Update => California Hwy. 20 closed in both directions due to Lake County fire - #CAfire #wildfires</td>
<td style='padding-left: .5em; padding-right: .2em; background-color: #fc7c7c; border-bottom: 2px solid grey; text-align: center;'>1</td>
</tr>
</tbody>
<tfoot><tr><td colspan='6'>
&dagger; primeras 6 observaciones</td></tr></tfoot>
</table>

De estas variables se utilizan sólo **text** como característica para predecir **target**. Se observa que el dataset tiene **7613** observaciones, donde **42.97%** de las observaciones son desastres reales.


```r
dataDisaster <- dataDisaster %>% select(text, target)
htmlTable(dataDisaster %>% summary(),
          caption = "Tabla 2. Distribución de variables.",
          tfoot = "",
          col.rgroup = c("none","#FC7C7C"),
          css.cell = "padding-left: .5em; padding-right: .2em;")
```

<table class='gmisc_table' style='border-collapse: collapse; margin-top: 1em; margin-bottom: 1em;' >
<thead>
<tr><td colspan='3' style='text-align: left;'>
Tabla 2. Distribución de variables.</td></tr>
<tr>
<th style='border-bottom: 1px solid grey; border-top: 2px solid grey;'> </th>
<th style='border-bottom: 1px solid grey; border-top: 2px solid grey; text-align: center;'>    text</th>
<th style='border-bottom: 1px solid grey; border-top: 2px solid grey; text-align: center;'>    target</th>
</tr>
</thead>
<tbody>
<tr>
<td style='text-align: left;'></td>
<td style='padding-left: .5em; padding-right: .2em; text-align: center;'>Length:7613       </td>
<td style='padding-left: .5em; padding-right: .2em; text-align: center;'>Min.   :0.0000  </td>
</tr>
<tr style='background-color: #fc7c7c;'>
<td style='background-color: #fc7c7c; text-align: left;'></td>
<td style='padding-left: .5em; padding-right: .2em; background-color: #fc7c7c; text-align: center;'>Class :character  </td>
<td style='padding-left: .5em; padding-right: .2em; background-color: #fc7c7c; text-align: center;'>1st Qu.:0.0000  </td>
</tr>
<tr>
<td style='text-align: left;'></td>
<td style='padding-left: .5em; padding-right: .2em; text-align: center;'>Mode  :character  </td>
<td style='padding-left: .5em; padding-right: .2em; text-align: center;'>Median :0.0000  </td>
</tr>
<tr style='background-color: #fc7c7c;'>
<td style='background-color: #fc7c7c; text-align: left;'></td>
<td style='padding-left: .5em; padding-right: .2em; background-color: #fc7c7c; text-align: center;'></td>
<td style='padding-left: .5em; padding-right: .2em; background-color: #fc7c7c; text-align: center;'>Mean   :0.4297  </td>
</tr>
<tr>
<td style='text-align: left;'></td>
<td style='padding-left: .5em; padding-right: .2em; text-align: center;'></td>
<td style='padding-left: .5em; padding-right: .2em; text-align: center;'>3rd Qu.:1.0000  </td>
</tr>
<tr style='background-color: #fc7c7c;'>
<td style='background-color: #fc7c7c; border-bottom: 2px solid grey; text-align: left;'></td>
<td style='padding-left: .5em; padding-right: .2em; background-color: #fc7c7c; border-bottom: 2px solid grey; text-align: center;'></td>
<td style='padding-left: .5em; padding-right: .2em; background-color: #fc7c7c; border-bottom: 2px solid grey; text-align: center;'>Max.   :1.0000  </td>
</tr>
</tbody>
<tfoot><tr><td colspan='3'>
</td></tr></tfoot>
</table>


## HC Corpora.


El dataset **[HC Corpora](https://twitter.com/hc_corpora)** está conformado por doce corpus divididos en cuatro idiomas (inglés, ruso, finlandés y alemán). Cada idioma tiene textos de twitter, blogs y sitios de noticias (en este caso se utiliza el corpus en inglés).


```r
news <- data.frame(text = readLines("en_US.news.txt", encoding = "UTF-8"),
                   stringsAsFactors = FALSE)
blogs <- data.frame(text = readLines("en_US.blogs.txt", encoding = "UTF-8"),
                    stringsAsFactors = FALSE)
twitter <- data.frame(text = readLines("en_US.twitter.txt", encoding = "UTF-8"),
                      stringsAsFactors = FALSE)

htmlTable(data.frame(obs = c(dim(blogs)[1], dim(news)[1], dim(twitter)[1]),
                     size = c(format(object.size(blogs), units = "Mb"),
                              format(object.size(news), units = "Mb"),
                              format(object.size(twitter), units = "Mb")),
                     row.names = c("blogs", "news", "twitter")),
          caption = "Tabla 3. Tamaño y peso de cada dataset.",
          col.rgroup = c("none","#FC7C7C"),
          css.cell = "padding-left: .5em; padding-right: .2em;")
```

<table class='gmisc_table' style='border-collapse: collapse; margin-top: 1em; margin-bottom: 1em;' >
<thead>
<tr><td colspan='3' style='text-align: left;'>
Tabla 3. Tamaño y peso de cada dataset.</td></tr>
<tr>
<th style='border-bottom: 1px solid grey; border-top: 2px solid grey;'> </th>
<th style='border-bottom: 1px solid grey; border-top: 2px solid grey; text-align: center;'>obs</th>
<th style='border-bottom: 1px solid grey; border-top: 2px solid grey; text-align: center;'>size</th>
</tr>
</thead>
<tbody>
<tr>
<td style='text-align: left;'>blogs</td>
<td style='padding-left: .5em; padding-right: .2em; text-align: center;'>899288</td>
<td style='padding-left: .5em; padding-right: .2em; text-align: center;'>255.4 Mb</td>
</tr>
<tr style='background-color: #fc7c7c;'>
<td style='background-color: #fc7c7c; text-align: left;'>news</td>
<td style='padding-left: .5em; padding-right: .2em; background-color: #fc7c7c; text-align: center;'>77259</td>
<td style='padding-left: .5em; padding-right: .2em; background-color: #fc7c7c; text-align: center;'>19.8 Mb</td>
</tr>
<tr>
<td style='border-bottom: 2px solid grey; text-align: left;'>twitter</td>
<td style='padding-left: .5em; padding-right: .2em; border-bottom: 2px solid grey; text-align: center;'>2360148</td>
<td style='padding-left: .5em; padding-right: .2em; border-bottom: 2px solid grey; text-align: center;'>319 Mb</td>
</tr>
</tbody>
</table>

```r
htmlTable(data.frame(dataset = c("blogs", "news", "twitter"),
           text = c(blogs$text[1], news$text[1], twitter$text[1])
           ),
          caption = "Tabla 4. Tamaño y peso de cada dataset.",
          col.rgroup = c("none","#FC7C7C"),
          css.cell = "padding-left: .5em; padding-right: .2em;")
```

<table class='gmisc_table' style='border-collapse: collapse; margin-top: 1em; margin-bottom: 1em;' >
<thead>
<tr><td colspan='3' style='text-align: left;'>
Tabla 4. Tamaño y peso de cada dataset.</td></tr>
<tr>
<th style='border-bottom: 1px solid grey; border-top: 2px solid grey;'> </th>
<th style='border-bottom: 1px solid grey; border-top: 2px solid grey; text-align: center;'>dataset</th>
<th style='border-bottom: 1px solid grey; border-top: 2px solid grey; text-align: center;'>text</th>
</tr>
</thead>
<tbody>
<tr>
<td style='text-align: left;'>1</td>
<td style='padding-left: .5em; padding-right: .2em; text-align: center;'>blogs</td>
<td style='padding-left: .5em; padding-right: .2em; text-align: center;'>In the years thereafter, most of the Oil fields and platforms were named after pagan “gods”.</td>
</tr>
<tr style='background-color: #fc7c7c;'>
<td style='background-color: #fc7c7c; text-align: left;'>2</td>
<td style='padding-left: .5em; padding-right: .2em; background-color: #fc7c7c; text-align: center;'>news</td>
<td style='padding-left: .5em; padding-right: .2em; background-color: #fc7c7c; text-align: center;'>He wasn't home alone, apparently.</td>
</tr>
<tr>
<td style='border-bottom: 2px solid grey; text-align: left;'>3</td>
<td style='padding-left: .5em; padding-right: .2em; border-bottom: 2px solid grey; text-align: center;'>twitter</td>
<td style='padding-left: .5em; padding-right: .2em; border-bottom: 2px solid grey; text-align: center;'>How are you? Btw thanks for the RT. You gonna be in DC anytime soon? Love to see you. Been way, way too long.</td>
</tr>
</tbody>
</table>

Se observa de la tabla 3 que el conjunto del dataset **HC Corpora** es bastante grande. La tabla 4 muestra una observación de cada subgrupo del dataset.


# Limpieza y tokenización.

Visto los datasets se procede a unirlos y limpiarlos, para ello consideraremos lo siguiente:

- Limpiar direcciones web, eliminando los  textos *http*, *https*, *com*, *co* y *org*.
- Eliminar signos de puntuación y números.
- Eliminar saltos de linea y espacios finales.
- Eliminar posibles emojis.
- Eliminar Stopwords.


```r
train <- rbind(news, blogs, twitter, dataDisaster$text) # hay alrededor de 3M de observaciones
rm(news, blogs, twitter, dataDisaster)
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
      text = tm::removeWords(text, letters) # elimina letras sueltas
      text = stringr::str_replace_all(text," $", "") # elimina espacios finales
      if(stopwords){text = tm::removeWords(text, tm::stopwords(language))} # elimina stopwords 
      if(is.null(remText)){text = tm::removeWords(text, remText)} # elimina palabras especificas 
      text = tm::stripWhitespace(text) # quita espacios en blanco repetidos
      text = stringr::str_replace_all(text,"^ ", "") # elimina espacios iniciales
      text = tm::stemDocument(text) # Stem words
      return(text)
}
it_train = itoken(train$text, 
                  preprocessor = cleanText, 
                  tokenizer = word_tokenizer,
                  n_chunks = 5,
                  progressbar = TRUE)

vocab = create_vocabulary(it_train, ngram = c(1L, 1L))
```


Con lo anterior se genera un vocabulario con 391796 términos, de los cuales a continuación se presentan los 40 más y menos frecuentes.

<br>

#### {.tabset .tabset-fade}

##### Palabras más frecuentes


```r
vocab_prune = vocab %>% 
      prune_vocabulary(term_count_min = 10, doc_count_min = 10) 

vocab[order(vocab$term_count, decreasing = TRUE),] %>% head(40) %>% 
      ggplot(aes(x=reorder(term, term_count), y=term_count)) +
      geom_bar(stat = "identity", fill="red", alpha = 0.7) +  coord_flip() +
      theme(legend.title=element_blank()) +
      xlab("Palabras") + ylab("Frecuencia") +
      labs(title = paste0("Palabras más frecuentes (",dim(vocab)[1], " palabras)"))+
      theme_linedraw()
```

![](Prediccion_Desastres_files/figure-html/mostFreqWords1-1.png)<!-- -->

##### Palabras menos frecuentes


```r
vocab[order(vocab$term_count, decreasing = TRUE),] %>% tail(40) %>% 
      ggplot(aes(x=reorder(term, term_count), y=term_count)) +
      geom_bar(stat = "identity", fill="red", alpha = 0.7) +  coord_flip() +
      theme(legend.title=element_blank()) +
      xlab("Palabras") + ylab("Frecuencia") +
      labs(title = paste0("Palabras menos frecuentes (",dim(vocab)[1], " palabras)"))+
      theme_linedraw()
```

![](Prediccion_Desastres_files/figure-html/mostFreqWords2-1.png)<!-- -->

##### Histograma


```r
vocab %>% 
   ggplot(aes(x=term_count %>% log())) +
   geom_histogram(fill="red", bins = 100, alpha = 0.7) + 
   xlab("Log(term_count)") + ylab("Frecuencia") + 
   labs(title = paste0("Histograma de conteo de palabras"))+
   theme_linedraw()
```

![](Prediccion_Desastres_files/figure-html/mostFreqWords3-1.png)<!-- -->

#### 

Como se observa en el histograma, existen muchos términos que aparecen sólo una vez. Estos términos no son útiles para el modelo y utilizan memoria, por lo que se eliminarán, como criterio se consideraron sólo los términos que aparezcan por lo menos 10 veces. Esto genera una reducción significativa, quedando 50247 términos en total.


# Vectorización utilizando **GloVe**.

La idea central al aplicar **GloVe** al problema es vectorizar una oración, dado que este modelo está orientado a palabras se procederá a determinar los vectores de las palabras para luego realizar una combinación de estos vectores y generar los vectores de las oraciones. Esto se fundamenta en la idea de que palabras relacionadas semánticamente tendrán un vector similar, esto hace que al ser agregados al vector de la oración se mantendrá un resultado similar.

## Vectores de palabras.

Con el vocabulario reducido se genera un modelo **GloVe** considerando 200 variables y 50 iteraciones. Con esto se reduce la dimensionalidad de manera significativa. A continuación se observa la evolución de la pérdida durante el entrenamiento.


```r
vectorizer = vocab_vectorizer(vocab_prune)
tfidf = TfIdf$new()
dtm = create_dtm(it_train, vectorizer)
dtm_tfidf = fit_transform(dtm, tfidf)
tcm <- create_tcm(it_train, vectorizer)

glove = GlobalVectors$new(rank = 200, x_max = 10, shuffle = TRUE)
word_vectors = glove$fit_transform(tcm, n_iter = 50)
```

```
INFO  [14:56:55.087] epoch 1, loss 0.2011 
INFO  [14:57:33.667] epoch 2, loss 0.1163 
INFO  [14:58:11.910] epoch 3, loss 0.1016 
INFO  [14:58:50.087] epoch 4, loss 0.0916 
INFO  [14:59:28.613] epoch 5, loss 0.0867 
INFO  [15:00:06.877] epoch 6, loss 0.0832 
INFO  [15:00:45.068] epoch 7, loss 0.0805 
INFO  [15:01:24.677] epoch 8, loss 0.0784 
INFO  [15:02:02.824] epoch 9, loss 0.0766 
INFO  [15:02:41.032] epoch 10, loss 0.0751 
INFO  [15:03:19.220] epoch 11, loss 0.0739 
INFO  [15:03:57.365] epoch 12, loss 0.0728 
INFO  [15:04:35.554] epoch 13, loss 0.0718 
INFO  [15:05:14.199] epoch 14, loss 0.0710 
INFO  [15:05:52.407] epoch 15, loss 0.0702 
INFO  [15:06:31.003] epoch 16, loss 0.0696 
INFO  [15:07:09.118] epoch 17, loss 0.0689 
INFO  [15:07:47.322] epoch 18, loss 0.0684 
INFO  [15:08:25.894] epoch 19, loss 0.0679 
INFO  [15:09:04.084] epoch 20, loss 0.0674 
INFO  [15:09:42.373] epoch 21, loss 0.0670 
INFO  [15:10:20.495] epoch 22, loss 0.0666 
INFO  [15:10:58.694] epoch 23, loss 0.0662 
INFO  [15:11:37.254] epoch 24, loss 0.0659 
INFO  [15:12:15.428] epoch 25, loss 0.0656 
INFO  [15:12:53.558] epoch 26, loss 0.0653 
INFO  [15:13:31.723] epoch 27, loss 0.0650 
INFO  [15:14:09.860] epoch 28, loss 0.0648 
INFO  [15:14:48.454] epoch 29, loss 0.0645 
INFO  [15:15:26.580] epoch 30, loss 0.0643 
INFO  [15:16:04.772] epoch 31, loss 0.0641 
INFO  [15:16:43.013] epoch 32, loss 0.0639 
INFO  [15:17:21.216] epoch 33, loss 0.0637 
INFO  [15:17:59.756] epoch 34, loss 0.0635 
INFO  [15:18:37.831] epoch 35, loss 0.0633 
INFO  [15:19:16.040] epoch 36, loss 0.0631 
INFO  [15:19:54.301] epoch 37, loss 0.0630 
INFO  [15:20:32.792] epoch 38, loss 0.0628 
INFO  [15:21:10.958] epoch 39, loss 0.0627 
INFO  [15:21:49.759] epoch 40, loss 0.0625 
INFO  [15:22:28.290] epoch 41, loss 0.0624 
INFO  [15:23:06.437] epoch 42, loss 0.0623 
INFO  [15:23:44.593] epoch 43, loss 0.0622 
INFO  [15:24:22.824] epoch 44, loss 0.0620 
INFO  [15:25:01.060] epoch 45, loss 0.0619 
INFO  [15:25:39.658] epoch 46, loss 0.0618 
INFO  [15:26:17.784] epoch 47, loss 0.0617 
INFO  [15:26:56.088] epoch 48, loss 0.0616 
INFO  [15:27:34.396] epoch 49, loss 0.0615 
INFO  [15:28:12.954] epoch 50, loss 0.0614 
```

```r
data.frame(loss = glove$get_history()[[1]], epoch = c(1:length(glove$get_history()[[1]]))) %>% 
   ggplot(aes(y = loss, x = epoch)) + 
   ggtitle("Entrenamiento modelo GloVe")+
   geom_line(colour = "red", size = 1) + 
   geom_point(colour = "red", size = 3) + 
   theme_linedraw()
```

![](Prediccion_Desastres_files/figure-html/GloveWord-1.png)<!-- -->

```r
rm(tcm, glove, train)
```


Para visualizar las relaciones generadas por el modelo se muestra a continuación una representación utilizando **t-SNE** en (*3D*). Para complementar la visualización se colorean las palabras en base a su sentimiento en **negro** para **neutro**, **azul** para **positivo** y **rojo** para **negativo**.



```r
# selecciona vocabulario mas frecuente
wvOrder <- vocab[order(vocab$term_count, decreasing = TRUE),][c(1:500),]
# calcula sentimientos
sentCalc <- function(name, row.names = TRUE){
      text <- name %>% t() %>% apply(1,function(x) get_sentiment(char_v = x, method = "syuzhet"))
      if(row.names){text <- text %>% as.data.frame(row.names = name)}
      if(!row.names){text <- text %>% as.data.frame(row.names = c(1:length(name)))}
      return(text)
}
em <- wvOrder$term %>% sentCalc()
em <- ifelse(em==0,'#000000',ifelse(em>0,'#0000FF','#FF0000'))
#
tsne <- Rtsne(word_vectors[wvOrder$term, ], 
              perplexity = 100, pca = FALSE, dims = 3,
              max_iter = 3000, verbose = FALSE)
tsne_plot <- tsne$Y %>% 
      as.data.frame() %>% 
      mutate(word = wvOrder$term)
p <- plot_ly(tsne_plot, x = ~V1, y = ~V2, z = ~V3, 
             mode = 'text', 
             type = 'scatter3d',
             text = ~word, textposition = 'middle right',
             textfont = list(color = em[,1], size = 12),
             width = 1000*0.75, height = 700*0.75) %>%
      layout(title = "Representación de palabras con GloVe", 
             scene = list(xaxis = list(title = 'x'),
                          yaxis = list(title = 'y'),
                          zaxis = list(title = 'z')
                          ))
p
```

<!--html_preserve--><div id="htmlwidget-dc0d31c369131ec18019" style="width:750px;height:525px;" class="plotly html-widget"></div>
<script type="application/json" data-for="htmlwidget-dc0d31c369131ec18019">{"x":{"visdat":{"40dc4a9312d8":["function () ","plotlyVisDat"]},"cur_data":"40dc4a9312d8","attrs":{"40dc4a9312d8":{"x":{},"y":{},"z":{},"mode":"text","text":{},"textposition":"middle right","textfont":{"color":["#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#0000FF","#0000FF","#000000","#000000","#0000FF","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#FF0000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#FF0000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#FF0000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#FF0000","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#0000FF","#0000FF","#000000","#0000FF","#000000","#000000","#000000","#FF0000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#0000FF","#0000FF","#000000","#000000","#0000FF","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#0000FF","#000000","#000000","#FF0000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#0000FF","#FF0000","#0000FF","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#0000FF","#0000FF","#000000","#000000","#000000","#000000","#FF0000","#000000","#FF0000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#0000FF","#000000","#000000","#000000","#000000","#FF0000","#000000","#000000","#0000FF","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#FF0000","#000000","#000000","#000000","#0000FF","#000000","#000000","#FF0000","#000000","#0000FF","#000000","#000000","#FF0000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#FF0000","#000000","#000000","#000000","#000000","#000000","#000000","#FF0000","#000000","#000000","#000000","#000000","#FF0000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#FF0000","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#FF0000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#FF0000","#000000","#0000FF","#000000","#000000","#FF0000","#000000","#000000","#0000FF","#000000","#000000","#000000","#FF0000","#FF0000","#0000FF","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#0000FF","#0000FF","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#FF0000","#0000FF","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#FF0000","#000000","#000000","#000000","#000000","#FF0000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#FF0000","#000000","#0000FF","#000000","#FF0000","#000000","#0000FF","#FF0000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#FF0000","#000000","#000000","#000000","#000000","#FF0000","#0000FF","#0000FF","#000000","#0000FF"],"size":12},"alpha_stroke":1,"sizes":[10,100],"spans":[1,20],"type":"scatter3d"}},"layout":{"width":750,"height":525,"margin":{"b":40,"l":60,"t":25,"r":10},"title":"Representación de palabras con GloVe","scene":{"xaxis":{"title":"x"},"yaxis":{"title":"y"},"zaxis":{"title":"z"}},"hovermode":"closest","showlegend":false},"source":"A","config":{"showSendToCloud":false},"data":[{"x":[-1.38130579617816,-0.403421178168294,-1.3458342010861,-2.89386282055525,-0.790210621165017,-0.768548243833261,-2.72867855687337,-1.77214559185279,-5.32432564914725,-6.16421073499655,-0.584622576072175,-0.347243395387371,-3.58412918444621,-3.53404183262322,-1.75537421922318,-2.42442165771563,-0.243173914056037,-2.9179371481171,-0.341828723893823,-1.97884532394713,-0.476783724438423,-4.68617144268436,3.62434643530142,-2.56705867302676,-2.77758637253129,0.0188379837612986,-4.37225654903463,-0.466968819868514,1.82801681125177,0.826266594263318,-6.12526162816192,2.67809612063662,0.810531083605693,-0.876848402739671,-0.539470736555447,0.979824132925484,0.200909863341861,-0.605251820438028,-1.60206313927722,-2.06437780208693,-1.47066303764804,0.282257877693905,-3.19730350377504,-1.54493527459065,-2.99523207246491,-3.79395273437837,2.10309168195906,-5.56534965167549,-1.0618035492155,5.19939397481159,-2.7031566684304,1.38763200304067,1.59632061452796,-1.56462662558786,0.0745412134576671,-1.71196629802152,-3.37762676756846,-3.18161264960467,-2.23009015415813,-7.82274235262853,-5.6333561002666,0.667932696515641,0.402834941198738,-1.06723929886007,1.07491683699658,5.3718882152553,-2.00205730965851,1.00129261585773,7.50213715858241,0.293344985315414,1.07622223412547,-6.21314578287135,-4.31014327617508,-3.33822887681797,1.79083106430247,-0.551880197597416,-1.14482205081243,-4.88758251483338,0.708032203169233,-3.79085264082437,0.643579232124467,-6.33399078287426,4.09208166367365,7.37867769054292,1.77991095153794,1.04818036617952,-2.34036505275025,-2.88156484510208,2.47017252173154,-0.474967252627379,-4.1162474444633,0.396622571960798,-7.85274736128299,0.252906936189692,-1.70234781028963,-1.42284157873358,-0.299337268306988,7.01435043579758,-2.40911921350793,0.751378649100767,0.456745210408893,-0.808734722663092,0.121912394780034,0.527105441307222,-1.12237345735963,0.778012998023893,4.74737657154038,-3.26326087873068,1.91530467486282,3.43856807642328,-1.62202266028924,0.38033961506771,3.42007134879306,1.75970942171136,-0.558055828658827,1.18388099150467,-0.933499915337344,-3.14705294469655,-2.75634400065069,5.44308001504637,-0.0765582629183264,-1.55704260083641,-6.87848269796583,-3.08397024945242,3.95683183386952,1.89188589241994,-4.89540492082048,-2.05706796853015,3.881433255519,-4.95089083261021,-0.488188639112505,-0.463197704444916,3.40967701587356,0.502081879263422,2.25048209802411,-1.72581037884262,6.97991777055898,1.51280658267065,0.507229948051054,0.318190400787148,7.20328116480733,-6.78745337355139,-3.99371328911198,2.73774478099982,1.88874468931237,-1.50976340414608,-1.15424459611108,3.02730711648213,-4.07362179614621,-5.38424508849758,-3.9265706326115,-0.811033137407996,2.3449520018788,-4.54040652608258,0.604552732238321,-7.35693968832145,-3.27170444304542,-3.06671253688139,-0.160269587951555,-0.403085758969128,0.616408279166422,-0.597598579504536,-1.58223256990836,0.0255467549108829,-7.47960180566389,5.64003497655923,3.4713921589149,1.19238203078262,-3.86618471552614,5.6417929340498,5.22565258448391,1.77582419054672,2.46400857069736,3.73380069504783,3.97985176770229,-6.29234910047173,5.16843955793087,0.347378008620884,2.47803326100378,2.97409160631789,-3.2014631356227,-3.11838995875478,-0.179914665150166,-6.50038241823551,-5.63749523949696,-4.91898113463558,-5.1479031215375,1.07332709217845,-1.90903503136771,4.90867008750387,2.37546256037047,-3.90411579410021,0.852337313892326,-5.25059757289098,-1.93209311855961,2.25238578349285,-2.87551144651048,-0.659292884090892,-0.863356148084015,-1.04317821931205,1.61856294735579,-2.67343997619702,4.69999741078528,-1.90101712740633,1.58991456935629,-1.78565473907644,-1.53672374939092,-6.8014499594293,3.43803175692722,3.45611685902954,-2.97408682733815,0.892541812063168,-2.13902755642659,1.58106394016023,-7.33487113237738,-3.1727939022674,8.40492983189974,1.62404491633635,-1.55046226786375,4.86111009371709,-0.660660762452918,4.55195239415898,0.819650250782762,1.52048088408692,1.26860969800252,-1.28993805934274,3.64008891137026,-1.73837311192447,-2.79285478149124,-4.80110531588624,6.62750428667388,-1.44144566395507,-4.83516671545219,-3.68277616450459,4.38979715999503,1.28851490590324,-0.265516822555441,2.21692430769921,3.72046803698681,-1.98222276952036,0.810801097110437,-4.35110787766055,0.543848522646452,1.86652688768488,-5.05215586026682,-0.228834575733578,-3.00477682291344,-0.5601373729703,2.07630147057346,0.876246501979554,-6.25570131993415,0.458143324014711,2.61877611452562,-2.86966423804957,-1.90532975982399,-0.76885561231702,3.13910406122146,-2.57809973062959,-0.656314683442239,0.168189904146449,3.92030366361768,-8.15775485782995,1.25130872837174,6.51324130427684,-5.13909499722504,2.76687421604496,3.32587728486135,-3.0664081990617,0.716530821560131,-3.32871183219373,-6.19879855488826,0.667563896220102,5.47464771120507,3.08854829480638,0.208504546259577,1.29580551885055,4.40991977050937,-2.04548253330193,-0.667290153860686,1.55609181374552,-0.782663339587978,4.21283348565612,-4.10679639059534,1.26242896227904,-3.23712785424605,0.204109177316949,0.396698851016256,-6.14907129213537,0.587702389061114,-4.35553006776751,-4.97752915177538,4.01355216533041,-0.466422578183601,-4.5702236738969,0.271478660519006,-4.51013397538311,-5.0255358331496,2.87967473688341,1.69396910694905,-1.07750382221432,-2.7162047303682,2.61012132821623,-0.595333588174776,2.84101780735719,-0.143361760089607,1.21904956587774,1.82148020544241,8.40776015946096,-1.72563810810199,-0.43786211250222,-4.97320280225765,5.10208751144337,4.75530185726983,2.95518005917379,-2.05068403212209,1.74304540389211,-3.10354407547008,-5.07163104141046,2.90392137479483,7.07188241192273,-0.88135867465566,4.32579994266285,1.46114909956198,-1.50179716580908,4.45956151867387,0.109775938985561,1.89139541860743,-2.36529214843498,5.91011929449367,-5.04088705718608,8.33155704265308,0.381136107896035,3.61829140719512,-2.01393968516169,0.036734599432242,-7.03323655182062,4.30841151198804,-1.71127425205662,-2.84764951333665,5.78920642835917,-1.38118051811333,3.01263351256793,2.79460376551589,-1.81825165696281,4.52153682997821,0.499562532299487,6.08854357455537,4.62326374656291,6.71287118831366,1.79666392875277,5.72222460159374,-2.7156499368676,4.17450704948542,-1.33016997620718,-2.97430900047875,6.32086206620698,-0.896154178940034,-2.02621502965825,5.67194359887966,1.77538590841018,5.69212018826084,-2.487744296887,-3.68617378033954,0.037064357493599,3.56788575595734,0.975563580351415,-4.04968871669624,1.26338998685201,-1.94701263355859,-0.753080099676863,5.42850330978161,2.72719152190631,-1.27899725229941,-2.71728386228718,4.33132662615184,0.38665081557584,-4.78979979398762,-1.31640282109543,-3.47766517564719,-1.57880528981682,-1.80719834232359,4.1112288283014,0.0441380891864912,-5.22524277172422,2.51332456164134,3.772480141765,2.25904949462042,1.02286203747683,-3.97292916203744,-4.6371402789188,3.74420887995789,2.86874796147541,4.48116361255635,8.23913921817422,0.463630720157735,-0.597245032474056,7.93428277508402,-2.04927094134079,4.02413856736841,-0.068823298127096,2.85427077569503,-0.0434815256482765,1.33821638279938,-0.162785494258217,2.51059757691474,0.72000559327564,0.585274567505046,-3.93111496669538,-1.14269281480622,3.12315144465235,-1.98346682214815,-1.96941071425871,0.453792960226593,3.03526572117372,2.72399984052373,-5.13465256240416,3.71663841168035,2.4328312765756,6.89950068325018,-0.565487676748919,3.45607101701983,5.77563805991917,4.54330862117396,0.830804819232501,-3.94866989627408,5.07942691809335,0.769952827976358,2.20683078620087,-2.60479055092003,-8.10799474249448,3.35799064436493,-0.854269551896523,-2.35722412519416,2.84119949674316,4.49542463587343,3.231094604007,0.801086196186017,0.168994080928402,0.615830003210123,1.98351773713298,2.96237238574137,-0.147566122777667,-8.02447921723449,5.39213178510584,-1.46749559582978,-3.60681286004504,-1.08846020385898,2.19230037463281,-1.62188355820667,3.07578793917338,3.51965646265675,3.83088626804469,-5.3657643375655,0.757676806739461,-2.08942098263234,2.59129251332015,-6.23828655969842,2.62055465975787,-3.10249763707739,-0.684410958039596,-1.67812311705253,-1.35479751294644,0.457499059505868,5.46090425354601,5.95248328784925,-0.533118761541964,1.23326510499505,0.612364262112789,0.980554767708443,-3.6429225778111,3.09595464682285,-5.77232510598578,-1.97100683038604,2.42163723654682,5.40310620497938,0.734806970463421,-6.36887623231883,0.984588606589192,4.57492337259382,1.72191168916699,-0.332013179993949,4.85691130060879,-1.40100328904492,-4.1390221924154,-2.24459999046545,1.49395695762684,-8.28466289550416,-5.22526967070652,5.09384526989413,2.0676187022022,-1.70247688496213,3.37263633434903,7.07282173889367,3.48765041509991,7.94881684067278,1.47375410304272,-8.51910056541633,4.0304041797364,3.74610331517408,4.25808416941313],"y":[-0.895331074368004,-1.28316902911382,-1.55751771919065,-0.712986060745873,1.467698762934,-1.36415326831298,-0.901435910131377,0.491476753723894,-3.35063687430501,1.02898168168233,0.66835551067372,-2.01603826328176,-1.11534653361355,-5.19383545605211,-0.269973962995714,-2.3225669319074,0.305200030773658,0.617986014976094,-1.31001717194577,-2.19179875907121,-1.57381846087054,2.46149803935844,-2.67480590470368,-1.03843252129878,0.181131857532884,-1.31473421558847,-1.69318049359389,-2.96317857619507,-1.24535515505893,-3.36160237618811,0.54889352048518,-1.84614475827691,2.03037624726817,-6.16799301014758,-0.821334657231806,-0.120915058933908,-0.090250587524041,0.153169533638239,0.463325289732575,-4.58754695545755,-3.47111959376729,0.544272842628079,-0.644935251624451,-0.400045465733602,2.35028192429307,1.71693195684277,1.24223392105838,1.62544356398327,0.386090210472875,-1.18052937301286,1.35231512804636,1.83955897125894,-4.22933160899602,-4.78699308050615,-0.971606338334508,-2.13199234689736,-2.85783136402724,-2.98648494352179,-3.03514846088358,-1.11952838816101,-4.52836285065247,-1.67895309492942,-3.48855273870582,4.0119809549202,-5.26579552208325,-1.17346231113782,3.41691285122446,-1.69200874671588,-2.92459225723522,-3.3769552297377,1.72908912561805,-0.596083498942372,-0.745766428927268,-3.10117712302834,-1.22468423937567,0.855078565242345,-0.342385624878208,5.70584989701455,-1.19146588761466,0.143409867750728,-0.769900549283269,-0.532391512418871,-1.71435238283872,-2.97589329895527,0.485312364892026,0.640999754210748,-0.830076264544685,1.10676257211869,-3.85313218200565,-0.489606767102218,-0.729242488260469,-3.9484507276143,-1.12220111498302,0.198871282965082,1.8885889964748,4.88363070108528,2.31208920136888,-2.84886297973369,2.41226922798082,-3.67048621191135,5.38356882651714,2.33015018214322,-0.725793344558442,-4.8416756252888,2.19543381097738,-1.88012539870689,-1.64448533955942,5.44873619067129,-0.302144371678946,-2.44057565423979,-3.84505907705149,-2.50337355681708,-4.71563639017768,-5.33698182509828,2.06081514221716,-3.94023898661711,-5.74354988479342,6.55037598060599,-3.0719796099214,-0.655321078515766,-4.10342010927142,-2.58729492899596,0.793039938418403,4.39313375615244,-4.1099310424366,1.91754275275944,5.78029171725114,-1.35763822646985,-5.92986153186064,-2.62331091441414,6.40129903768013,-7.7938894362234,-5.45822406682075,-0.914903682212887,-2.99081812613668,1.36358211206819,-2.32463116526016,-4.25995484326904,-5.4779058271496,0.527796836012161,-2.98066621270703,3.63426049707091,1.10410976372926,4.55209567922965,1.1000703610426,-1.15676399128734,3.64419926956153,1.54011118249818,-1.58373586276033,1.94757771461477,1.2004759824236,3.44445943113706,3.70388189892782,-2.54952372195005,0.382368721016711,-0.222023066065327,-5.72647853456891,-1.87980987636915,3.99355625649418,-8.15036435036101,0.821783088488799,-2.22746766679924,-0.325837185645582,2.97011777837534,-0.154274235858202,4.71645071331127,-4.76406497614928,0.289054931368205,-5.11949319040621,3.91950274007665,-0.88401466532994,0.472355215365515,5.46193540213871,-0.677684218935843,-0.170873393734679,-1.64265683327766,4.48893536023483,4.90019144437875,-1.01791549027384,-0.297825799542705,2.95825523919337,-2.93081807885603,0.726263254797063,-0.398416389907209,-1.89922905771042,-2.41396273678106,-2.37870205645443,0.795387226337787,-5.74265341624359,-6.69987669448305,2.4995788671958,-1.22862785647751,1.48530004571069,-2.14265679988345,3.90428715525985,0.724946437905034,6.66007471615987,-0.702493915810625,0.782733927369975,3.98480993058678,-1.01086068890214,-5.52730613058768,0.228811894854851,3.82031712170848,-0.787801077877526,-6.55972654089694,-4.65068120022779,3.74728974903551,-2.5168749874356,0.360054876447253,1.19159351538466,7.19044651671888,3.60042701664096,-1.34104144111186,-4.23930798743618,-5.84692725528074,-2.90560848532456,0.268029919165117,-3.53100846756636,-6.6918497765665,-0.539015592053427,-3.47587867421349,4.9503623652683,-2.70212209943922,-1.29475545480667,4.45724664593824,0.0071216349510475,-4.09260343091753,2.8361037548984,-1.00620435777302,-2.60541850898316,1.37476414084236,5.8260660730901,-1.75198623603367,3.99163230309268,1.26631906947445,-6.08920386556472,2.93245430480458,3.12226826642842,-4.45953962946398,1.15363857004783,2.89985835788966,4.49112455202322,-5.41788880928091,-0.720699507945937,4.71734569731595,2.89471077347012,6.30062707030677,-6.33559010772365,5.06192617394759,1.45525855382368,-3.70297283282638,-6.20881387351766,-5.13013141220195,-3.44855493420345,5.24252729741526,0.394537671908168,-5.86579963766216,8.32171706180252,1.51812220995346,1.35496265493957,0.193482301310586,-4.01179056485011,0.704797721261903,4.19799210232556,-6.264731964156,-0.975661505932632,3.00302610741944,3.70717212040495,2.48061034152068,-5.02444704634755,6.31200170868324,3.21634188518671,0.656855723228403,0.140482073826995,-1.73486357255867,-0.959885224118836,8.1431313952181,8.32321968087556,6.46219380118352,2.01292907710256,-5.77437155689229,-0.736202222541554,0.329513356648919,3.39950804208564,1.07587344511679,3.17583146914435,-1.70673846658472,1.98837305560817,1.52877748923047,1.70234528283167,1.03135220742027,-2.51997508860754,5.20171457143838,1.45025001861568,-2.35731376223778,0.569407050683944,-2.05006071711547,0.192880800127845,2.56460170646782,5.72815622635841,1.42159626974844,-0.534687929484835,3.912495717983,-4.21598885413269,-0.596196147237265,1.78502691675184,-2.89309559316554,-3.50892682780677,3.02399569399857,0.547215473621954,-0.885750213892967,-0.314000761628468,0.679928464057553,-4.65142794670058,-6.87997877993016,1.29388344396051,1.77856881826657,-4.27687559699361,-2.84569148279805,-1.66431441111615,0.829216442260899,2.47126113226847,1.82647425073521,2.70553996463133,-1.48805675275001,-2.33976404349312,1.6605137246152,3.75909354705269,-4.48648742031055,2.50776532403024,-2.59844831610221,-1.38252374530928,-4.90508329103594,4.03877539089642,0.502202327565434,5.88395632434543,-3.81192676688883,4.06390026265511,5.57240904509527,2.61786463988678,1.16998025620973,-1.51642715735917,5.47972576685885,4.15998923855643,-0.779440442026909,2.41092142051943,-0.5531868290948,0.758620827026094,3.5791467027384,3.91408659137729,3.32098838274253,0.270218346524468,2.71906632063918,3.52803773813035,3.88177546097701,3.14478983773887,8.11503269207684,5.56106601986924,3.29432902156169,6.06083745030852,1.79174759636543,-1.5699391106009,3.87556404861674,3.66834576890612,-7.41726737190005,-4.84010413890039,-1.72938469916521,1.99718017313145,5.10702048902152,1.75701191058454,-4.61405655175794,5.28871613855316,-2.65260849549759,3.98570073474003,-0.793044044818861,-1.7671817382125,7.48536079349731,-3.92523931543199,-0.58694624128253,-5.52714335321852,1.75760366662779,2.02583590221895,-4.76756855092582,1.52695392699456,0.377461191417063,1.46094550534031,-7.49798507539469,-5.12483277878563,5.23081967665598,0.246592780802219,1.35344495437596,1.54442835184254,2.52844859872059,-3.70712639070652,4.55611224949984,1.60318401177221,7.4569406331276,1.54219310190919,6.76606156126554,-4.18685342965936,-1.98105741928643,-1.71607768937797,2.38986388060512,-0.24954949875458,6.8669219124878,1.60103303457953,4.03834742226151,0.825678710623044,2.78073134312312,-2.49592210915947,-5.81314076917101,-0.0501029133196489,2.74388674810453,3.77454211976054,4.22799185456483,-5.84217055258002,2.57841423329661,-2.12505345335292,2.56824646312966,-3.83604820373921,1.70749980848107,1.69713211951004,0.744488020208904,4.04779531398949,3.38824234965072,2.38235796412739,4.33457195050356,-3.4197238127835,0.198218565694058,2.78761589092259,6.59336527007918,1.28186584864783,-4.35502884735298,-3.57168173693513,6.64308578419354,2.17623789907891,-0.267273543631842,-5.770876107075,1.63494040900013,4.44667320429834,-0.945346614016855,0.0968316128277285,3.35879820937416,0.70094752861006,-3.52021910628662,7.34593620552336,-2.09965215514919,-6.01957080171469,4.24508848561092,-0.521436416454158,3.68357548472106,-2.13062534307145,4.36769955705123,-4.80329498833651,-4.35899304420221,-2.45151559458846,-1.16035869558527,-5.33268702985676,-1.02298866432362,5.90821964634323,-2.43149326219335,1.78167162568913,1.98589284678934,0.747480625008849,-1.61691407619358,3.00195128587337,2.61899542220451,-4.02242551114206,0.929224319468325,1.4285478593083,0.436944925999642,-4.28876615636063,-4.60289683293731,1.84141725818211,3.12815226141273,-4.49965928466813,1.82354253011802,0.561148267935627,-4.93521999541522,-8.28845120667685,1.36863200760886,7.41784780536682,-4.10143145754772,3.73616475433947,-7.25961724482097,0.380186716241992,0.984138119762168,-0.93167913215944,1.26679660554826,-3.89870686285894,1.36025844278484,-1.09838565577998,3.1552905673928,1.63608512924605,-0.578024710228052,-1.76068078351859,4.76095162026425,-4.74190020682865,-5.80703998233591],"z":[-1.30708756582403,-2.64769730626703,-1.695347624016,-3.31743581522561,-2.24693214306587,-3.23079593405501,-2.16890274572478,-1.70518081341658,3.07286818595436,-1.0418693636495,-4.54451488194843,-0.431997539228396,1.66086913922456,0.991928081692334,-1.05861439466116,-2.61897731578328,-2.83760584082184,-5.03364642637609,-0.379879718298344,-2.25732464630758,-1.81538919295812,-1.65370427616477,0.712063591440429,-1.87369613682938,-1.04473080766359,-3.13536456112092,1.65565039070613,0.770630654889602,0.0639910924103024,-0.958414805836035,-0.624478324413809,-1.13324215857977,-4.9286044580316,-0.435105736618594,0.252475091766129,-0.772680294008594,-1.04189948642359,-0.850398574318376,-3.04950333303695,0.728270559422415,0.486627876713386,-0.498875797712242,-3.35116323292086,-0.596338541311845,-1.99543373101653,-1.29287139840426,5.52879751614583,-1.28565198326195,0.173429967952618,4.17294031809736,-1.31706663309547,-2.38061362636071,-1.25726645320169,4.20799019340835,-1.81899806043209,-0.720142874579756,0.391599550372708,-4.25683157515904,-0.0632818289454343,-0.777524927340312,0.991159608535021,0.342857890085567,-1.95544857797806,6.20680902494727,1.15867090285567,4.41983084425021,-2.46350161292743,-1.23012467511042,-0.41962523463884,3.33242422111745,-1.95899764143734,-5.4043683825359,1.97395519487423,-3.35399691673128,0.737532361409756,-4.58709758030018,-3.81431283042686,2.38271862348657,-3.78324260549778,-0.983994334573976,0.337884659790571,-5.24586978971001,-0.986268508335394,-0.442816801579626,-1.50224985041606,-3.32511419830507,1.53393691012593,3.40998257098396,6.65897143152419,0.227860356283167,-0.713968836377798,-0.816716888855946,-0.845817246312739,-1.13174020322854,1.09807944619337,-0.361546316286347,-1.52133535555309,-3.80826597165732,-0.8862321573347,-0.656920911785597,-1.05513894872202,-1.56139616390521,0.653047798843992,4.1452328678456,-3.4116642927086,-0.169405370516607,2.7479610678656,-3.07821911748534,0.653058852690993,0.922657542033495,3.15690468215214,1.44877421361284,-4.50552088453114,5.47804519052372,5.75492984263471,-0.871446710295506,-0.477091713177349,0.788087638615743,2.17088397134076,-1.78845712973548,1.19596746566061,1.73233242158797,-0.311546612874619,0.762209019504757,0.559827434303506,-1.60242581025515,2.54060768880472,0.131513463030026,3.03295653099728,1.02146510556534,0.933502464497515,0.180533163883264,4.32251913294213,1.72799376831527,1.29654141283508,-0.211402776198327,1.05617561969628,-2.52850297365608,-3.94072175663913,0.581691051917514,-0.502138428234867,-1.32371132873584,-1.58687816982998,-5.29562898982146,1.11718512236953,3.40905492686963,0.65939668586426,-1.82827308934359,3.51973409451301,-1.36718415121327,-3.23092196509644,-0.168065153543617,0.672328404530694,1.16152580538704,0.244774333603206,-0.418419943629188,-4.94161668540755,1.13701453472784,1.63207623803733,0.0428055804128195,2.85602449052583,1.44686379068791,2.77263773638878,-3.25318192413541,0.206218397442269,-0.0710124233973504,-4.59616465949097,0.0995619155475332,-1.53739988961154,-3.31934302633196,0.19397522637939,-4.90581518414277,-1.48121950351835,2.01302245445267,-1.6837403992722,-3.7654317484745,-2.82990463757649,3.10420108453084,2.96225795562832,-1.64777373603564,0.591566667627625,0.926916111517815,0.431315464766129,-4.76553802347153,3.49466761076296,1.98498553340014,2.22680697749877,-3.40326090388446,-3.59682661673933,0.185981189863623,1.52016345907902,-0.451221990295757,2.01781410441678,0.354877771595067,1.65358239831986,-0.2570393258267,0.839800907285995,1.17607965179649,0.237553558848883,6.1919622687496,1.8523273826802,2.5008449759048,2.22765646162568,1.39397639821715,1.72788367267255,-3.01128682826914,3.99903451577979,-1.35231016982411,-3.11737273468518,3.81585195017501,3.47565050808332,-1.02397830019813,-2.53884658627126,1.24701439737355,-0.412074268318201,-4.91739114200645,2.80560218068958,1.05030014563898,6.30977923725598,0.205069864475853,1.91300151377664,-0.929700915714594,4.57930197633508,1.09516105786704,-4.61549992264854,-4.09393243616672,4.90146659705458,3.59085002285841,0.525119612622737,-0.601648875342806,-3.81588002179839,-3.40069381743517,3.01225544338433,4.86606441640554,4.77905226318008,-0.641581801788567,2.29993087532021,-3.47886680475793,1.91972800081815,5.67241565030632,0.356471993292872,2.91782224095572,0.205650991769764,5.23065088978376,2.26980878864006,-2.66246756202317,-2.12983600176201,0.804669315436922,4.9263708906691,4.85768059394909,-3.40697047732029,-6.70705823013652,3.99542720935488,4.61717238616362,-3.22107013893519,-1.59106160905548,0.536646931783317,-4.34681945594806,-4.54176997258099,0.671289887128855,-4.34707974152798,-0.254905123319877,-0.77450023229165,1.13293280012175,5.60996599704458,4.00200781817857,2.64659546496904,0.346004734858917,-1.04811056548961,-4.76182049665602,1.05626691386037,-2.78420327841839,1.98580543450025,0.253783696633274,0.922089821770101,1.52491600170892,-3.66482900491786,3.28183693922175,-4.54394433353978,1.71283979404902,0.639631220356244,4.31560158752193,3.53794067900952,1.87716376172891,-6.27723621740419,1.94077148501779,0.490136948698155,-3.57516993934134,-0.629221170816182,-3.31603227770517,1.23142986074287,-4.75575896989733,2.01027774274909,-0.940985178414023,-7.86632492938409,-0.420171380911656,6.2891575215396,-4.44643711175989,3.60889106020906,-0.158673801961553,-2.98216150523,-2.86285564077961,6.25478137603371,-1.34438989897787,-6.34323306400801,4.38719056293719,2.28061075918423,2.79315832785999,-3.24026386618413,-6.46076642560345,6.23486462806617,-5.55263772758366,-0.481149253457094,-5.45310465141106,5.78245637184634,0.507949161737972,0.107907107181271,1.24256564698355,7.14891306901805,-3.7883361136639,2.13868771571129,-0.762677747703312,1.24609250290491,2.81254372480575,-0.250366986232742,2.71885788634702,2.48515751345601,-5.99670214548012e-005,-4.13427899793061,-1.4582707312306,-0.680703198752256,2.05909982012037,0.486327177286027,2.44892170557155,-5.77607907692821,1.38105091451245,-2.08730139560474,3.10226352024125,-1.72067349112072,-0.0565881325233318,3.511773474635,-0.687509937885679,4.75500584991953,3.13252313806761,-4.65455687741507,3.46979906013537,-0.984738158706597,1.0134059816491,1.07755817416818,-4.44741888920411,-3.81093434457595,4.13389306757233,-4.08735045893173,-1.10125936618655,2.58508078328927,-3.43798442802164,-4.34964110852171,3.257223232055,-0.0206695496500668,-0.606669805246106,0.0988643470664559,-1.19770179067797,5.46838063798317,-5.71422002498947,-4.85334927156868,-1.70498091571246,0.409339684801697,3.21442370431889,1.4716865034866,3.02388558154178,-4.11894325039058,-5.51132713833567,3.10032934184896,-6.26813347930521,4.66153235290355,6.24253273555761,4.13175211137684,-5.03948329352061,1.34883740823245,-6.71180781863985,-3.476493977383,4.80816447725623,-7.43381917651216,-3.53450018067344,-0.351075406431073,1.20568473509042,0.192282740601424,-1.63808287496314,-1.5208450115427,-0.9239929041862,-3.05879645914457,1.37224189988284,1.21044569293491,-0.670560137005948,-6.73813348266998,-1.10462948780499,-2.51034547123358,0.418333371587963,2.38716089531458,-5.15335955164913,7.09401025547788,1.63988127052108,-1.30366392069494,-6.91301266294842,-6.59179937000703,-0.788759930767696,7.30073833954567,-6.95375960797547,4.41269625427911,0.492742875637799,4.18504245920275,3.96413250416741,1.17643990950306,-2.6161595471543,3.6116748859448,5.57550591825374,3.88634200818912,-0.21118091362133,1.48358289609501,1.57233372525765,4.42230010725068,3.11284880491296,-1.4357349652818,0.826180854961937,-6.96318405476441,-2.70467891145925,0.0398148347010377,-3.21045342376626,3.58500064612313,-0.208028887465383,-1.00502792488228,-0.0128112027223225,5.845197935428,-0.825239874820528,-0.91127886731552,0.929277642986107,3.42104443599308,2.46703497551917,-4.19084579015818,0.579085727023203,0.507032541826373,2.02480713610978,-0.184696175255729,0.418663982158156,2.20758169734718,-5.04316274281358,-0.29431600400029,-2.54322074223837,4.15084801349304,4.29776582493276,1.09239753136855,-4.68690217884167,-2.68558908536097,1.71979553960315,6.11737400813186,6.83755965617922,2.36697483770318,-3.08960184986964,4.74358063348766,4.44238491139999,0.615952640195221,4.79008220272666,7.09223277616828,2.89590700278098,-1.47540760178315,2.66353859259425,-2.61844433478057,3.36074238208151,2.39194645608708,1.47643130897143,0.244427436919421,-0.229706516425269,5.3226367900244,-5.41620858433426,-3.71281511443294,-2.30568003849605,0.496493130733973,0.279967308789067,-2.23489229648941,6.34173788783351,-0.0694231795067775,0.0197640824243464,-5.08001843679741,-5.19724793087271,-2.63289093088643,-0.254705277938191,-0.294204304897455,3.85466080485332,-5.59975518846177,5.50046400762322,5.13544681538104,-0.25050639080541,-0.874594474391049,-5.83042806466967,-2.48041235081091,6.4196458213189,0.140963983961544,2.4908320498187,-3.81044601104478,4.37784167666668],"mode":"text","text":["just","can","like","get","one","will","go","time","love","day","make","know","good","thank","now","see","work","new","think","look","want","year","peopl","come","back","need","great","don","thing","say","today","us","use","follow","realli","way","much","well","take","rt","re","even","got","right","first","last","feel","week","still","life","start","also","said","lol","tri","let","hope","show","ll","night","happi","never","call","littl","friend","live","two","ve","book","im","mani","play","best","watch","someth","made","give","home","help","next","alway","game","world","read","may","find","better","keep","man","sure","wait","talk","tonight","lot","long","around","everi","school","end","tell","place","anoth","thought","guy","put","ever","god","big","mean","person","oh","dont","post","girl","old","ask","pleas","run","miss","chang","everyon","yes","morn","head","word","part","hous","stop","famili","fun","open","tweet","kid","happen","someon","sinc","stori","name","check","though","write","hour","final","free","seem","bad","away","differ","nice","month","done","turn","point","enjoy","actual","tomorrow","music","wonder","move","twitter","hard","didn","pretti","set","weekend","state","blog","might","meet","busi","learn","plan","high","believ","interest","win","job","hand","real","idea","went","wish","yet","team","beauti","awesom","amaz","found","hear","food","without","soon","enough","excit","leav","must","walk","mayb","alreadi","bit","everyth","hey","care","left","noth","sound","haha","minut","share","mind","stay","citi","three","anyth","parti","song","movi","kind","hate","eat","rememb","question","face","anyon","support","top","heart","yeah","came","readi","class","bring","room","cool","money","cours","fan","includ","becom","fuck","far","hit","togeth","boy","favorit","line","second","close","babi","eye","season","pictur","mom","gonna","saw","side","reason","listen","white","least","creat","friday","told","problem","sleep","mother","true","took","full","pick","birthday","light","power","fact","probabl","els","event","car","black","water","almost","children","perfect","sometim","buy","quit","whole","won","decid","finish","late","design","doesn","complet","card","forward","beer","list","moment","past","small","number","die","order","video","caus","abl","film","seen","cut","drink","group","experi","project","shit","phone","later","earli","women","student","guess","import","less","break","case","that","stuff","ago","product","visit","news","cant","other","sorri","add","summer","offic","ok","half","nation","fall","possibl","dream","sit","servic","serious","public","understand","issu","special","compani","sign","art","coupl","pass","market","cover","drive","countri","continu","american","begin","hot","ad","offer","send","welcom","wrong","short","hold","build","page","stand","record","pay","kill","sweet","red","glad","wear","heard","bodi","paper","rock","howev","matter","often","email","join","total","present","expect","deal","report","photo","along","social","park","act","color","men","didnt","ive","piec","challeng","area","young","store","lost","allow","crazi","ya","definit","base","save","bed","parent","clear","charact","rest","dog","human","success","either","shop","manag","instead","local","wow","sunday","direct","near","hair","speak","answer","train","easi","realiz","com","rather","lead","knew","saturday","law","goe","perform","street","mention","ur","spend","agre","provid","chanc","step","ass","woman","inspir","note","wanna","sad","outsid","funni","age","natur","inform","isn","recent","quick","ill","date","certain","yesterday","damn","site","develop","sever","celebr","usual","futur","ladi","facebook","posit","green","danc","four","text","monday","star","member","felt","hell","consid","author","receiv","media","fight","dinner","grow","comment","child"],"textposition":["middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right","middle right"],"textfont":{"color":["#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#0000FF","#0000FF","#000000","#000000","#0000FF","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#FF0000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#FF0000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#FF0000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#FF0000","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#0000FF","#0000FF","#000000","#0000FF","#000000","#000000","#000000","#FF0000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#0000FF","#0000FF","#000000","#000000","#0000FF","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#0000FF","#000000","#000000","#FF0000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#0000FF","#FF0000","#0000FF","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#0000FF","#0000FF","#000000","#000000","#000000","#000000","#FF0000","#000000","#FF0000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#0000FF","#000000","#000000","#000000","#000000","#FF0000","#000000","#000000","#0000FF","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#FF0000","#000000","#000000","#000000","#0000FF","#000000","#000000","#FF0000","#000000","#0000FF","#000000","#000000","#FF0000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#FF0000","#000000","#000000","#000000","#000000","#000000","#000000","#FF0000","#000000","#000000","#000000","#000000","#FF0000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#FF0000","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#FF0000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#FF0000","#000000","#0000FF","#000000","#000000","#FF0000","#000000","#000000","#0000FF","#000000","#000000","#000000","#FF0000","#FF0000","#0000FF","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#0000FF","#0000FF","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#FF0000","#0000FF","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#FF0000","#000000","#000000","#000000","#000000","#FF0000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#FF0000","#000000","#0000FF","#000000","#FF0000","#000000","#0000FF","#FF0000","#000000","#0000FF","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#000000","#0000FF","#000000","#000000","#FF0000","#000000","#000000","#000000","#000000","#FF0000","#0000FF","#0000FF","#000000","#0000FF"],"size":12},"type":"scatter3d","marker":{"color":"rgba(31,119,180,1)","line":{"color":"rgba(31,119,180,1)"}},"error_y":{"color":"rgba(31,119,180,1)"},"error_x":{"color":"rgba(31,119,180,1)"},"line":{"color":"rgba(31,119,180,1)"},"frame":null}],"highlight":{"on":"plotly_click","persistent":false,"dynamic":false,"selectize":false,"opacityDim":0.2,"selected":{"opacity":1},"debounce":0},"shinyEvents":["plotly_hover","plotly_click","plotly_selected","plotly_relayout","plotly_brushed","plotly_brushing","plotly_clickannotation","plotly_doubleclick","plotly_deselect","plotly_afterplot","plotly_sunburstclick"],"base_url":"https://plot.ly"},"evals":[],"jsHooks":[]}</script><!--/html_preserve-->


La representación selecciona las 500 palabras más comunes. Se observan varias relaciones de palabras con un contexto similar, como por ejemplo:

- day, yesterday, today, month, week, weekend, tomorrow, season, year, saturday, sunday.
- tweet, twitter, facebook.
- write, read, book.
- women, woman, man, men.
- girl, boy, baby.
- kid, children, child, mom, mother, family, parent.
- happy, birthday, celebr.
- white, black.
- red, green.
- etc.

Estas relaciones semánticas son útiles para poder codificar oraciones completas. Además, se tiene que las palabras **negativas** y **positivas** no son cercanas, a excepción de algunos casos aislados como **white**/**black** y **money**/**pay**.





## Vectores ponderados de oraciones.

Para intentar representar oraciones que incluyan relaciones semánticas se considera la suma ponderada utilizando los vectores del modelo **GloVe** generados en el punto anterior. Esto se lleva a cabo de la siguiente manera:

$$V_s=\frac{\sum_{i=1}^{n}{\overline{dtm_i}V_{w_i}}}{n}$$

Donde:

  - $V_s$: Vector ponderado de la oración.
  - $V_{w_i}$: Vector de cada palabra $w_i$.
  - $\overline{dtm_i}$: Factor de ponderación de cada $w_i$. Este se determina a partir de la $dtm$ normalizada mediante ***Tf-Idf***.
  - $n$: Numero de palabras en la oración.

Se utiliza la normalización ***Tf-Idf*** para que las palabras más comunes sean menos representativas y no influyan de sobremanera en el promedio.

Aplicando nuevamente **t-SNE**(*2D*) al **dtm** (Document-term matrix), **dtm tfidf** (Document-term matrix normalizado Tf-Idf) y al modelo **GloVe** se observa que en ningún caso hay una clara relación entre el target (desastre/no desastre) y los datos. Por otro, lado también se observa que ambos **dtm** presentan algunos patrones, que en el caso de **Glove** se tienden a perder.

<br>

#### {.tabset .tabset-fade}

##### Vectorización con GloVe


```r
trainSample <- fread("train.csv") %>% select(text, target)

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
      text = tm::removeWords(text, letters) # elimina letras sueltas
      text = stringr::str_replace_all(text," $", "") # elimina espacios finales
      if(stopwords){text = tm::removeWords(text, tm::stopwords(language))} # elimina stopwords
      if(is.null(remText)){text = tm::removeWords(text, remText)} # elimina palabras especificas
      text = tm::stripWhitespace(text) # quita espacios en blanco repetidos
      text = stringr::str_replace_all(text,"^ ", "") # elimina espacios iniciales
      text = tm::stemDocument(text) # Stem words
      return(text)
}

vocab_prune = vocab %>% prune_vocabulary(term_count_min = 10, doc_count_min = 10)
# limpiar y crear dtm con el nuevo texto
it_train = itoken(trainSample$text,
                  preprocessor = cleanText,
                  tokenizer = word_tokenizer,
                  n_chunks = 5,
                  progressbar = FALSE)

vectorizer = vocab_vectorizer(vocab_prune)
dtm = create_dtm(it_train, vectorizer)
# ajustar dtm con tfidf calculado anteriormente
dtm_tfidf = transform(dtm, tfidf)
# Codificar en base a Glove calculdo anteriormente (normalizado)
nword <- slam::row_sums(dtm, na.rm = T)
nword[nword==0] <- 1


docCod <- slam::matprod_simple_triplet_matrix(
   slam::as.simple_triplet_matrix(dtm_tfidf),
   slam::as.simple_triplet_matrix(word_vectors)
   )/nword

docCod <- list(docCod = docCod, nword = nword)

docCodS <- docCod$docCod

set.seed(1)
samp <- sample.int(dim(trainSample)[1], 500, replace = F)

# t-SNE GloVe
tsne2 <- Rtsne(docCodS[samp,], 
               perplexity = 0.3*length(samp), pca = TRUE, dims = 2,
               check_duplicates = F, max_iter = 1000, theta = 0.1, eta = 500,
               verbose = F, exaggeration_factor = 30)

tsne_plot2 <- tsne2$Y %>%
   as.data.frame() %>%
   mutate(Sentence = ifelse(trainSample$target[samp]==0,'no disaster','disaster') %>% as.factor())

tsne_plot2 %>% 
   GGally::ggpairs(columns = c(1,2), 
                   aes(colour=Sentence),
                   upper = list(continuous = GGally::wrap("cor", alpha = 1, size = 8)),
                   lower = list(continuous = GGally::wrap("points", alpha = 0.5, size = 6)),
                   diag = list(continuous = GGally::wrap("densityDiag", alpha = 0.7)),
                   title = "Representación t-SNE de vectorización GloVe"
                   )+
   scale_fill_manual(values = c("#FF0000", "#000000"))+
   scale_color_manual(values = c("#FF0000", "#000000"))+
   theme_linedraw()
```

![](Prediccion_Desastres_files/figure-html/GloveSentence1-1.png)<!-- -->

##### Vectorización con DTM


```r
# t-SNE dtm
tsne2 <- Rtsne(dtm[samp,] %>% as.matrix(), 
               perplexity = 0.3*length(samp), pca = TRUE, dims = 2,
               check_duplicates = F, max_iter = 1000, theta = 0.1, eta = 500,
               verbose = F, exaggeration_factor = 30)

tsne_plot2 <- tsne2$Y %>%
   as.data.frame() %>%
   mutate(Sentence = ifelse(trainSample$target[samp]==0,'no disaster','disaster') %>% as.factor())

tsne_plot2 %>% 
   GGally::ggpairs(columns = c(1,2), 
                   aes(colour=Sentence),
                   upper = list(continuous = GGally::wrap("cor", alpha = 1, size = 8)),
                   lower = list(continuous = GGally::wrap("points", alpha = 0.5, size = 6)),
                   diag = list(continuous = GGally::wrap("densityDiag", alpha = 0.7)),
                   title = "Representación t-SNE de Document-term matrix"
                   )+
   scale_fill_manual(values = c("#FF0000", "#000000"))+
   scale_color_manual(values = c("#FF0000", "#000000"))+
   theme_linedraw()
```

![](Prediccion_Desastres_files/figure-html/GloveSentence2-1.png)<!-- -->

##### Vectorización con DTM normalizado Tf-Idf


```r
# t-SNE dtm_tfidf
tsne2 <- Rtsne(dtm_tfidf[samp,] %>% as.matrix(), 
               perplexity = 0.3*length(samp), pca = TRUE, dims = 2,
               check_duplicates = F, max_iter = 1000, theta = 0.1, eta = 500,
               verbose = F, exaggeration_factor = 30)

tsne_plot2 <- tsne2$Y %>%
   as.data.frame() %>%
   mutate(Sentence = ifelse(trainSample$target[samp]==0,'no disaster','disaster') %>% as.factor())

tsne_plot2 %>% 
   GGally::ggpairs(columns = c(1,2), 
                   aes(colour=Sentence),
                   upper = list(continuous = GGally::wrap("cor", alpha = 1, size = 8)),
                   lower = list(continuous = GGally::wrap("points", alpha = 0.5, size = 6)),
                   diag = list(continuous = GGally::wrap("densityDiag", alpha = 0.7)),
                   title = "Representación t-SNE de Document-term matrix normalizado Tf-Idf"
                   )+
   scale_fill_manual(values = c("#FF0000", "#000000"))+
   scale_color_manual(values = c("#FF0000", "#000000"))+
   theme_linedraw()
```

![](Prediccion_Desastres_files/figure-html/GloveSentence3-1.png)<!-- -->

#### 




# Modelo XGBTree

Para predecir consideraremos un modelo ***XGB Tree*** y los siguientes parámetros:

- Validación cruzada con 10 folds.
- Búsqueda de hiperparámetros aleatoria, con 50 búsquedas.
- Muestreo ***up***, esto debido a que está levemente desbalanceada el target.
- Se centrarán y escalarán las variables predictoras.
- Métrica ***ROC***.


```r
fitControl <- trainControl(method = "cv",
                           number = 10,
                           search = "random",
                           summaryFunction = twoClassSummary,
                           classProbs = TRUE,
                           sampling = "up",
                           verboseIter = FALSE
                           )

dataGloVe <- data.frame(target = factor(trainSample$target, labels = c("X0", "X1")), docCod$docCod)

set.seed(1)
xgbTreeGloVe <- caret::train(target ~.,
                             data = dataGloVe,
                             preProcess = c("center", "scale"),
                             method = "xgbTree",
                             trControl = fitControl,
                             metric = "ROC",
                             tuneLength = 50,
                             verbose = FALSE
                             )
ggplot(xgbTreeGloVe) + ggtitle("Resultado del Modelo XGBTree con GloVe")+
   geom_point(colour = "red", size = 3) + 
   theme_linedraw()
```

![](Prediccion_Desastres_files/figure-html/GloveXGBTree-1.png)<!-- -->

Este modelo entrega buenos resultados, con un ***ROC*** mayor a **0.86**. Si lo comparamos con un [modelo anterior](https://rpubs.com/desareca/NLP_getting_started_xgbTree) que como máximo llegó a **0.83** es una buena mejora.

Es necesario considerar que este modelo se desarrolló con una vectorización **GloVe** de 200 variables en oposición al modelo con el que se compara que utiliza 1122 variables.


# Observaciones

Es posible generar una vectorización a partir de **GloVe** para oraciones, aunque es necesario considerar una base de datos los suficientemente grande para la vectorización de palabras y que el modelo sea representativo y se puedan generalizar las relaciones entre palabras. De lo anterior y como este modelo fue desarrollado en un computador personal, es perfectamente mejorable el modelo de vectorización de palabras.

La ponderación utilizando **Tf-Idf** resultó ser una buena opción. Es posible mejorar estos pesos utilizando otro modelo que determine pesos optimos para cada problema, pero como primer paso es una buena opción.



