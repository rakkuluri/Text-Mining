rm(list=ls())

setwd("C:/Users/Ravinder/Desktop/Text Mining")

library(ggplot2)
library(caret)
require(tm)
library(data.table)


#load file
#---------------------------
xyztext.raw=read.csv("TextClassification_Data.csv", header = TRUE, stringsAsFactors = FALSE)
View(xyztext.raw)
#data cleanup
#---------------------------
xyztext.raw=xyztext.raw[,-1]
xyztext.raw=xyztext.raw[,-3]
xyztext.raw=xyztext.raw[,-1]

#according to doc specification only 5 categories, so 6th category JUNK have to be removed
xyztext.raw_nojunk=xyztext.raw[!(xyztext.raw$categories=="JUNK"),]
xyztext.raw_nojunk$categories=toupper(xyztext.raw_nojunk$categories)
xyztext.raw_nojunk$sub_categories=toupper(xyztext.raw_nojunk$sub_categories)

xyztext.raw_nojunk$categories=as.factor(xyztext.raw_nojunk$categories)
xyztext.raw_nojunk$sub_categories=as.factor(xyztext.raw_nojunk$sub_categories)


#understanding data
#--------------------------
prop.table(table(xyztext.raw_nojunk$categories))
prop.table(table(xyztext.raw_nojunk$sub_categories))

#Multivariate Scatter plot
xyztext.raw_nojunk$categories=as.factor(xyztext.raw_nojunk$categories)
xyztext.raw_nojunk$sub_categories=as.factor(xyztext.raw_nojunk$sub_categories)
ggplot(xyztext.raw_nojunk,aes_string(x=xyztext.raw_nojunk$categories,y=xyztext.raw_nojunk$sub_categories))+geom_point(aes_string(colour=xyztext.raw_nojunk$categories),size=4)+theme_bw()+xlab("Categories")+ylab("Sub Categories")+ggtitle("Distribution of categories and sub categories")+theme(text=element_text(size=6))+scale_colour_discrete(name="Categories")+scale_shape_discrete(name="Sub categories")


#splitting data
#-----------------------------
set.seed(32984)
indexes <- createDataPartition(xyztext.raw_nojunk$categories, times = 1,
                               p = 0.8, list = FALSE)
# createDataPartition from carret

train=xyztext.raw_nojunk[indexes,]
test=xyztext.raw_nojunk[-indexes,]

prop.table(table(train$categories))
prop.table(table(test$categories))


#term document matrix
#----------------------------------
# Create corpus
library(tm)
docs <- Corpus(VectorSource(train$SUMMARY))
summary(docs)

View(docs)

# Clean corpus
docs <- tm_map(docs, content_transformer(tolower))
docs <- tm_map(docs, removeNumbers)
docs <- tm_map(docs, removeWords, stopwords("english"))
docs <- tm_map(docs, removePunctuation)
docs <- tm_map(docs, stripWhitespace)
docs <- tm_map(docs, stemDocument, language = "english")
#convert to text document
#docs <- tm_map(docs, PlainTextDocument)

#feature engineering
#-------------------------------------
# Create dtm with TF
#dtm <- DocumentTermMatrix(docs)

#Create dtm with TF-IDF
dtm <-DocumentTermMatrix(docs,control = list(weighting = function(x) weightTfIdf(x, normalize = FALSE)))
#(log((total documents)/(number of docs with the term))

#inspect(dtm)
#dtm1 = as.matrix(dtm)
#View(dtm1)


#feature extraction
#------------------------------
#remove sparse
new_docterm_corpus <- removeSparseTerms(dtm,sparse = 0.99)
?removeSparseTerms()

colS <- colSums(as.matrix(new_docterm_corpus))

doc_features <- data.table(name = attributes(colS)$names, count = colS)

#install.packages('wordcloud')
library(wordcloud)
wordcloud(names(colS), colS, min.freq = 10, scale = c(6,.1), colors = brewer.pal(6, 'Dark2'))

