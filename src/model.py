import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pickle # pacote para salvar o modelo

dataFrame = pd.read_csv('../data/movie_metadata.csv')
# retorna as primeiras 5 linhas da tabela
# print(dataFrame.head())

# retorna uma tupla com linha x colunass
# print(dataFrame.shape)

# coluna x tipo do dado
# print(dataFrame.dtypes)

#lista o nome das colunas do dataframe
# print(list(dataFrame.columns))

#descartando a coluna com o link do IMDB do filme
dataFrame.drop('movie_imdb_link', axis=1, inplace=True)

# retorna quantos valores diferentes nessa coluna e quantas ocorrências de cada um 
# print(dataFrame["color"].value_counts())

dataFrame.drop('color', axis=1, inplace=True)

#verificando quais colunas possuem valores nulos
# print(dataFrame.isna().any())

#análise quantitativa de dados nulos
# print(dataFrame.isna().sum())

# descartando linhas das colunas com baixa ocorrência de nulos
dataFrame.dropna(axis=0, subset=['director_name', 'num_critic_for_reviews',
                               'duration','director_facebook_likes','actor_3_facebook_likes',
                               'actor_2_name','actor_1_facebook_likes','actor_1_name','actor_3_name',
                               'facenumber_in_poster','num_user_for_reviews','language','country',
                               'actor_2_facebook_likes','plot_keywords', 'title_year'],inplace=True)

# Verificar das colunas que tem muitos valores nulos, qual valor é mais frequente
# print(dataFrame["content_rating"].value_counts())

# substituir os nulos pelo valor mais frequente da coluna
dataFrame['content_rating'].fillna('R', inplace=True)

# substituir os valores nulos pela mediana dos valores
dataFrame['aspect_ratio'].fillna(dataFrame['aspect_ratio'].median(), inplace=True)
dataFrame['budget'].fillna(dataFrame['budget'].median(), inplace=True)
dataFrame['gross'].fillna(dataFrame['gross'].median(), inplace=True)

#verificando se ainda há linhas com valores nulos
# dataFrame.isna().sum()

#retorna o número de linhas duplicadas
# dataFrame.duplicated().sum()

# removendo linhas duplicadas
dataFrame.drop_duplicates(inplace=True)

dataFrame.drop('language', axis=1, inplace=True)
dataFrame.drop('country', axis=1, inplace=True)

# Criando uma nova coluna com o ganho real do filme (ganhos totais - orçamento)
dataFrame['Profit'] = dataFrame['budget'].sub(dataFrame['gross'], axis=0)

# Criando uma nova coluna com o ganho real do filme em porcentagem
dataFrame['Profit_Percentage'] = (dataFrame['Profit']/dataFrame['gross'])*100

# print(dataFrame.head())

#salvando o dataframe em outro csv(vai que quebra né, backup nunca fez mal a ningém)
dataFrame.to_csv('temp-data/dados_imdb_analiseexpl.csv', index=False)

#criando gráfico de correlaciona lucro e nota do IMDB
ggplot(aes(x='imdb_score', y='Profit'), data=dataFrame) +\
    geom_line() +\
    stat_smooth(colour='blue', span=1)

#criando gráfico de correlaciona likes no facebook do filme e nota do IMDB

(ggplot(dataFrame)+\
    aes(x='imdb_score', y='movie_facebook_likes') +\
    geom_line() +\
    labs(title='Nota no IMDB vs likes no facebook do filme', x='Nota no IMDB', y='Likes no facebook')
)

#gráfico dos primeiros 20 filmes com melhor nota x atores principais
plt.figure(figsize=(10,8))

dataFrame= dataFrame.sort_values(by ='imdb_score' , ascending=False)
dataFrame2=dataFrame.head(20)
ax=sns.pointplot(dataFrame2['actor_1_name'], dataFrame2['imdb_score'], hue=dataFrame2['movie_title'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()

#retirando algumas colunas com dados categóricos
dataFrame.drop(columns=['director_name', 'actor_1_name', 'actor_2_name', 
                 'actor_3_name', 'plot_keywords', 'movie_title', 'genres'], axis=1, inplace=True)

#retirando as colunas criadas, serviram para visualizar os dados, mas não é boa prática ter colunas dependentes entre si
dataFrame.drop(columns=['Profit', 'Profit_Percentage'], axis=1, inplace=True)

# Visualizando mapa de calor para identificar colunas que são dependentes entre si
corr = dataFrame.corr()
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})
plt.figure(figsize=(13,7))
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask, 1)] = True
a = sns.heatmap(corr,mask=mask, annot=True, fmt='.2f')
rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)
roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)

#criando uma nova coluna combinando as duas colunas muito correlacionadas
dataFrame['Other_actors_facebook_likes'] = dataFrame['actor_2_facebook_likes'] + dataFrame['actor_3_facebook_likes']

#criando uma nova coluna combinando as duas colunas muito correlacionadas
dataFrame['critic_review_ratio'] = dataFrame['num_critic_for_reviews']/dataFrame['num_user_for_reviews']

#removendo as colunas muito dependented
dataFrame.drop(columns=['actor_2_facebook_likes', 'actor_3_facebook_likes',
                 'cast_total_facebook_likes', 'num_critic_for_reviews', 'num_user_for_reviews'], axis=1, inplace=True)

#categorizando os valores de nota do imdb
dataFrame['imdb_binned_score']=pd.cut(dataFrame['imdb_score'], bins=[0,4,6,8,10], right=True, labels=False)+1

# transforma todos os vaalores da coluna em novas colunas e atribuindo valor boleano
dataFrame = pd.get_dummies(data = dataFrame, columns=['content_rating'], prefix=['content_rating'], drop_first=True)

# Backup pt 2
dataFrame.to_csv('temp-data/dados_imdb_com_nota.csv', index=False)

# selecionando os inputs do modelo
X=pd.DataFrame(columns=['duration','director_facebook_likes','actor_1_facebook_likes','gross',
                        'num_voted_users','facenumber_in_poster','budget','title_year','aspect_ratio',
                        'movie_facebook_likes','Other_actors_facebook_likes','critic_review_ratio',
                        'content_rating_G','content_rating_GP',
                        'content_rating_M','content_rating_NC-17','content_rating_Not Rated',
                        'content_rating_PG','content_rating_PG-13','content_rating_Passed',
                        'content_rating_R','content_rating_TV-14','content_rating_TV-G',
                        'content_rating_TV-PG','content_rating_Unrated','content_rating_X'],data=dataFrame)

#[TREINAMENTO SUPERVISIONADO], definindo as respostas para servir de comparação pro modelo
y = pd.DataFrame(columns=['imdb_binned_score'], data=dataFrame)

#dividindo os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#normalizando e transformando os dados para que não fiquem com valores tão distoantes
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)

# Aplicando normalização no conjunto de teste
X_test = sc_X.transform(X_test)

#[ treinamento com REGRSSÂO LOGISTICA ]
logit =LogisticRegression(verbose=1, max_iter=1000)
logit.fit(X_train,np.ravel(y_train,order='C'))
y_pred=logit.predict(X_test)

#os valores preditos vem em um array
# print(y_pred)

# para visualizá-los melhor printamos a matriz de confusão
cnf_matrix =  metrics.confusion_matrix(y_test, y_pred)

#  fução de boniteza pra matriz de confusão
# alternativa:
# print(cnf_matrix)
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

#imprimindo a matriz de confusão bonita
plot_confusion_matrix(cnf_matrix, classes=['1','2', '3', '4'],
                      title='Matriz de confusão não normalizada', normalize=False)

#métricas finais
print(metrics.classification_report(y_test, y_pred, target_names=['1','2', '3', '4']))

#definindo em qual caminho vamos salvar o modelo
modelo_treinado = 'modelo_imdb.sav'

#salvando o modelo
pickle.dump(logit, open(modelo_treinado, 'wb'))

#carregando o modelo treinado
modelo_carregado = pickle.load(open(modelo_treinado, 'rb'))

#Olhando o conteúdo de um vetor de teste
X_test[0]

#fazendo predição do novo dado com o modelo carregado
print(modelo_carregado.predict([X_test[0]]))

