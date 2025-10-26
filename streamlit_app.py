import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# Funções comuns
def carregar_dados(caminho_excel="mangas.xlsx", nome_planilha=None):
    excel_data = pd.ExcelFile(caminho_excel)
    if nome_planilha is None:
        nome_planilha = excel_data.sheet_names[0]  # Pega a primeira aba
    df = pd.read_excel(excel_data, sheet_name=nome_planilha)

    # 🔹 Normaliza os nomes das colunas (resolve "Descrição", "Descricao", " descrição ", etc.)
    df.columns = [
        col.strip()
        .replace("ç", "c")
        .replace("Ç", "C")
        .replace("ã", "a")
        .replace("õ", "o")
        .replace("é", "e")
        .replace("ê", "e")
        .replace("í", "i")
        .replace("Á", "A")
        .replace("É", "E")
        .replace("Í", "I")
        .replace("Ó", "O")
        .replace("Ú", "U")
        .lower()
        for col in df.columns
    ]

    # Ajuste das colunas esperadas
    colunas = [
        "id", "titulo_romaji", "titulo_ingles", "titulo_nativo", "generos", "tags",
        "nota_media", "popularidade", "favoritos", "capitulos", "volumes",
        "status", "relacoes", "autores", "capa"
    ]
    if "descricao" in df.columns:  # 🔹 Agora detecta automaticamente
        colunas.append("descricao")

    # Reorganiza as colunas, adicionando as que faltarem
    for col in colunas:
        if col not in df.columns:
            df[col] = ""

    df = df.reindex(columns=colunas, fill_value="")
    df["generos"] = df["generos"].fillna("")
    df["tags"] = df["tags"].fillna("")
    df["nota_media"] = pd.to_numeric(df["nota_media"], errors="coerce").fillna(0)
    df["popularidade"] = pd.to_numeric(df["popularidade"], errors="coerce").fillna(0)
    return df


# Método 1 - TF-IDF

def preparar_vetores_idf(df):
    df["texto"] = df["generos"] + " " + df["tags"]
    tfidf = TfidfVectorizer(token_pattern=r"[^, ]+")
    tfidf_matrix = tfidf.fit_transform(df["texto"])
    numericos = df[["nota_media", "popularidade"]]
    scaler = MinMaxScaler()
    numericos_norm = scaler.fit_transform(numericos)
    return hstack([tfidf_matrix, numericos_norm]).tocsr()


def recomendar_idf(df, vetores, titulo, coluna_titulo="titulo_romaji", top_n=10):
    if titulo not in df[coluna_titulo].values:
        return []
    idx = df[df[coluna_titulo] == titulo].index[0]
    vetor_base = vetores[idx]
    similaridades = cosine_similarity(vetor_base, vetores)[0]
    indices = np.argsort(similaridades)[::-1][1:top_n + 1]
    return df.iloc[indices]


# ==============================
# Método 2 - Pesos (One-hot)
# ==============================
def preparar_vetores_pesos(df, peso_generos=1, peso_tags=1, peso_num=1):
    generos_vec = df["generos"].str.get_dummies(sep=", ") * peso_generos
    tags_vec = df["tags"].str.get_dummies(sep=", ") * peso_tags
    numericos = df[["nota_media", "popularidade"]]
    scaler = MinMaxScaler()
    numericos_norm = pd.DataFrame(scaler.fit_transform(numericos), columns=numericos.columns) * peso_num
    return pd.concat([generos_vec, tags_vec, numericos_norm], axis=1)


def recomendar_pesos(df, vetores, titulo, coluna_titulo="titulo_romaji", top_n=10):
    if titulo not in df[coluna_titulo].values:
        return []
    idx = df[df[coluna_titulo] == titulo].index[0]
    vetor_base = vetores.iloc[[idx]]
    similaridades = cosine_similarity(vetor_base, vetores)[0]
    indices = np.argsort(similaridades)[::-1][1:top_n + 1]
    return df.iloc[indices]



# Método 3 - Pesos + TF-IDF

def preparar_vetores_pidf(df, peso_generos=1, peso_tags=1, peso_num=1):
    tfidf_gen = TfidfVectorizer(token_pattern=r"[^, ]+")
    generos_matrix = tfidf_gen.fit_transform(df["generos"]) * peso_generos
    tfidf_tag = TfidfVectorizer(token_pattern=r"[^, ]+")
    tags_matrix = tfidf_tag.fit_transform(df["tags"]) * peso_tags
    numericos = df[["nota_media", "popularidade"]]
    scaler = MinMaxScaler()
    numericos_norm = scaler.fit_transform(numericos) * peso_num
    return hstack([generos_matrix, tags_matrix, numericos_norm]).tocsr()


def recomendar_pidf(df, vetores, titulo, coluna_titulo="titulo_romaji", top_n=10):
    if titulo not in df[coluna_titulo].values:
        return []
    idx = df[df[coluna_titulo] == titulo].index[0]
    vetor_base = vetores[idx]
    similaridades = cosine_similarity(vetor_base, vetores)[0]
    indices = np.argsort(similaridades)[::-1][1:top_n + 1]
    return df.iloc[indices]



# Streamlit App

st.set_page_config(page_title="Manga Oracle", layout="wide")
st.title("📖 Manga Oracle - Sistema de Recomendação")
with st.expander("ℹ️ Sobre o trabalho"):
    st.markdown("""
O **Manga Oracle** (tradução literal: *Oráculo dos Mangás*) é um Trabalho de Conclusão de Curso feito por **Diego Oliveira Aluizio**, com orientação do **Professor Doutor Ivan Carlos Alcantara de Oliveira**.  

O projeto tem como objetivo recomendar mangás com base em uma entrada utilizando os dados obtidos via API do site de catalogação [AniList](https://anilist.co/).  

Para mais detalhes, visite a página no [GitHub do projeto](https://github.com/DiegoAluizio/TCC-MangaOracle).
""")


df = carregar_dados()

# Escolha de idioma de busca
idioma_busca = st.radio("Buscar título em:", ["Romaji (Japonês com caracteres romanos)", "Inglês"])
coluna_titulo = "titulo_romaji" if idioma_busca == "Romaji (Japonês com caracteres romanos)" else "titulo_ingles"

# Escolha do método

metodo = st.radio("Escolha o método de recomendação:", ["TF-IDF", "Pesos", "Pesos + TF-IDF"])

with st.expander("ℹ️ Sobre os métodos"):
    st.markdown("""
O método TF-IDF é um método que seleciona características "raras" de um elemento e aumenta o peso delas para averiguar sua semelhanças com outros.

O método de Pesos funciona ao pré-determinar os pesos de um grupo de características para a comparação e análise de semelhança. No caso do Manga Oracle, os pesos foram atribuídos a três grupos de características:
                
**Gêneros:** Características descritivas amplas, como Terror, Ação e Romance.
                
**Tags:** Características descritivas mais restritas. Dizem respeito a temas e aspectos mais intrínsecos à obra como Dêmonios, Protagonista Masculino e Depressão.
                
**Variáveis numéricas:** Neste campo estão as duas variáveis de ordem numérica utilizadas como parâmetro para a recomendação: Popularidade na plataforma (medida em quantidade de listas em que o determinado mangá está presente) e Nota média na plataforma (nota média que os usuários da plataforma Anilist deram para aquele mangá).

O método Pesos + TF-IDF é uma mescla dos dois métodos anteriores.

Para mais detalhes, visite a página no [GitHub do projeto](https://github.com/seu_usuario/seu_repositorio).
""")

peso_generos, peso_tags, peso_num = 1, 1, 1
if metodo in ["Pesos", "Pesos + TF-IDF"]:
    st.subheader("⚖️ Ajuste os pesos")
    peso_generos = st.slider("Peso para Gêneros", 1, 5, 2)
    peso_tags = st.slider("Peso para Tags", 1, 5, 3)
    peso_num = st.slider("Peso para Variáveis Numéricas", 1, 5, 1)

# Escolha do mangá
titulo = st.selectbox(f"Selecione um mangá com título em {idioma_busca}:", df[coluna_titulo].dropna().unique())
top_n = st.slider("Número de recomendações:", 5, 20, 10)

if st.button("Gerar Recomendações"):
    if metodo == "TF-IDF":
        vetores = preparar_vetores_idf(df)
        recomendados = recomendar_idf(df, vetores, titulo, coluna_titulo, top_n)
    elif metodo == "Pesos":
        vetores = preparar_vetores_pesos(df, peso_generos, peso_tags, peso_num)
        recomendados = recomendar_pesos(df, vetores, titulo, coluna_titulo, top_n)
    else:
        vetores = preparar_vetores_pidf(df, peso_generos, peso_tags, peso_num)
        recomendados = recomendar_pidf(df, vetores, titulo, coluna_titulo, top_n)

    st.subheader(f"🔎 Recomendações para **{titulo}**:")

    for _, row in recomendados.iterrows():
        with st.container():
            cols = st.columns([1, 3])
            with cols[0]:
                if pd.notna(row["capa"]):
                    st.image(row["capa"], width=150)
                else:
                    st.text("Sem capa")
            with cols[1]:
                st.markdown(f"**{row['titulo_romaji']}**")
                st.write(f"📊 Nota Média no Anilist: {row['nota_media']} | ⭐ Favoritos: {row['favoritos']} | 📈 Popularidade: {row['popularidade']}")
                st.write(f"📚 Volumes: {row['volumes']} | 🖊️ Autores: {row['autores']}")
                with st.expander("📖 Sinopse"):
                    if "descricao" in row and pd.notna(row["descricao"]) and len(str(row["descricao"]).strip()) > 0:
                        st.markdown(row["descricao"], unsafe_allow_html=True)
                    else:
                        st.write("Sinopse indisponível.")
                with st.expander("📖 Gêneros"):

                    st.markdown(row["generos"])


