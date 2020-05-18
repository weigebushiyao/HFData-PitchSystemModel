from sklearn.decomposition import PCA

#采用pca进行数据降维,并返回降维后的数据。
def pca_decomposition(x,k=2):
    pca=PCA(n_components=k)
    pca.fit(x)
    newX=pca.fit_transform(x)
    return newX
