import numpy 					as np
import pandas 					as pd
from scipy.io.arff 				import loadarff
from sklearn.metrics 			import f1_score
from sklearn.metrics 			import accuracy_score
from sklearn.metrics 			import confusion_matrix
from sklearn.metrics 			import classification_report
from sklearn.neighbors 			import KNeighborsClassifier
from sklearn.multiclass 		import OneVsRestClassifier
from sklearn.preprocessing 		import RobustScaler
from sklearn.model_selection	import train_test_split

def Describe(df):
	print("\n###	Show Info ###\n")
	print(df.describe())
	print("\n####################\n")

def Labels(df):

    return df.columns

def Shape(df):
    
    rows 	= df.shape[0]
    columns = df.shape[1]
    
    return rows, columns

def Open_DataFrame(name):
    
    Raw_data = loadarff(name)
    df = pd.DataFrame(Raw_data[0])
    
    return df

def Missing_atr(df):
    
    n = df.isnull().sum(axis = 1)
    
    if(n.sum() == 0):
        return True
    else:
        return False

def Boxplot(df):

	df.boxplot()

def Correlation(df, n):
   
    Matriz_corr = df.corr()
    
    #Matriz_corr.style.background_gradient(cmap='coolwarm')
    
    corr_matrix = df.corr().abs()
    
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    
    to_drop = [column for column in upper.columns if any(upper[column] > n)]
    
    df.drop(df[to_drop], axis = 1, inplace=True)
    
    return df

def Histogram(df):
    
    df_bin = df[columns[72:]]
    
    df_bin = df_bin.astype(int)
    
    name_columns = Labels(df_bin)
    
    for name in name_columns:
        print("Valores: ",df_bin[name].value_counts().to_dict())
        plt.show(df_bin.hist(column=name, color='red'))
        plt.clf()
        plt.cla()
        plt.close()
        
def Outliers(df, n):
    
    name_columns = Labels(df)
    rows, column = Shape(df)
    index = []
    
    for i in range(column-6):
        
        name = name_columns[i]
        Desc = df[name].describe()
        
        #Outilier value
        Q1 = Desc[4] 
        Q3 = Desc[6]
        FIQ = Q3 - Q1
        
        l_sup = Q3 + (n*FIQ) 
        l_inf = Q1 - (n*FIQ) 
        
        for j in range(df.shape[0]):
            
            if j in index:
                continue
            
            cell_value = df[name][j]
            
            if(cell_value > l_sup) or (cell_value < l_inf):
                df.drop([j], inplace = True)
                index.append(j)
                
    return df

def KNN(df):

	rows, columns = Shape(df)

	columns = columns - 6

	#Float part
	df_num = df.iloc[:, :columns]

	#Bin part
	df_bin = df.iloc[:, columns:]
	df_bin = df_bin.astype(int)

	#Split
	X_train, X_test, Y_train, Y_test = train_test_split(df_num, df_bin, test_size=0.20, random_state=0)

	#Normalização
	rc = RobustScaler()
	X_train = rc.fit_transform(X_train)
	X_test = rc.fit_transform(X_test)

	#
	knn = KNeighborsClassifier(n_neighbors=3)

	knn.fit(X_train, Y_train)

	Y_pred = (knn.predict(X_test))

	acc = accuracy_score(Y_test, Y_pred, normalize = True)

	return acc, Y_pred, Y_test

def SVM(df):

	oneVsRest = OneVsRestClassifier(SVC(kernel = 'linear',  C = 0.5)).fit(X_train, Y_train.astype(int))
	y_pred = oneVsRest.predict(X_test)
	Y_test_array = Y_test.astype(int).values
	sparsePred = sparse.csc_matrix(y_pred)
	sparseYtest = sparse.csc_matrix(Y_test_array)
	acc = accuracy_score(sparseYtest, sparsePred, normalize = True)
	print(acc)

def Confusion_matrix(Y_pred, Y_test):

	matrix = confusion_matrix(Y_test.values.argmax(axis=1), Y_pred.argmax(axis=1))
	return matrix

def Score_f1(Y_pred, Y_test):

	f1 = pd.DataFrame([f1_score(Y_test, Y_pred, average='micro')], columns=['f1_Score']).round(2)
	return f1

def Sensibility(confusion_matrix):

	sensibility = []

	for i in range(0, 6):
		aux_var = 0
		rowSum = 0

		for j in range(0, 6):
			if(i == j):
				aux_var = confusion_matrix[i][j]

			rowSum += confusion_matrix[i][j]
		sensibility.append(aux_var/rowSum)

	return sensibility

def Specificity(confusion_matrix):

	sumDiagonal = 0
	specificity = []

	for i in range(0, 6):
		for j in range(0, 6):
			if(i == j):
				sumDiagonal += confusion_matrix[i][j]

	for i in range(0, 6):
		sumCol = 0
		ownDig = 0
		for j in range(0, 6):
			if(i == j):
				ownDig = confusion_matrix[i][j]
			sumCol += confusion_matrix[j][i]
		tn = sumDiagonal - ownDig
		specificity.append(tn/(tn + (sumCol - ownDig)))

	return specificity

def main():

    df 		= Open_DataFrame('emotions.arff')

    df_out 	= Outliers(df, 1.5)

    df_cor 	= Correlation(df_out, 0.80)
    
    acc, pred, test = KNN(df_cor)

    matrix 	= Confusion_matrix(pred, test)

    Score 	= Score_f1(pred, test)

    Sen 	= Sensibility(matrix)

    Spec 	= Specificity(matrix)

    print("\nAcurácia : ",acc*100,"%")

    print("\n###Matriz de confunsão")
    print(matrix)

    print("\nScore :",Score['f1_Score'][0]*100,"%\n")

    print("Sensibilidade :",Sen,"\n")

    print("Especificidade :",Spec)

if __name__ == "__main__":
    main()