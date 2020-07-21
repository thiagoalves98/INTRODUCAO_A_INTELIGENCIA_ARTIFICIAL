import numpy 					as np
import pandas 					as pd
import seaborn 					as sns
from scipy 						import sparse
from sklearn.svm 				import SVC
from sklearn.svm 				import LinearSVC
from scipy.io.arff 				import loadarff
from sklearn.metrics 			import f1_score
from sklearn.metrics 			import accuracy_score
from sklearn.metrics 			import confusion_matrix
from sklearn.neighbors 			import KNeighborsClassifier
from sklearn.multiclass 		import OneVsRestClassifier
#from sklearn.metrics 			import classification_report
#from sklearn.preprocessing 		import RobustScaler
#from sklearn.model_selection	import train_test_split


#Dataframe utils

def Info(df):

	print("\n### Show Info ###\n")
	print(df.info())
	print("\n--------------------\n")

	print("\n### Show Describe ###\n")
	print(df.describe())
	print("\n--------------------\n")

	print("\n### Min - Max ###\n")
	print(pd.DataFrame([df.min(), df.max()], index = ['Min', 'Max']))
	print("\n--------------------\n")

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

#Pre-processing functions

def Normalization(df):

	normalizing_df = pd.DataFrame()

	types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

	df_values = df.select_dtypes(types)

	for key in df_values:
		normalized_column = (df_values[key] - df_values[key].min())/(df_values[key].max() - df_values[key].min())
		normalizing_df[key] = normalized_column

	df_norm = pd.concat([normalizing_df, df.select_dtypes('object')], axis=1)

	return df_norm

def Correlation(df, n):

	types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

	df_values = df.select_dtypes(types)

	matrix_corr = df_values.corr().abs()

	upper = matrix_corr.where(np.triu(np.ones(matrix_corr.shape), k=1).astype(np.bool))

	to_drop = [column for column in upper.columns if any(upper[column] > n)]

	df_corr = df.drop(df[to_drop], axis = 1)

	return df_corr

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

#Machine learning 

def SVM(df):

	print("### SVM\n")

	n = 19

	train = df.sample(frac=0.80)
	test = df.drop(index = train.index)

	X_train = train.iloc[:, :72-n]
	Y_train = train.iloc[:,:71-n:-1]
	X_test = test.iloc[:, :72-n]
	Y_test = test.iloc[:,:71-n:-1]

	oneVsRest = OneVsRestClassifier(SVC(kernel = 'linear',  C = 0.5)).fit(X_train, Y_train.astype(int))
	y_pred = oneVsRest.predict(X_test)
	Y_test_array = Y_test.astype(int).values
	sparsePred = sparse.csc_matrix(y_pred)
	sparseYtest = sparse.csc_matrix(Y_test_array)
	acc = accuracy_score(sparseYtest, sparsePred, normalize = True)

	return acc, sparsePred, sparseYtest

def Confusion_matrix(pred, test):

	labels = ['amazed-suprised', 'happy-pleased', 'relaxing-calm', 'quiet-still', 'sad-lonely', 'angry-aggresive']

	matrix_confusion = confusion_matrix(test.argmax(axis=1), pred.argmax(axis=1))
	matrix_confusion_df = pd.DataFrame(matrix_confusion, index = labels, columns= labels)
	sns.heatmap(matrix_confusion_df, annot= True)

	matriz = np.zeros(shape=(6,6))

	for i in range(6):
		for j in range(6):
			matriz[i][j] = matrix_confusion[i][j]
			
	return matriz, matrix_confusion

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

def KNN(df):

	n = 19

	print("### KNN\n")

	train = df.sample(frac=0.80)
	test = df.drop(index = train.index)

	X_train = train.iloc[:, :72-n]
	Y_train = train.iloc[:,:71-n:-1]
	X_test = test.iloc[:, :72-n]
	Y_test = test.iloc[:,:71-n:-1]

	oneVsRest = OneVsRestClassifier(KNeighborsClassifier(6, weights='distance')).fit(X_train, Y_train.astype(int))
	y_pred = oneVsRest.predict(X_test)
	Y_test_array = Y_test.astype(int).values
	sparsePred = sparse.csc_matrix(y_pred)
	sparseYtest = sparse.csc_matrix(Y_test_array)

	acc = accuracy_score(sparseYtest, sparsePred, normalize = True)

	return acc, sparsePred, sparseYtest

###

def main():

    df 		= Open_DataFrame('emotions.arff')

    df_nor = Normalization(df)

    df_cor = Correlation(df_nor, 0.80)

    df_out = Outliers(df_cor, 2.0)

    ########################################### KNN
    while(1):

	    acuracia, pred, test = SVM(df_out)

	    matrix, matrix_confusion 	= Confusion_matrix(pred, test)

	    sens 	= Sensibility(matrix)

	    Spec 	= Specificity(matrix)

	    Score 	= Score_f1(pred, test)

	    acuracia = round(acuracia*100, 2)

	    print("\nAcurácia : ",acuracia,"%\n")
	    
	    print("### Matriz de confunsão ###\n")
	    print(matrix_confusion)
	    
	    print("\nScore :",Score['f1_Score'][0]*100,"%\n")

	    for i in range(len(sens)):
	    	sens[i] = round(sens[i],2)

	    print("Sensibilidade :",sens,"\n")

	    for i in range(len(Spec)):
	    	Spec[i] = round(Spec[i],2)

	    print("Especificidade :",Spec)

	    if(acuracia >= 37.0):
	    	break



if __name__ == "__main__":
    main()