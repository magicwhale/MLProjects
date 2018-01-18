import scipy.io as sio
import numpy as np
from sklearn import linear_model as lm
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

"""
Takes a regressor, train data and test data and predicts vals for test data
"""
def predict_vals(regressor, data_matrix, label_vector, test_matrix):
	regressor.fit(data_matrix, label_vector)
	predicted_vals = regressor.predict(test_matrix)
	# round precit values
	predicted_vals = np.rint(predicted_vals)
	# print r^2 value of regressor
	print(regressor.score(data_matrix, label_vector))
	return predicted_vals

def output_predictions(predictions, filename):
	out_file = open(filename, "w")
	out_file.write("dataid,prediction\n")
	#writer = csv.writer(out_file)
	for index, element in enumerate(predictions):
		try:
			out_file.write(str(index + 1) + "," + str(int(element[0])))
		except:
			out_file.write(str(index + 1) + "," + str(int(element)))
		out_file.write("\n")


def predict_avgs(data_matrix, label_vector, test_matrix, params):
	predicted_vals = 0
	start = 0
	end = int(data_matrix.shape[0]/50)
	for i in range(50):
		regressor = SVR(C = 2.2, kernel = 'rbf', gamma = 0.5)
		regressor.fit(data_matrix[start:end, :], label_vector[start:end])
		print("finished fit\n")
		predicted_vals += np.rint(regressor.predict(test_matrix))
		start = end
		end += int(data_matrix.shape[0]/50)
		print("finished predict\n")

	predicted_vals = predicted_vals/50
	return predicted_vals

if __name__ == "__main__":
	# load data
	data = sio.loadmat('MSdata.mat')
	data_matrix = data['trainx']
	label_vector = data['trainy']
	test_matrix = data['testx']
	# Change data type for double precision
	data_matrix = data_matrix.astype(float)
	label_vector = label_vector.astype(float)
	test_matrix = test_matrix.astype(float)

	# OLS regressor
	#output_predictions(predict_vals(lm.LinearRegression(), data_matrix, label_vector, test_matrix), "ols_predictions.txt")

	# Ridge regressor (with cross validation to select lambda weight for regularization term)
	#output_predictions(predict_vals(lm.RidgeCV(), data_matrix, label_vector, test_matrix), "ridgeCV_predictions.txt")

	# Lasso regressor with cv
	#output_predictions(predict_vals(lm.LassoCV(), data_matrix, label_vector.ravel(), test_matrix), "lassoCV_predictions.txt")
	
	# SVR Partition
	#parameter_grid = [{'kernel' : ['rbf'], 'C' : [0.1, 0.5, 1, 2.2], 'gamma' : [0.1, 0.5, 1, 3, 6]}]
	#regressor = GridSearchCV(SVR(), parameter_grid, cv = 5)
	#regressor.fit(data_matrix[0:int(data_matrix.shape[0]/50), :], label_vector[:int(data_matrix.shape[0]/50)].ravel())
	#params = regressor.best_params_, regressor.best_score_
	#print("Finished GRID SEARCH\n")
	#print(str(params[0]['C']) + "\n")
	#print(str(params[0]['gamma']) + "\n")
	predicted_vals = predict_avgs(data_matrix, label_vector.ravel(), test_matrix, dict())
	output_predictions(predicted_vals)
	


	"""parameter_grid = [{'kernel' : ['rbf'], 'alpha' : [0.1, 0.5, 1, 2, 3], 'gamma' : [0.1, 0.5, 1, 3, 6, 10]}]
	regressor = GridSearchCV(KernelRidge(), parameter_grid, cv = 5)
	regressor.fit(data_matrix, label_vector)
	regressor = KernelRidge(alpha = params[0]['alpha'], kernel = 'kernel', gamma = params[0]['gamma'])"""
	#output_predictions(predict_vals(SVR(), data_matrix, label_vector.ravel(), test_matrix), "SVR_predictions.txt")

